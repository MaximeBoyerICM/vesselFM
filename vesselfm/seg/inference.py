""" Script to perform inference with vesselFM."""
import os
import sys
import logging
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
import hydra
import numpy as np
from huggingface_hub import hf_hub_download
from monai.inferers import SlidingWindowInfererAdapt
from skimage.morphology import remove_small_objects
from skimage.exposure import equalize_hist

from vesselfm.seg.utils.data import generate_transforms
from vesselfm.seg.utils.io import determine_reader_writer
from vesselfm.seg.utils.evaluation import Evaluator, calculate_mean_metrics

from custom_array import Array

path = os.path.join(os.path.expanduser("~"), "code", "ClearMap3")
sys.path.insert(0, path)
import ClearMap.ParallelProcessing.BlockProcessing as blkp
import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as array_processing
import ClearMap.IO.IO as cm_io

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def load_model(cfg, device):
    try:
        logger.info(f"Loading model from {cfg.ckpt_path}.")
        ckpt = torch.load(Path(cfg.ckpt_path), map_location=device, weights_only=True)
    except:
        logger.info(f"Loading model from Hugging Face.")
        hf_hub_download(repo_id='bwittmann/vesselFM', filename='meta.yaml') # required to track downloads
        ckpt = torch.load(
            hf_hub_download(repo_id='bwittmann/vesselFM', filename='vesselFM_base.pt'),
            map_location=device, weights_only=True
        )

    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(ckpt)
    return model

def get_paths(cfg):
    image_paths = list(Path(cfg.image_path).iterdir())
    if cfg.mask_path:
        mask_paths = [Path(cfg.mask_path) / f"{p.name}" for p in image_paths]
        assert all(
            mask_path.exists() for mask_path in mask_paths
        ), "All mask paths must exist mask name has to be the same as the image name."
    else:
        mask_paths = None
    return image_paths, mask_paths

def resample(image, factor=None, target_shape=None):
    if factor == 1:
        return image
    
    if target_shape:
        _, _, new_d, new_h, new_w = target_shape
    else:
        _, _, d, h, w = image.shape
        new_d, new_h, new_w = int(round(d / factor)), int(round(h / factor)), int(round(w / factor))
    return F.interpolate(image, size=(new_d, new_h, new_w), mode="trilinear", align_corners=False)

@hydra.main(config_path="configs", config_name="inference", version_base="1.3.2")
def main(cfg):
    # seed libraries
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # set device
    logger.info(f"Using device {cfg.device}.")
    device = cfg.device

    # load model and ckpt
    model = load_model(cfg, device)
    model.to(device)
    model.eval()

    # init pre-processing transforms
    transforms = generate_transforms(cfg.transforms_config)

    # i/o
    output_folder = Path(cfg.output_folder)
    output_folder.mkdir(exist_ok=True)

    image_paths, mask_paths = get_paths(cfg)
    logger.info(f"Found {len(image_paths)} images in {cfg.image_path}.")

    file_ending = (cfg.image_file_ending if cfg.image_file_ending else image_paths[0].suffix)

    # init sliding window inferer
    logger.debug(f"Sliding window patch size: {cfg.patch_size}")
    logger.debug(f"Sliding window batch size: {cfg.batch_size}.")
    logger.debug(f"Sliding window overlap: {cfg.overlap}.")
    inferer = SlidingWindowInfererAdapt(
        roi_size=cfg.patch_size, sw_batch_size=cfg.batch_size, overlap=cfg.overlap, 
        mode=cfg.mode, sigma_scale=cfg.sigma_scale, padding_mode=cfg.padding_mode,
        sw_device='cuda', device='cpu', progress=True
    )


    from utils.checkpoint import Checkpoint
    # loop over images
    with torch.no_grad():
        checkpoint = Checkpoint(output_folder / 'checkpoint.json', active=True)
        for idx, image_path in enumerate(image_paths):
            image_name = image_path.name
            array = Array(image_path)
            source = cm_io.as_source(array.source)
            original_shape = source.shape
            logger.info(f'Processing {array.source.name} of shape: {original_shape}')

            sink, sink_shape = array_processing.initialize_sink(sink=output_folder /
                                                                f"{image_path.name.split('.')[0]}_{cfg.file_app}.npy",
                                                                shape=original_shape, dtype=np.uint8,
                                                                return_buffer=False, return_shape=True)
            patch_to_block_ratio = cfg.blocking.patch_to_block_ratio
            block_size = patch_to_block_ratio*cfg.patch_size[0]

            logger.info(f'Splitting in blocks...')
            blocks = blkp.split_into_blocks(source, processes=16,
                                            axes=cfg.blocking.axes,
                                            size_max=block_size, size_min=block_size,
                                            overlap=cfg.blocking.overlap)

            total_blocks = len(blocks)

            if checkpoint.is_image_done(image_name, total_blocks):
                logger.info(f'{image_name} already fully processed, skipping.')
                continue

            logger.info(f'Splitted. Starting prediction.')
            for i, block in enumerate(blocks):

                if checkpoint.is_block_done(image_name, i):
                    logger.info(f'Block {i}/{total_blocks} already done, skipping.')
                    continue

                image = block.array.astype(np.float32)
                image = transforms(image)[None]  # WARNING may be a bottleneck
                image = image.to(dtype=torch.float32)
                block_shape = image.shape
                logger.info(f"Block {i}/{len(blocks)} - {image.shape}")
                preds = []  # average over test time augmentations

                for scale in cfg.tta.scales:
                    # apply test time augmentation
                    if cfg.tta.invert:
                        image = 1 - image if image.mean() > cfg.tta.invert_mean_thresh else image

                    # WARNING not tested yet. could cause issues
                    if cfg.tta.equalize_hist:
                        image_np = image.cpu().squeeze().numpy()
                        image_equal_hist_np = equalize_hist(image_np, nbins=cfg.tta.hist_bins)
                        image = torch.from_numpy(image_equal_hist_np).to(image.device)[None][None]

                    image_resampled = resample(image, factor=scale)
                    logger.info(f'Running inference patch-wise | scale={scale}')
                    logits = inferer(image_resampled, model)
                    logits = resample(logits, target_shape=block_shape)
                    preds.append(logits.cpu().squeeze())
                    logger.info(f'Inference for scale={scale} completed')

                del image, image_resampled, logits
                # merging
                logger.info(f'Block {i} | Stacking scales')
                if cfg.merging.max:
                    pred = torch.stack(preds).max(dim=0)[0].sigmoid()
                else:
                    pred = torch.stack(preds).mean(dim=0).sigmoid()
                pred_thresh = (pred > cfg.merging.threshold).numpy()
                del pred

                # post-processing
                logger.info(f'Block {i} | Postprocessing')
                if cfg.post.apply:
                    pred_thresh = remove_small_objects(
                        pred_thresh, min_size=cfg.post.small_objects_min_size,
                        connectivity=cfg.post.small_objects_connectivity
                    )

                # save final pred
                # sink = output_folder / f"{image_path.name.split('.')[0]}_{cfg.file_app}_pred.{file_ending}"
                # save_writer.write_seg(pred_thresh.astype(np.uint8), sink)

                sink_slicing = block.slicing
                result_slicing = tuple(slice(None, min(ss, rs)) for ss, rs in zip(sink_shape, pred_thresh.shape[0:]))
                sink_slicing = blkp.blk.slc.sliced_slicing(result_slicing, sink_slicing, sink_shape)
                result_slicing = (0, 0) + result_slicing
                sink[sink_slicing] = pred_thresh.astype(bool)

                if hasattr(sink, 'flush'):
                    sink.flush()

                checkpoint.mark_block_done(image_name, i, total_blocks)
if __name__ == "__main__":
    main()
