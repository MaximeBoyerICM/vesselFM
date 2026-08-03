import json
import os
from pathlib import Path


class Checkpoint:
    """Tracks which blocks of which images have already been processed."""

    def __init__(self, path, active=True):
        self.path = Path(path)
        self.active = active
        self.data = self._load() if self.active else {}

    def _load(self):
        if self.path.exists():
            with open(self.path, 'r') as f:
                return json.load(f)
        return {}

    def _save(self):
        if not self.active:
            return
        tmp = self.path.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump(self.data, f)
        os.replace(tmp, self.path)

    def is_block_done(self, image_name, block_idx):
        if not self.active:
            return False
        return block_idx in self.data.get(image_name, {}).get('done_blocks', [])

    def is_image_done(self, image_name, total_blocks):
        if not self.active:
            return False
        entry = self.data.get(image_name)
        return bool(entry) and entry.get('total_blocks') == total_blocks \
            and len(entry.get('done_blocks', [])) == total_blocks

    def mark_block_done(self, image_name, block_idx, total_blocks):
        if not self.active:
            return
        entry = self.data.setdefault(image_name, {'done_blocks': [], 'total_blocks': total_blocks})
        entry['total_blocks'] = total_blocks
        if block_idx not in entry['done_blocks']:
            entry['done_blocks'].append(block_idx)
        self._save()