#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class SessionTree:
    """
    Creates a model for session dir.
    root/
        checkpoints/
        logs/
        params.json
        dataset_indices.pkl
    """
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)

        self.root.mkdir(exist_ok=True, parents=True)
        self.checkpoints.mkdir(exist_ok=True, parents=True)
        self.logs.mkdir(exist_ok=True, parents=True)

    @property
    def params(self):
        return self.root / "params.json"

    @property
    def checkpoints(self):
        return self.root / "checkpoints"
    
    @property
    def indices(self):
        return self.root / "dataset_indices.pkl"
    
    @property
    def logs(self):
        return self.root / "logs/"
    
    @property
    def last_ckpt(self):
        return self.checkpoints / "last.ckpt"

    @property
    def best_ckpt(self):
        if (self.checkpoints / "best.ckpt").exists():
            self.checkpoints / "best.ckpt"
        return self.last_ckpt
            
    def dump_indices(self, indices):
        with open(self.indices, 'wb') as pk:
            pickle.dump(indices, pk)



if __name__ == '__main__':
    pass
