#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import inspect
from dataclasses import dataclass, asdict

__all__ = ["Config"]

@dataclass
class Config:
    def isolate(self, config):
        specifics = inspect.signature(config).parameters
        my_specifics = {k: v for k, v in asdict(self).items() if k in specifics}
        return config(**my_specifics)

    def to_json(self, filename):
        config = json.dumps(asdict(self), indent=2)
        with open(filename, 'w') as f:
            f.write(config)
    
    @classmethod
    def from_json(cls, filename):
        with open(filename, 'r') as f:
            js = json.loads(f.read())
        config = cls(**js)
        return config

if __name__ == '__main__':
    pass