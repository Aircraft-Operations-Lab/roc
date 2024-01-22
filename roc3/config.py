#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import os
from os.path import expanduser
home = expanduser("~")

ROC_CFG_PATH = home + '/.roc_cfg'

default_options = {
    'bada4_path': os.path.join(home, 'BADA4.1'),
    }

class Config(dict):
    def __init__(self):
        dict.__init__(self)
        self.update(default_options)
        try:
            with open(ROC_CFG_PATH, 'r') as f:
                self.update(json.loads(f.read()))
        except IOError:
            self.save()
    def save(self):
        with open(ROC_CFG_PATH, 'w') as f:
            f.write(json.dumps(dict(self)))        