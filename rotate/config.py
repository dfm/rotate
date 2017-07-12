# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os

__all__ = ["ROTATE_DATA_PATH"]

ROTATE_DATA_PATH = os.environ.get("ROTATE_DATA_PATH",
                                  os.path.expanduser(
                                      os.path.join("~", "rotate")))
