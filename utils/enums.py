from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from enum import Enum

class Mode(Enum):
    TRAIN   = 1
    EVAL    = 2
    PREDICT = 3
