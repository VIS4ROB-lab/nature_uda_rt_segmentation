import pathlib
from typing import Union, Tuple, Sequence, Optional, Dict, List
from random import choice, choices
import time
import math

import numpy as np
import mmcv

from uda_helpers.dataset.copypaste import CopyPasteAug as _CopyPaste

from ..builder import PIPELINES


@PIPELINES.register_module()
class CopyPaste(_CopyPaste):
    """
    Copy and Paste.

    Inherited from uda_helpers.dataset.copypaste.CopyPasteAug. Please check the package for details
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
