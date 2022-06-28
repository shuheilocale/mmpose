# Copyright (c) OpenMMLab. All rights reserved.
from .fmap_proc_neck import FeatureMapProcessor
from .fpn import FPN
from .gap_neck import GlobalAveragePooling
from .posewarper_neck import PoseWarperNeck
from .tcformer_mta_neck import MTA

<<<<<<< HEAD
__all__ = [
    'GlobalAveragePooling', 'PoseWarperNeck', 'FPN', 'FeatureMapProcessor'
]
=======
__all__ = ['GlobalAveragePooling', 'PoseWarperNeck', 'FPN', 'MTA']
>>>>>>> a8c23bf0 (add tcformer (#1447))
