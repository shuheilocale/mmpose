# Copyright (c) OpenMMLab. All rights reserved.
<<<<<<< HEAD
from .ae_loss import AssociativeEmbeddingLoss
from .classification_loss import BCELoss, JSDiscretLoss, KLDiscretLoss
from .heatmap_loss import (AdaptiveWingLoss, KeypointMSELoss,
                           KeypointOHKMMSELoss)
from .loss_wrappers import CombinedLoss, MultipleLossWrapper
=======
from .classfication_loss import BCELoss
from .heatmap_loss import AdaptiveWingLoss, FocalHeatmapLoss
from .mesh_loss import GANLoss, MeshLoss
from .mse_loss import JointsMSELoss, JointsOHKMMSELoss
from .multi_loss_factory import AELoss, HeatmapLoss, MultiLossFactory
>>>>>>> fd7ff851 (Add CID to mmpose (#1604))
from .regression_loss import (BoneLoss, L1Loss, MPJPELoss, MSELoss, RLELoss,
                              SemiSupervisionLoss, SmoothL1Loss,
                              SoftWeightSmoothL1Loss, SoftWingLoss, WingLoss)

__all__ = [
<<<<<<< HEAD
<<<<<<< HEAD
    'KeypointMSELoss', 'KeypointOHKMMSELoss', 'SmoothL1Loss', 'WingLoss',
    'MPJPELoss', 'MSELoss', 'L1Loss', 'BCELoss', 'BoneLoss',
    'SemiSupervisionLoss', 'SoftWingLoss', 'AdaptiveWingLoss', 'RLELoss',
    'KLDiscretLoss', 'MultipleLossWrapper', 'JSDiscretLoss', 'CombinedLoss',
    'AssociativeEmbeddingLoss', 'SoftWeightSmoothL1Loss'
=======
    'JointsMSELoss',
    'JointsOHKMMSELoss',
    'HeatmapLoss',
    'AELoss',
    'MultiLossFactory',
    'MeshLoss',
    'GANLoss',
    'SmoothL1Loss',
    'WingLoss',
    'MPJPELoss',
    'MSELoss',
    'L1Loss',
    'BCELoss',
    'BoneLoss',
    'SemiSupervisionLoss',
    'SoftWingLoss',
    'AdaptiveWingLoss',
    'RLELoss',
    'FocalHeatmapLoss',
>>>>>>> fd7ff851 (Add CID to mmpose (#1604))
=======
    'JointsMSELoss', 'JointsOHKMMSELoss', 'HeatmapLoss', 'AELoss',
    'MultiLossFactory', 'MeshLoss', 'GANLoss', 'SmoothL1Loss', 'WingLoss',
    'MPJPELoss', 'MSELoss', 'L1Loss', 'BCELoss', 'BoneLoss',
    'SemiSupervisionLoss', 'SoftWingLoss', 'AdaptiveWingLoss', 'RLELoss',
    'SoftWeightSmoothL1Loss', 'FocalHeatmapLoss'
>>>>>>> 3fbd7b69 ([Feature] support DEKR (#1693))
]
