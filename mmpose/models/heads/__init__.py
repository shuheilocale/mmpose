# Copyright (c) OpenMMLab. All rights reserved.
<<<<<<< HEAD
from .base_head import BaseHead
from .coord_cls_heads import RTMCCHead, SimCCHead
from .heatmap_heads import (AssociativeEmbeddingHead, CIDHead, CPMHead,
                            HeatmapHead, MSPNHead, ViPNASHead)
from .hybrid_heads import DEKRHead
from .regression_heads import (DSNTHead, IntegralRegressionHead,
                               RegressionHead, RLEHead)

__all__ = [
    'BaseHead', 'HeatmapHead', 'CPMHead', 'MSPNHead', 'ViPNASHead',
    'RegressionHead', 'IntegralRegressionHead', 'SimCCHead', 'RLEHead',
    'DSNTHead', 'AssociativeEmbeddingHead', 'DEKRHead', 'CIDHead', 'RTMCCHead'
=======
from .ae_higher_resolution_head import AEHigherResolutionHead
from .ae_multi_stage_head import AEMultiStageHead
from .ae_simple_head import AESimpleHead
from .cid_head import CIDHead
from .deconv_head import DeconvHead
from .deeppose_regression_head import DeepposeRegressionHead
from .dekr_head import DEKRHead
from .hmr_head import HMRMeshHead
from .interhand_3d_head import Interhand3DHead
from .mtut_head import MultiModalSSAHead
from .temporal_regression_head import TemporalRegressionHead
from .topdown_heatmap_base_head import TopdownHeatmapBaseHead
from .topdown_heatmap_multi_stage_head import (TopdownHeatmapMSMUHead,
                                               TopdownHeatmapMultiStageHead)
from .topdown_heatmap_simple_head import TopdownHeatmapSimpleHead
from .vipnas_heatmap_simple_head import ViPNASHeatmapSimpleHead
from .voxelpose_head import CuboidCenterHead, CuboidPoseHead

__all__ = [
    'TopdownHeatmapSimpleHead', 'TopdownHeatmapMultiStageHead',
    'TopdownHeatmapMSMUHead', 'TopdownHeatmapBaseHead',
    'AEHigherResolutionHead', 'AESimpleHead', 'AEMultiStageHead', 'CIDHead',
    'DeepposeRegressionHead', 'TemporalRegressionHead', 'Interhand3DHead',
    'HMRMeshHead', 'DeconvHead', 'ViPNASHeatmapSimpleHead', 'CuboidCenterHead',
<<<<<<< HEAD
    'CuboidPoseHead', 'MultiModalSSAHead'
>>>>>>> d3c17d5e ([Feature] Gesture recognition algorithm MTUT on NVGesture dataset (#1380))
=======
    'CuboidPoseHead', 'MultiModalSSAHead', 'DEKRHead'
>>>>>>> 3fbd7b69 ([Feature] support DEKR (#1693))
]
