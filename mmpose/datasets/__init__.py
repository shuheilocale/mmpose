# Copyright (c) OpenMMLab. All rights reserved.
from .builder import build_dataset
from .dataset_wrappers import CombinedDataset
from .datasets import *  # noqa
from .samplers import MultiSourceSampler
from .transforms import *  # noqa

<<<<<<< HEAD
__all__ = ['build_dataset', 'CombinedDataset', 'MultiSourceSampler']
=======
from .datasets import (  # isort:skip
    AnimalATRWDataset, AnimalFlyDataset, AnimalHorse10Dataset,
    AnimalLocustDataset, AnimalMacaqueDataset, AnimalPoseDataset,
    AnimalZebraDataset, Body3DH36MDataset, BottomUpAicDataset,
    BottomUpCocoDataset, BottomUpCocoWholeBodyDataset,
    BottomUpCrowdPoseDataset, BottomUpMhpDataset, DeepFashionDataset,
    Face300WDataset, FaceAFLWDataset, FaceCocoWholeBodyDataset,
    FaceCOFWDataset, FaceWFLWDataset, FreiHandDataset,
    HandCocoWholeBodyDataset, InterHand2DDataset, InterHand3DDataset,
    MeshAdversarialDataset, MeshH36MDataset, MeshMixDataset, MoshDataset,
    OneHand10KDataset, PanopticDataset, TopDownAicDataset, TopDownCocoDataset,
    TopDownCocoWholeBodyDataset, TopDownCrowdPoseDataset,
    TopDownFreiHandDataset, TopDownH36MDataset, TopDownJhmdbDataset,
    TopDownMhpDataset, TopDownMpiiDataset, TopDownMpiiTrbDataset,
    TopDownOCHumanDataset, TopDownOneHand10KDataset, TopDownPanopticDataset,
    TopDownPoseTrack18Dataset, TopDownPoseTrack18VideoDataset,
    Body3DMviewDirectPanopticDataset, Body3DMviewDirectShelfDataset,
    Body3DMviewDirectCampusDataset, NVGestureDataset)

__all__ = [
    'TopDownCocoDataset', 'BottomUpCocoDataset', 'BottomUpMhpDataset',
    'BottomUpAicDataset', 'BottomUpCocoWholeBodyDataset', 'TopDownMpiiDataset',
    'TopDownMpiiTrbDataset', 'OneHand10KDataset', 'PanopticDataset',
    'HandCocoWholeBodyDataset', 'FreiHandDataset', 'InterHand2DDataset',
    'InterHand3DDataset', 'TopDownOCHumanDataset', 'TopDownAicDataset',
    'TopDownCocoWholeBodyDataset', 'MeshH36MDataset', 'MeshMixDataset',
    'MoshDataset', 'MeshAdversarialDataset', 'TopDownCrowdPoseDataset',
    'BottomUpCrowdPoseDataset', 'TopDownFreiHandDataset',
    'TopDownOneHand10KDataset', 'TopDownPanopticDataset',
    'TopDownPoseTrack18Dataset', 'TopDownJhmdbDataset', 'TopDownMhpDataset',
    'DeepFashionDataset', 'Face300WDataset', 'FaceAFLWDataset',
    'FaceWFLWDataset', 'FaceCOFWDataset', 'FaceCocoWholeBodyDataset',
    'Body3DH36MDataset', 'AnimalHorse10Dataset', 'AnimalMacaqueDataset',
    'AnimalFlyDataset', 'AnimalLocustDataset', 'AnimalZebraDataset',
    'AnimalATRWDataset', 'AnimalPoseDataset', 'TopDownH36MDataset',
    'TopDownPoseTrack18VideoDataset', 'build_dataloader', 'build_dataset',
    'Compose', 'DistributedSampler', 'DATASETS', 'PIPELINES', 'DatasetInfo',
    'Body3DMviewDirectPanopticDataset', 'Body3DMviewDirectShelfDataset',
    'Body3DMviewDirectCampusDataset', 'NVGestureDataset'
]
>>>>>>> d3c17d5e ([Feature] Gesture recognition algorithm MTUT on NVGesture dataset (#1380))
