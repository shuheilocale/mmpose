# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from PIL import Image

from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.models.builder import build_pose_estimator
from mmpose.structures import PoseDataSample
from mmpose.structures.bbox import bbox_xywh2xyxy


def dataset_meta_from_config(config: Config,
                             dataset_mode: str = 'train') -> Optional[dict]:
    """Get dataset metainfo from the model config.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        dataset_mode (str): Specify the dataset of which to get the metainfo.
            Options are ``'train'``, ``'val'`` and ``'test'``. Defaults to
            ``'train'``

    Returns:
        dict, optional: The dataset metainfo. See
        ``mmpose.datasets.datasets.utils.parse_pose_metainfo`` for details.
        Return ``None`` if failing to get dataset metainfo from the config.
    """
    try:
        if dataset_mode == 'train':
            dataset_cfg = config.train_dataloader.dataset
        elif dataset_mode == 'val':
            dataset_cfg = config.val_dataloader.dataset
        elif dataset_mode == 'test':
            dataset_cfg = config.test_dataloader.dataset
        else:
            raise ValueError(
                f'Invalid dataset {dataset_mode} to get metainfo. '
                'Should be one of "train", "val", or "test".')

        if 'metainfo' in dataset_cfg:
            metainfo = dataset_cfg.metainfo
        else:
            import mmpose.datasets.datasets  # noqa: F401, F403
            from mmpose.registry import DATASETS

            dataset_class = DATASETS.get(dataset_cfg.type)
            metainfo = dataset_class.METAINFO

        metainfo = parse_pose_metainfo(metainfo)

    except AttributeError:
        metainfo = None

    return metainfo


def init_model(config: Union[str, Path, Config],
               checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               cfg_options: Optional[dict] = None) -> nn.Module:
    """Initialize a pose estimator from a config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights. Defaults to ``None``
        device (str): The device where the anchors will be put on.
            Defaults to ``'cuda:0'``.
        cfg_options (dict, optional): Options to override some settings in
            the used config. Defaults to ``None``

    Returns:
        nn.Module: The constructed pose estimator.
    """

    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None
    config.model.train_cfg = None

    # register all modules in mmpose into the registries
    init_default_scope(config.get('default_scope', 'mmpose'))

    model = build_pose_estimator(config.model)
    model = revert_sync_batchnorm(model)
    # get dataset_meta in this priority: checkpoint > config > default (COCO)
    dataset_meta = None

    if checkpoint is not None:
        ckpt = load_checkpoint(model, checkpoint, map_location='cpu')

        if 'dataset_meta' in ckpt.get('meta', {}):
            # checkpoint from mmpose 1.x
            dataset_meta = ckpt['meta']['dataset_meta']

    if dataset_meta is None:
        dataset_meta = dataset_meta_from_config(config, dataset_mode='train')

    if dataset_meta is None:
        warnings.simplefilter('once')
        warnings.warn('Can not load dataset_meta from the checkpoint or the '
                      'model config. Use COCO metainfo by default.')
        dataset_meta = parse_pose_metainfo(
            dict(from_file='configs/_base_/datasets/coco.py'))

    model.dataset_meta = dataset_meta

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_topdown(model: nn.Module,
                      img: Union[np.ndarray, str],
                      bboxes: Optional[Union[List, np.ndarray]] = None,
                      bbox_format: str = 'xyxy') -> List[PoseDataSample]:
    """Inference image with a top-down pose estimator.

    Args:
        model (nn.Module): The top-down pose estimator
        img (np.ndarray | str): The loaded image or image file to inference
        bboxes (np.ndarray, optional): The bboxes in shape (N, 4), each row
            represents a bbox. If not given, the entire image will be regarded
            as a single bbox area. Defaults to ``None``
        bbox_format (str): The bbox format indicator. Options are ``'xywh'``
            and ``'xyxy'``. Defaults to ``'xyxy'``

    Returns:
        List[:obj:`PoseDataSample`]: The inference results. Specifically, the
        predicted keypoints and scores are saved at
        ``data_sample.pred_instances.keypoints`` and
        ``data_sample.pred_instances.keypoint_scores``.
    """
    init_default_scope(model.cfg.get('default_scope', 'mmpose'))
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    cfg = model.cfg
    device = next(model.parameters()).device
    if device.type == 'cpu':
        device = -1

    if use_multi_frames:
        assert 'frame_weight_test' in cfg.data.test.data_cfg
        # use multi frames for inference
        # the number of input frames must equal to frame weight in the config
        assert len(imgs_or_paths) == len(
            cfg.data.test.data_cfg.frame_weight_test)

    # build the data pipeline
    _test_pipeline = copy.deepcopy(cfg.test_pipeline)

    has_bbox_xywh2cs = False
    for transform in _test_pipeline:
        if transform['type'] == 'TopDownGetBboxCenterScale':
            has_bbox_xywh2cs = True
            break
    if not has_bbox_xywh2cs:
        _test_pipeline.insert(
            0, dict(type='TopDownGetBboxCenterScale', padding=1.25))
    test_pipeline = Compose(_test_pipeline)
    _pipeline_gpu_speedup(test_pipeline, next(model.parameters()).device)

    assert len(bboxes[0]) in [4, 5]

    if dataset_info is not None:
        dataset_name = dataset_info.dataset_name
        flip_pairs = dataset_info.flip_pairs
    else:
        warnings.warn(
            'dataset is deprecated.'
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        # TODO: These will be removed in the later versions.
        if dataset in ('TopDownCocoDataset', 'TopDownOCHumanDataset',
                       'AnimalMacaqueDataset'):
            flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                          [13, 14], [15, 16]]
        elif dataset == 'TopDownCocoWholeBodyDataset':
            body = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                    [13, 14], [15, 16]]
            foot = [[17, 20], [18, 21], [19, 22]]

            face = [[23, 39], [24, 38], [25, 37], [26, 36], [27, 35], [28, 34],
                    [29, 33], [30, 32], [40, 49], [41, 48], [42, 47], [43, 46],
                    [44, 45], [54, 58], [55, 57], [59, 68], [60, 67], [61, 66],
                    [62, 65], [63, 70], [64, 69], [71, 77], [72, 76], [73, 75],
                    [78, 82], [79, 81], [83, 87], [84, 86], [88, 90]]

            hand = [[91, 112], [92, 113], [93, 114], [94, 115], [95, 116],
                    [96, 117], [97, 118], [98, 119], [99, 120], [100, 121],
                    [101, 122], [102, 123], [103, 124], [104, 125], [105, 126],
                    [106, 127], [107, 128], [108, 129], [109, 130], [110, 131],
                    [111, 132]]
            flip_pairs = body + foot + face + hand
        elif dataset == 'TopDownAicDataset':
            flip_pairs = [[0, 3], [1, 4], [2, 5], [6, 9], [7, 10], [8, 11]]
        elif dataset == 'TopDownMpiiDataset':
            flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        elif dataset == 'TopDownMpiiTrbDataset':
            flip_pairs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11],
                          [14, 15], [16, 22], [28, 34], [17, 23], [29, 35],
                          [18, 24], [30, 36], [19, 25], [31, 37], [20, 26],
                          [32, 38], [21, 27], [33, 39]]
        elif dataset in ('OneHand10KDataset', 'FreiHandDataset',
                         'PanopticDataset', 'InterHand2DDataset'):
            flip_pairs = []
        elif dataset in 'Face300WDataset':
            flip_pairs = [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11],
                          [6, 10], [7, 9], [17, 26], [18, 25], [19, 24],
                          [20, 23], [21, 22], [31, 35], [32, 34], [36, 45],
                          [37, 44], [38, 43], [39, 42], [40, 47], [41, 46],
                          [48, 54], [49, 53], [50, 52], [61, 63], [60, 64],
                          [67, 65], [58, 56], [59, 55]]

        elif dataset in 'FaceAFLWDataset':
            flip_pairs = [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [8, 9],
                          [12, 14], [15, 17]]

        elif dataset in 'FaceCOFWDataset':
            flip_pairs = [[0, 1], [4, 6], [2, 3], [5, 7], [8, 9], [10, 11],
                          [12, 14], [16, 17], [13, 15], [18, 19], [22, 23]]

        elif dataset in 'FaceWFLWDataset':
            flip_pairs = [[0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27],
                          [6, 26], [7, 25], [8, 24], [9, 23], [10, 22],
                          [11, 21], [12, 20], [13, 19], [14, 18], [15, 17],
                          [33, 46], [34, 45], [35, 44], [36, 43], [37, 42],
                          [38, 50], [39, 49], [40, 48], [41, 47], [60, 72],
                          [61, 71], [62, 70], [63, 69], [64, 68], [65, 75],
                          [66, 74], [67, 73], [55, 59], [56, 58], [76, 82],
                          [77, 81], [78, 80], [87, 83], [86, 84], [88, 92],
                          [89, 91], [95, 93], [96, 97]]

        elif dataset in 'AnimalFlyDataset':
            flip_pairs = [[1, 2], [6, 18], [7, 19], [8, 20], [9, 21], [10, 22],
                          [11, 23], [12, 24], [13, 25], [14, 26], [15, 27],
                          [16, 28], [17, 29], [30, 31]]
        elif dataset in 'AnimalHorse10Dataset':
            flip_pairs = []

        elif dataset in 'AnimalLocustDataset':
            flip_pairs = [[5, 20], [6, 21], [7, 22], [8, 23], [9, 24],
                          [10, 25], [11, 26], [12, 27], [13, 28], [14, 29],
                          [15, 30], [16, 31], [17, 32], [18, 33], [19, 34]]

        elif dataset in 'AnimalZebraDataset':
            flip_pairs = [[3, 4], [5, 6]]

        elif dataset in 'AnimalPoseDataset':
            flip_pairs = [[0, 1], [2, 3], [8, 9], [10, 11], [12, 13], [14, 15],
                          [16, 17], [18, 19]]
        else:
            h, w = img.shape[:2]

        bboxes = np.array([[0, 0, w, h]], dtype=np.float32)
    else:
        if isinstance(bboxes, list):
            bboxes = np.array(bboxes)

        assert bbox_format in {'xyxy', 'xywh'}, \
            f'Invalid bbox_format "{bbox_format}".'

        if bbox_format == 'xywh':
            bboxes = bbox_xywh2xyxy(bboxes)

    # construct batch data samples
    data_list = []
    for bbox in bboxes:
        if isinstance(img, str):
            data_info = dict(img_path=img)
        else:
            data_info = dict(img=img)
        data_info['bbox'] = bbox[None]  # shape (1, 4)
        data_info['bbox_score'] = np.ones(1, dtype=np.float32)  # shape (1,)
        data_info.update(model.dataset_meta)
        data_list.append(pipeline(data_info))

    if data_list:
        # collate data list into a batch, which is a dict with following keys:
        # batch['inputs']: a list of input images
        # batch['data_samples']: a list of :obj:`PoseDataSample`
        batch = pseudo_collate(data_list)
        with torch.no_grad():
            results = model.test_step(batch)
    else:
        results = []

    return results


def inference_bottomup(model: nn.Module, img: Union[np.ndarray, str]):
    """Inference image with a bottom-up pose estimator.

    Args:
        model (nn.Module): The bottom-up pose estimator
        img (np.ndarray | str): The loaded image or image file to inference

    Returns:
        List[:obj:`PoseDataSample`]: The inference results. Specifically, the
        predicted keypoints and scores are saved at
        ``data_sample.pred_instances.keypoints`` and
        ``data_sample.pred_instances.keypoint_scores``.
    """
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    # prepare data batch
    if isinstance(img, str):
        data_info = dict(img_path=img)
    else:
        data_info = dict(img=img)
    data_info.update(model.dataset_meta)
    data = pipeline(data_info)
    batch = pseudo_collate([data])

    with torch.no_grad():
        results = model.test_step(batch)

    return results
