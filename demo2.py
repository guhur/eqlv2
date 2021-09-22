import warnings
from argparse import ArgumentParser
from mmdet.apis import init_detector
from mmdet.apis.inference import LoadImage
from mmdet.core import bbox2roi, bbox2result
import torch
from mmcv.ops import RoIAlign, RoIPool
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
import torch
from helper import inference_detector

parser = ArgumentParser()
parser.add_argument('img', help='Image file')
parser.add_argument('config', help='Config file')
parser.add_argument('checkpoint', help='Checkpoint file')
args = parser.parse_args()

# Specify the path to model config and checkpoint file
# config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# build the model from a config file and a checkpoint file
model = init_detector(args.config, args.checkpoint, device='cuda:0')

# test a single image and show the results
img = args.img
# or img = mmcv.imread(img), which will only load it once

results = inference_detector(model, [img, img])
import ipdb; ipdb.set_trace()


# or save the visualization results to image files
bbox_results = bbox2result(results['bbox'], results['labels'],
                           model.roi_head.bbox_head.num_classes)
model.show_result(img, bbox_results, out_file='result.jpg')
