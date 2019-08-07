#!/usr/bin/env python
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2 as cv
import argparse

def draw_boxes(im, class_name, dets, thresh=0.5):
  """Draw detected bounding boxes."""
  inds = np.where(dets[:, -1] >= thresh)[0]
  if len(inds) == 0:
    return

  im = im[:, :, (2, 1, 0)]
  fig, ax = plt.subplots(figsize=(12, 12))
  ax.imshow(im, aspect='equal')

  height, width = im.shape[:2]
  x_scale = width / 500
  y_scale = height / 375

  for i in inds:
    bbox = dets[i, :4]
    score = dets[i, -1]

    ax.add_patch(
      plt.Rectangle((bbox[0] * x_scale, bbox[1] * y_scale),
                    (bbox[2] - bbox[0]) * x_scale,
                    (bbox[3] - bbox[1]) * y_scale, fill=False,
                    edgecolor='red', linewidth=3.5)
      )
    ax.text((bbox[0])  * x_scale, (bbox[1] - 2)  * y_scale,
      '{:s} {:.3f}'.format(class_name, score),
      bbox=dict(facecolor='blue', alpha=0.5),
      fontsize=14, color='white')

  ax.set_title(('{} detections with '
                'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                thresh),
                fontsize=14)
  plt.axis('off')
  plt.tight_layout()
  plt.draw()

def demo(net, image_file, labels, conf_thre, nms_thre):
  """Detect object classes in an image using pre-computed object proposals."""

  # Load the demo image
  im = cv.imread(image_file)
  resized_im = cv.resize(im, (500, 375))

  # Detect all object classes and regress object bounds
  timer = Timer()
  timer.tic()
  scores, boxes = im_detect(net, resized_im)
  timer.toc()
  print(('Detection took {:.3f}s for '
          '{:d} object proposals').format(timer.total_time, boxes.shape[0]))

  # Visualize detections for each class
  for cls_ind, cls in enumerate(labels[1:]):
    cls_ind += 1 # because we skipped background
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thre)
    dets = dets[keep, :]
    draw_boxes(im, cls, dets, thresh=conf_thre)

def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='Inference Demo for PAIV FRCNN Trained Models')
  parser.add_argument('--gpu', dest='gpu_id', help='Specify to use GPU device instead of CPU [-1]',
                      default=-1, type=int)
  parser.add_argument('--model', dest='model', help='Caffe Model to Use',
                      default='./model.caffemodel')
  parser.add_argument('--deploy', dest='deploy', help='Deployment Prototxt File to Use',
                      default='./test.prototxt')
  parser.add_argument('--labels', dest='labels', help='Objects to detect',
                      default='./classname.txt')
  parser.add_argument('--image', dest='image', help='Image to test detection on',
                      default='./test.jpg')
  parser.add_argument('--conf_thre', dest='conf_thre', help='Image to test detection on',
                      default=0.8, type=float)
  parser.add_argument('--nms_thre', dest='nms_thre', help='Image to test detection on',
                      default=0.3, type=float)

  args = parser.parse_args()

  return args

if __name__ == '__main__':
  cfg.TEST.HAS_RPN = True  # Use RPN for proposals

  args = parse_args()
  prototxt = args.deploy
  caffemodel = args.model

  with open(args.labels) as f:
    for line in f:
      labels = eval(line)

  labels = ('__background__',) + labels

  if not os.path.isfile(caffemodel):
    raise IOError(('{:s} not found.\nPlease use the correct Caffe model').format(caffemodel))

  if args.gpu_id > -1:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    cfg.GPU_ID = args.gpu_id
  else:
    caffe.set_mode_cpu()

  net = caffe.Net(prototxt, caffemodel, caffe.TEST)

  print('Loaded network {:s}'.format(caffemodel))

  demo(net, args.image, labels, args.conf_thre, args.nms_thre)

  plt.show()


