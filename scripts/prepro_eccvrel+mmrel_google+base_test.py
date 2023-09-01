"""
Add the spatial relations for objects and OCRs as in ECCV textvqa paper.
And the angle relations between OCRs.

Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jB', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jB', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
import torch
import torchvision.models as models
import skimage.io
from PIL import Image
from collections import defaultdict
from clear_caption import clear_ocr, clear_cap
from tqdm import *
import time
import pickle
from shapely.geometry import Polygon

def norm(box, w, h):
  box[0] = box[0]/w
  box[1] = box[1]/h
  box[2] = box[2]/w
  box[3] = box[3]/h
  return box, box[2]-box[0], box[3]-box[1]

def IoU(boxA, boxB):
  # determine the (x, y)-coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  # compute the area of intersection rectangle
  interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
  # compute the area of both the prediction and ground-truth
  # rectangles
  boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
  boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

  iou = interArea / float(boxAArea + boxBArea - interArea)

  return iou

def polygon_IoU(boxA, boxB):
  
  def get_union(A, B):
    return Polygon(A).union(Polygon(B)).area

  def get_intersection_over_union(A, B):
    return get_intersection(A, B) / get_union(A, B)

  def get_intersection(A, B):
    return Polygon(A).intersection(Polygon(B)).area

  A = [(boxA[0], boxA[1]), (boxA[2], boxA[3]), (boxA[4], boxA[5]), (boxA[6], boxA[7])]
  B = [(boxB[0], boxB[1]), (boxB[2], boxB[3]), (boxB[4], boxB[5]), (boxB[6], boxB[7])]
  return get_intersection_over_union(A, B)

def polygon_IoU_IoA_IoB(poly_A, poly_B):
  
  def get_union(A, B):
    return A.union(B).area

  def get_intersection_over_union(A, B):
    inter = get_intersection(A, B)
    union = get_union(A, B)
    Aa = A.area
    Ba = B.area
    union = np.where(union==0, np.ones_like(union), union)
    Aa = np.where(Aa==0, np.ones_like(Aa), Aa)
    Ba = np.where(Ba==0, np.ones_like(Ba), Ba)
    return  inter/Aa, inter/Ba, inter/union

  def get_intersection(A, B):
    return A.intersection(B).area

  return get_intersection_over_union(poly_A, poly_B)

def get_degree(boxA, boxB, output_degree=False):
  def unit_vector(vector):
      """ Returns the unit vector of the vector.  """
      norm_vector = np.linalg.norm(vector)
      norm_vector = np.where(norm_vector==0, np.ones_like(norm_vector), norm_vector)
      return vector / norm_vector
  # c1 = boxB[0]+boxB[2]-boxA[0]-boxA[2]
  # c2 = boxB[1]+boxB[3]-boxA[1]-boxA[3]
  centerA_x = (boxA[0]+boxA[2]+boxA[4]+boxA[6])/4.0
  centerA_y = (boxA[1]+boxA[3]+boxA[5]+boxA[7])/4.0
  centerB_x = (boxB[0]+boxB[2]+boxB[4]+boxB[6])/4.0
  centerB_y = (boxB[1]+boxB[3]+boxB[5]+boxB[7])/4.0
  c1 = centerB_x-centerA_x
  c2 = centerB_y-centerA_y
  v1 = np.array((c1, c2))
  v2 = np.array((1,0))
  v1_u = unit_vector(v1)
  v2_u = unit_vector(v2)
  angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
  degree = np.degrees(angle)
  # if degree == 180:
  #     degree = 179
  if c2 < 0:
      degree = 360 - degree
  if degree == 360:
      degree = 359

  if output_degree:
    return degree

  d = degree // 22.5
  if d == 0:
    d = 16
  spatial_rel = (d+1) // 2 + 4
  return spatial_rel

def get_vector_angle(v1, v2):
  dot = np.dot(v1, v2)
  mul = (np.linalg.norm(v1, 2) * np.linalg.norm(v2, 2))
  mul = np.where(mul==0, np.ones_like(mul), mul)
  r = np.arccos(dot / mul)
  deg = r * 180 / np.pi
  a1 = np.array([*v1, 0])
  a2 = np.array([*v2, 0])
  a3 = np.cross(a1, a2)

  if np.sign(a3[2]) > 0:
    deg = 360 - deg
  
  if deg == 360:
    deg = 0
  
  if np.isnan(deg):
    deg = 0
  assert deg < 360

  # if 0<=deg<=1 or 359<=deg<360:
  #   spatial_rel = 1
  # else:
  #   d = deg // 22.5
  #   if d == 0:
  #     d = 16
  #   spatial_rel = (d+1) // 2 + 1
  # return spatial_rel
  return deg

def get_relation(boxes, w, h):
  box_num = boxes.shape[0]
  relation = np.zeros((box_num, box_num))
  for A in range(box_num):
    for B in range(box_num):
      if A == B:
        relation[A,B] = 1
      elif (boxes[A][0]-boxes[B][0]) <= 0 and (boxes[A][1]-boxes[B][1]) <= 0 and \
           (boxes[A][2]-boxes[B][2]) >= 0 and (boxes[A][3]-boxes[B][3]) <= 0 and \
           (boxes[A][4]-boxes[B][4]) >= 0 and (boxes[A][5]-boxes[B][5]) >= 0 and \
           (boxes[A][6]-boxes[B][6]) <= 0 and (boxes[A][7]-boxes[B][7]) >= 0 :
        relation[A,B] = 2
      elif (boxes[A][0]-boxes[B][0]) >= 0 and (boxes[A][1]-boxes[B][1]) >= 0 and \
           (boxes[A][2]-boxes[B][2]) <= 0 and (boxes[A][3]-boxes[B][3]) >= 0 and \
           (boxes[A][4]-boxes[B][4]) <= 0 and (boxes[A][5]-boxes[B][5]) <= 0 and \
           (boxes[A][6]-boxes[B][6]) >= 0 and (boxes[A][7]-boxes[B][7]) <= 0 :
        relation[A,B] = 3
      elif polygon_IoU(boxes[A], boxes[B]) >= 0.5:
        relation[A,B] = 4
      else:
        boxA, Aw, Ah = norm(boxes[A].copy(),w,h)
        boxB, Bw, Bh = norm(boxes[B].copy(),w,h)
        relation[A,B] = get_degree(boxA, boxB)
  return relation

def get_ocr_geo_relation(boxes, w, h):
  box_num = boxes.shape[0]
  geo_relation = np.zeros((box_num, box_num, 13))
  
  if box_num > 0:
    for i in [0,2,4,6]:
      boxes[:,i] = boxes[:,i]/w
    for i in [1,3,5,7]:
      boxes[:,i] = boxes[:,i]/h
    
    Ws = []
    Hs = []
    for A in range(box_num):
      boxA = boxes[A].copy()
      vA_w = [boxA[2]+boxA[4]-boxA[0]-boxA[6], boxA[3]+boxA[5]-boxA[1]-boxA[7]]
      vA_h = [boxA[6]+boxA[4]-boxA[0]-boxA[2], boxA[7]+boxA[5]-boxA[1]-boxA[3]]
      Ws.append(np.sqrt(vA_w[0]**2 + vA_w[1]**2))
      Hs.append(np.sqrt(vA_h[0]**2 + vA_h[1]**2))

    for A in range(box_num):
      boxA = boxes[A].copy()
      Aw, Ah = Ws[A], Hs[A]
      poly_A = Polygon([(boxA[0], boxA[1]), (boxA[2], boxA[3]), (boxA[4], boxA[5]), (boxA[6], boxA[7])])
      vA_w = [boxA[2]+boxA[4]-boxA[0]-boxA[6], boxA[3]+boxA[5]-boxA[1]-boxA[7]]
      vA_h = [boxA[6]+boxA[4]-boxA[0]-boxA[2], boxA[7]+boxA[5]-boxA[1]-boxA[3]]
      
      for B in range(box_num):
        boxB = boxes[B].copy()
        Bw, Bh = Ws[B], Hs[B]
        poly_B = Polygon([(boxB[0], boxB[1]), (boxB[2], boxB[3]), (boxB[4], boxB[5]), (boxB[6], boxB[7])])
        
        d_w = abs(Aw-Bw)
        d_h = abs(Ah-Bh)
        ioA, ioB, iou = polygon_IoU_IoA_IoB(poly_A, poly_B)
        degree = get_degree(boxA, boxB, output_degree=True)
        dist = poly_A.distance(poly_B)

        vB_w = [boxB[2]+boxB[4]-boxB[0]-boxB[6], boxB[3]+boxB[5]-boxB[1]-boxB[7]]
        vB_h = [boxB[6]+boxB[4]-boxB[0]-boxB[2], boxB[7]+boxB[5]-boxB[1]-boxB[3]]
        horizontal_degree = get_vector_angle(vA_w, vB_w)
        vertical_degree = get_vector_angle(vA_h, vB_h)
        geo_relation[A, B] = np.array([
          Aw, Ah, Bw, Bh, d_w, d_h,
          ioA, ioB, iou,
          degree, dist,
          horizontal_degree, vertical_degree
        ])
  return geo_relation

def build_relations(imgs, params):
  subtask_id = params['subtask_id']-1 # starting from 0

  # clear ocr tokens and captions
  img_ids = []
  data_imgs = []
  for img in tqdm(imgs):
    if img['image_id'] not in img_ids:
      data_imgs.append(img)
      img_ids.append(img['image_id'])
  imgs_per_id = len(img_ids)//params['subtask_num']
  imgs_last_subtask = len(img_ids)-imgs_per_id*(params['subtask_num']-1)
    
  assert imgs_per_id*(params['subtask_num']-1)+imgs_last_subtask == len(img_ids)
  if params['subtask_num'] == params['subtask_id']:
    imgs_this_id = imgs_last_subtask
  else:
    imgs_this_id = imgs_per_id
  data_imgs = data_imgs[subtask_id*imgs_per_id : subtask_id*imgs_per_id+imgs_this_id]
  # data_imgs = data_imgs[:2]

  img2rel = {}
  img2ocr_geo_rel = {}
  for img in tqdm(data_imgs):
    ocr_file = np.load(str(params['input_ocr_root'])+'/'+img['image_id']+'_info.npy',\
                        allow_pickle=True)[()]
    dup_ocr_tokens = ocr_file['ocr_tokens']

    if params['input_aug_ocr_root']:
      aug_ocr_file = np.load(str(params['input_aug_ocr_root'])+'/'+img['image_id']+'_info.npy',\
                         allow_pickle=True)[()]
      aug_dup_ocr_tokens =  aug_ocr_file['ocr_tokens']
      dup_ocr_tokens = dup_ocr_tokens + aug_dup_ocr_tokens

    dup_ocr_tokens = dup_ocr_tokens[:params['max_ocr_len']]
    ocr_tokens = [clear_ocr(token).lower()for token in dup_ocr_tokens]
    
    img['ocr_ids'] = list(range(len(ocr_tokens)))
    img['new_ocr_tokens'] = ocr_tokens

    # spatial relation for ocr and obj
    obj_boxes = np.load(str(params['input_obj_root'])+'/'+str(img['image_id'])+'_info.npy', \
                allow_pickle=True)[()]['boxes']
    padded_obj_boxes = np.zeros([obj_boxes.shape[0], 8])
    for i,j in zip((0,1,2,3,4,5,6,7), (0,1,2,1,2,3,0,3)):
      padded_obj_boxes[:,i] = obj_boxes[:,j]
    
    ocr_boxes = ocr_file['ocr_boxes']
    if params['input_aug_ocr_root']:
      # origin is google, aug is base
      aug_ocr_boxes = aug_ocr_file['ocr_boxes']
      if aug_ocr_boxes.shape[0] > 0:
        aug_padded_ocr_boxes = np.zeros([aug_ocr_boxes.shape[0], 8])
        for i,j in zip((0,1,2,3,4,5,6,7), (0,1,2,1,2,3,0,3)):
          aug_padded_ocr_boxes[:,i] = aug_ocr_boxes[:,j]
        aug_ocr_boxes = aug_padded_ocr_boxes

        if ocr_boxes.shape[0] == 0:
          ocr_boxes = aug_ocr_boxes
        else:
          ocr_boxes = np.concatenate([ocr_boxes, aug_ocr_boxes],axis=0)
    ocr_boxes = ocr_boxes[:params['max_ocr_len']]

    if ocr_boxes.shape[0] > 0:
      all_boxes = np.concatenate([padded_obj_boxes, ocr_boxes],axis=0)
    else:
      all_boxes = padded_obj_boxes

    all_boxes_relations = get_relation(all_boxes, img['image_width'], img['image_height'])
    img['spatial_relations'] = all_boxes_relations
    img2rel[img['image_id']] = img['spatial_relations']

    # relation of the horizontal and vertical angle between ocrs
    geo_relation = get_ocr_geo_relation(ocr_boxes, img['image_width'], img['image_height'])
    img2ocr_geo_rel[img['image_id']] = geo_relation

    # for i in range(20):
    #   for j in range(20):
    #     print(geo_relation[i,j])
    #     input()

  return img2rel, img2ocr_geo_rel

def main(params):

  # train = json.load(open(params['input_json_PR']+'train.json', 'r'))
  # val = json.load(open(params['input_json_PR']+'val.json', 'r'))
  # imgs = train['data']
  # imgs.extend(val['data'])
  test = json.load(open(params['input_json_PR']+'test.json', 'r'))
  imgs = test['data']

  if not os.path.isdir(params['output_json'].split(params['output_json'].split('/')[-1])[0]):\
    os.makedirs(params['output_json'].split(params['output_json'].split('/')[-1])[0])

  seed(123) # make reproducible
  
  # create the relation dict
  img2rel, img2ocr_geo_rel = build_relations(imgs, params)
  out = {}
  out['img2rel'] = img2rel
  out['img2ocr_geo_rel'] = img2ocr_geo_rel

  pickle.dump(out, open(params['output_json']+str(params['subtask_id'])+'.pkl', 'wb'))
  print('wrote ', params['output_json']+str(params['subtask_id'])+'.pkl')
  print('image num: ', len(img2rel))

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--input_json_PR', default='TextCaps_0.1_', required=True, help='prefix of input json file to process into hdf5')
  parser.add_argument('--output_json', default='TC.json', help='output json file')
  parser.add_argument('--output_h5', default='TC', help='output h5 file')
  parser.add_argument('--output_vocab', default='TC_vocab', help='output vocab file')
  parser.add_argument('--images_root', default='', help='root location in which images are stored, to be prepended to file_path in input json')
  parser.add_argument('--input_ocr_root', default='m4c_textvqa_ocr_en_frcn_features', required=True, help='')
  parser.add_argument('--input_obj_root', default='m4c_textvqa_ocr_en_frcn_features', required=True, help='')
  parser.add_argument('--input_aug_ocr_root', default=None, help='')

  # options
  parser.add_argument('--max_length', default=20, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
  parser.add_argument('--word_count_threshold', default=10, type=int, help='only words that occur more than this number of times will be put in vocab')
  parser.add_argument('--max_ocr_len', default=100, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
  parser.add_argument('--subtask_num', default=20, type=int)
  parser.add_argument('--subtask_id', type=int)

  

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print('parsed input parameters:')
  print(json.dumps(params, indent = 2))
  main(params)
