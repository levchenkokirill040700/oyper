import os, sys, argparse
import numpy as np
import cv2
from yolo_detector import YoloDetector

class Stats:
  def __init__(self, categories=None):
    self.TPs, self.FPs, self.FNs = [], [], []
    self.categories = categories
    self.ious_per_category = [] if categories is None else {label:[] for label in categories}

  def __call__(self, detections, gt_objects, iouThreshold=0.4, threshold=0.0):
    for det in detections:
      label, box, confidence = det['label'], det['box'], det['confidence']
      if not self._in_categories(label): continue
      if confidence < threshold: continue
      iou =  self._max_IOU_same_category(label, box, gt_objects)
      self.append_iou(label, iou)
      if iou >= iouThreshold: self.TPs.append(label)
      else:                   self.FPs.append(label)
    for gt_obj in gt_objects:
      label, box = gt_obj['label'], gt_obj['box']
      if not self._in_categories(label): continue
      iou =  self._max_IOU_same_category(label, box, detections)
      if iou < iouThreshold:  self.FNs.append(label)

  def append_iou(self, label, iou):
    if self.categories is None: self.ious_per_category.append(iou)
    elif label in self.ious_per_category: self.ious_per_category[label].append(iou)

  def get_iou_stats(self, category=None):
    if self.categories is None: ious = self.ious_per_category
    else:
      if category is None:
        ious = []
        for _,v in self.ious_per_category.items(): ious += v
      else: ious = self.ious_per_category.get(category, [])
    if len(ious) == 0: return None, None
    return np.mean(ious), np.std(ious)

  def get_recognition_stats(self, category=None):
    if category is None: tps, fps, fns = len(self.TPs), len(self.FPs), len(self.FNs)
    else:
      tps = len([el for el in self.TPs if el == category])
      fps = len([el for el in self.FPs if el == category])
      fns = len([el for el in self.FNs if el == category])
    pre = None if tps+fps == 0 else tps/float(tps+fps)
    rec = None if tps+fns == 0 else tps/float(tps+fns)
    f1s = None if pre is None or rec is None else 2.*pre*rec/(pre+rec)
    return pre, rec, f1s

  def _in_categories(self, label):
    if self.categories is  None: return True
    if label in self.categories: return True

  def _max_IOU_same_category(self, label, box, objects):
    ious = [self._iou(box, obj['box']) for obj in objects if obj['label'] == label]
    iou  = 0. if len(ious) == 0 else max(ious)
    return iou

  @staticmethod
  def _iou(b1, b2):
    l, t = max(b1['left'],  b2['left']),  max(b1['top'],    b2['top'])
    r, b = min(b1['right'], b2['right']), min(b1['bottom'], b2['bottom'])
    interArea = max(0, r-l+1) * max(0, b-t+1)
    b1Area = (b1['right'] - b1['left'] + 1) * (b1['bottom'] - b1['top'] + 1)
    b2Area = (b2['right'] - b2['left'] + 1) * (b2['bottom'] - b2['top'] + 1)
    return interArea / float(b1Area + b2Area - interArea)


def get_yolo_annotation_filename(yolo_img_fn):
  els = yolo_img_fn.split('/')
  els[-2] = 'labels'
  path = '/'.join(els)
  path = '.'.join(path.split('.')[:-1]) + '.txt'
  return path

def log_detections(objs, dets, det_log):
  box2Str = lambda b: "{},{},{},{}".format(b['left'],b['top'],b['right'],b['bottom'])
  scr2Str = lambda s: "{}".format(s)
  gtObj2Str = lambda obj: obj['label'] + ',' + box2Str(obj['box'])
  det2Str = lambda obj: gtObj2Str(obj) + ',' + scr2Str(obj['confidence'])
  s = '--' if len(objs) == 0 else gtObj2Str(objs[0])
  for obj in objs[1:]: s += ';' + gtObj2Str(obj)
  if len(objs) > 0: s += '--'
  for obj in dets: s += det2Str(obj) + ';'
  if s.endswith(','): s = s[:-1]
  det_log.write(s + '\n')

def parse_yolo_annotation(ann_fn, labels, iw, ih):
  lines = open(ann_fn).read().splitlines()
  objs = []
  for line in lines:
    label_ind, cx, cy, w, h = line.split()
    label = labels.get(int(label_ind), 'UNDEFINED')
    cx, cy = int(float(cx) * iw), int(float(cy) * ih)
    w,  h  = int(float(w)  * iw), int(float(h)  * ih)
    l,  t  = cx - int(w/2.), cy - int(h/2.)
    r,  b  = l + w, t  + h
    objs.append({"label": label, "box": {"left": l, "top": t, "right": r, "bottom": b}})
  return objs

def parse_args():
  parser = argparse.ArgumentParser("Estimate accuracy of clothes detector (bounding box position, class prediction)")
  parser.add_argument('-d', '--dataset',     type=str, required=True, help="List of test dataset files (yolo formatted, like <test-folder>/yolo.txt)")
  parser.add_argument('-n', '--names',       type=str, required=True, help="Yolo file with labelnames of interest")
  parser.add_argument('-yc','--yolo_cfg',    type=str, required=True, help="Yolo config (model) file")
  parser.add_argument('-yw','--yolo_weights',type=str, required=True, help="Yolo weights file")
  parser.add_argument('-yt','--yolo_threshold',type=float,default=0.01,help="Yolo threshold")
  parser.add_argument('-dl','--detection_log', type=str,  default='detection_log.txt')
  parser.add_argument('-sl','--stats_log',     type=str,  default='stats_log.txt')
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()
  detector = YoloDetector(args.yolo_cfg, args.yolo_weights, args.names)
  labels   = {i:name for i, name in enumerate(open(args.names).read().splitlines())}
  threshold= args.yolo_threshold
  iouThreshold=0.4
  test_images_filenames = open(args.dataset).read().splitlines()
  print("Going to test {} yolo model against {} test images".
         format(args.yolo_weights.split('/')[-1], len(test_images_filenames)))

  categories1 = ('full_body', 'lower_body', 'upper_body')
  categories2 = ('Blazer',  'Blouse', 'Cardigan',  'Coat',       'Cutoffs', 
  	             'Dress',   'Hoodie', 'Jacket',    'Jeans',      'Joggers',
                 'Jumpsuit','Kimono', 'Leggings',  'Romper',     'Shorts',
                 'Skirt',   'Sweater','Sweatpants','Sweatshorts','Tank',
                 'Tee',     'Top')
  stats1, stats2 = Stats(categories1), Stats(categories2)
  
  det_log = open(args.detection_log, 'w')

  for i, img_fn in enumerate(test_images_filenames):
    print("Processing {}\t({} from {})".format(img_fn, i+1, len(test_images_filenames)))
    img    = cv2.imread(img_fn)
    if img is None:
      print("Cannot read " + img_fn)
      continue
    ih, iw = img.shape[:2]
    ann_fn = get_yolo_annotation_filename(img_fn)
    objs   = parse_yolo_annotation(ann_fn, labels, iw, ih)
    print("Calling detector")
    dets   = detector(img, threshold)
    print("Returned to main function")
    print("GT:", objs)
    print("Dets:", dets)
    print("----------------------")
    stats1(dets, objs, iouThreshold, threshold)
    stats2(dets, objs, iouThreshold, threshold)
    log_detections(objs, dets, det_log)

  def print_log_stats(categories, stats, sts_log):
    toS = lambda v: 'None' if v is None else '{:.2f}'.format(v) 
    for category in categories:
      pre, rec, f1s = stats.get_recognition_stats(category)
      iou_mean, iou_std = stats.get_iou_stats(category)
      s = "{}: pre {} rec {} f1s {} iou_mean {} iou_std {}".format(category,
        toS(pre), toS(rec), toS(f1s), toS(iou_mean), toS(iou_std))
      print(s)
      sts_log.write(s + '\n')
    pre, rec, f1s = stats.get_recognition_stats()
    iou_mean, iou_std = stats.get_iou_stats()
    s = "{}: pre {} rec {} f1s {} iou_mean {} iou_std {}".format('ALL',
      toS(pre), toS(rec), toS(f1s), toS(iou_mean), toS(iou_std))
    print(s)
    sts_log.write(s + '\n')

  sts_log = open(args.stats_log, 'w')
  print_log_stats(categories1, stats1, sts_log)
  print_log_stats(categories2, stats2, sts_log)
