"""Script to evaluate scylla weapon detector.
   Input:  test images. Groundtruth is obtained from image name ('generic*' vs 'classname*')
   Output: general eval results (accuracy/precision/recall/f1_score) and the same per classname"""

import sys, os, argparse
sys.path.append(os.path.join(os.getcwd(),'python/'))
import xml.etree.ElementTree as ET
import darknet as dn
import cv2

# intersection over union
def IoU(b1, b2):
  ib = (max(b1[0],b2[0]), max(b1[1],b2[1]), min(b1[2],b2[2]), min(b1[3],b2[3])) # intersection box
  if ib[0] > ib[2] or ib[1] > ib[3]: return 0.0 # no intersection at all
  iarea = float(ib[2] - ib[0] + 1) * (ib[3] - ib[1] + 1)
  a1 = float(b1[2] - b1[0] + 1) * (b1[3] - b1[1] + 1)
  a2 = float(b2[2] - b2[0] + 1) * (b2[3] - b2[1] + 1)
  return iarea / (a1 + a2 - iarea)

class Estimation:
  def __init__(self):
    self.TP = 0
    self.TN = 0
    self.FP = 0
    self.FN = 0
    self.pos = 0
    self.neg = 0

  @staticmethod
  def getAccPreRecF1S(TP, FP, TN, FN):
    acc = -1 if TP+FP+FN+TN == 0 else float(TP+TN)/(TP+FP+FN+TN)
    pre = -1 if TP+FP == 0 else float(TP)/(TP+FP)
    rec = -1 if TP+FN == 0 else float(TP)/(TP+FN)
    f1s = -1 if pre==-1 or rec == -1 else 2.*rec*pre/(rec+pre)
    return acc, pre, rec, f1s

  def printStats(self):
    print("TP\tFP\tTN\tFN\tpos\tneg")
    print("{}\t{}\t{}\t{}\t{}\t{}".format(self.TP, self.FP, self.TN, self.FN, self.pos, self.neg))
    acc, pre, rec, f1s = Estimation.getAccPreRecF1S(self.TP, self.FP, self.TN, self.FN)
    print("Acc\tPre\tRec\tF1S")
    print("{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(acc, pre, rec, f1s))

class EstimationTable:
  def __init__(self, classnames, iouThreshold):
    self.noCondition = 'no_object'
    self.noDetection = 'no_detection'
    self.recognitionTable = {}
    for condition in classnames:    # first index - condition (groundtruth)
      self.recognitionTable[condition] = {}
      for detection in classnames: # second key - detection result
       self.recognitionTable[condition][detection] = 0
      self.recognitionTable[condition][self.noDetection] = 0
    self.recognitionTable[self.noCondition] = {}
    for detection in classnames: self.recognitionTable[self.noCondition][detection] = 0
    self.recognitionTable[self.noCondition][self.noDetection] = 0
    self.classnames = classnames
    self.iouThreshold = iouThreshold

  def _checkDetected(self, dets_c, gtBox, condition=None):
    if condition is None: dets_c = {key:det for key, det in dets_c.items() if IoU(det['box'], gtBox) >= self.iouThreshold}
    else:                 dets_c = {key:det for key, det in dets_c.items() if IoU(det['box'], gtBox) >= self.iouThreshold and
                                                                              det['cls'] == condition}
    if len(dets_c) == 0:
      return self.noDetection, None
    bestKey = list(dets_c.keys())[0]
    bestDet = dets_c[bestKey]
    for key, det in dets_c.items():
      if det['score'] > bestDet['score']:
        bestKey = key
        bestDet = det
    return bestDet['cls'], bestKey
    
  def analyseDetections(self, dets, gtObjs):
    # use copies of detections and gt objects
    dets_c   = {i:det   for i, det   in enumerate(dets)   if det['cls']   in self.classnames}
    gtObjs_c = {i:gtObj for i, gtObj in enumerate(gtObjs) if gtObj['cls'] in self.classnames}
    # first, check for clear TN
    if len(dets_c) == 0 and len(gtObjs_c) == 0:
      self.recognitionTable[self.noCondition][self.noDetection] += 1
      return # no need to proceed
    # second, look for all TPs
#    for gtKey in list(gtObjs_c.keys()):
#      gtObj = gtObjs_c[gtKey]
#      condition, gtBox  = gtObj['cls'], gtObj['box']
#      detection, detKey = self._checkDetected(dets_c, gtBox, condition) # returns self.noDetection, None if no corresponding detections have been found
#      if detection == condition:  # must be checked to sure detection is not self.noDetection
#        self.recognitionTable[condition][detection] += 1
#        del gtObjs_c[gtKey]
#        if detKey is not None: del dets_c[detKey]
    # next look for all confusions
    for gtKey in list(gtObjs_c.keys()):
      gtObj = gtObjs_c[gtKey]
      condition, gtBox  = gtObj['cls'], gtObj['box']
      detection, detKey = self._checkDetected(dets_c, gtBox)  # returns self.noDetection, None if no detections have been found
      self.recognitionTable[condition][detection] += 1
      del gtObjs_c[gtKey]
      if detKey is not None: del dets_c[detKey]
    # finally check for clear FPs
    condition = self.noCondition
    for det in dets_c.values():
      detection = det['cls']
      self.recognitionTable[condition][detection] += 1

  def printTable(self):
    def list2Line(l, t, s=''):
      for el in l: s += '{}{}'.format(el, t)
      return s
    print("Recognition Table:")
    print("\t\t\t Detections")
    print("\t\t" + list2Line(list(self.recognitionTable[list(self.recognitionTable.keys())[0]].keys()), '  '))
    for condition, row_dict in self.recognitionTable.items():
      tab = '\t' if len(condition) < 8 else ''
      print(condition + ':\t' + tab + list2Line(row_dict.values(), '\t'))

def convertWithThreshold(yoloDets, thresholds):
  dets = []
  for det in yoloDets: # convert to better format
    cls   = det[0].decode("utf-8")
    score = det[1]
    w, h  = det[2][2], det[2][3]
    l, t  = det[2][0] - w/2., det[2][1] - h/2.
    r, b  = l + w, t + h
    box   = [l, t, r, b]
    threshold = thresholds.get(cls) if cls in thresholds else 0.0
    if score >= threshold: dets.append({'cls':cls, 'score':score, 'box':box})
  return dets

def estimateDets(dets, gtObjs, classnames, estimation):
  neg = True
  for gtObj in gtObjs:
    if gtObj['cls'] not in classnames: continue
    neg = False
    estimation.pos += 1
    if not inList(gtObj['cls'], gtObj['box'], [(det['cls'], det['box']) for det in dets]):
      estimation.FN += 1
  for det in dets:
    if det['cls'] not in classnames: continue
    if     inList(det['cls'], det['box'], [(gtObj['cls'], gtObj['box']) for gtObj in gtObjs]):
      estimation.TP += 1
    else:
      estimation.FP += 1
  if neg:
    estimation.neg += 1
    if len(dets) == 0: estimation.TN += 1

def getAnnotation(annFn):
  tree=ET.parse(open(annFn))
  root = tree.getroot()
  size = root.find('size')
  w = int(size.find('width').text)
  h = int(size.find('height').text)
  objs = []
  for obj in root.iter('object'):
    difficult = obj.find('difficult').text
    cls = obj.find('name').text
    xmlbox = obj.find('bndbox')
    box = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
           float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
    objs.append({'cls':cls, 'box':box})
  return objs

def getDataset(rootFolder):
  replaceExt = lambda fn, ext: '.'.join(fn.split('.')[:-1]) + '.' + ext

  annFolder = os.path.join(rootFolder, 'annotations')
  imgFolder = os.path.join(rootFolder, 'images')
  imgFns    = os.listdir(imgFolder)
  samples   = []
  for imgFn in imgFns:
    annFn = os.path.join(annFolder, replaceExt(imgFn, 'xml'))
    if not os.path.isfile(annFn): continue
    objs = getAnnotation(annFn)
    samples.append({'imgFn': os.path.join(imgFolder, imgFn), 'objs':objs})
  return samples

def inList(cls, box, objs):
  for obj in objs:
    if cls != obj[0]: continue
    iou = IoU(box, obj[1])
    if iou > 0: return True
  return False


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfg',       type=str,  required=True, help="yolo .cfg")
  parser.add_argument('-m', '--model',     type=str,  required=True, help="yolo .weights")
  parser.add_argument('-l', '--labels',    type=str,  required=True, help="classnames file (.data)")
  parser.add_argument('-d', '--dataset',   type=str,  required=True, help="voc root (with annotations and images)")
  parser.add_argument('-t', '--threshold', type=str,  default='0.5', help="threshold; one - used for all classes; many, splitted by comma - each per class; default: 0.5")
  parser.add_argument('-v', '--verbose',   action='store_true', help="print detections")

  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()
  
  verbose = args.verbose
  dataset = []
  for root in args.dataset.split(','): dataset += getDataset(root)
  classnames = open(args.labels).read().splitlines()
  print(classnames)
  perClassEstimations = {classname:Estimation() for classname in classnames}
  generalEstimation   = Estimation()
  estimationTable     = EstimationTable(classnames, 0.1)
  
  cfg  = args.cfg
  met  = args.labels
  wgh  = args.model
  thrs = [float(el) for el in args.threshold.split(',')]
  if len(thrs) == 1: thrs = thrs * len(classnames)
  assert(len(thrs) == len(classnames)),"Provide thresholds amount the same as classnames amount or single"
  thrs = {cls:thrs[i] for i, cls in enumerate(classnames)}
  print(cfg, met, wgh)
  
  dn.set_gpu(0)
  net  = dn.load_net(cfg.encode('utf-8'), wgh.encode('utf-8'), 0)
  meta = dn.load_names(met.encode('utf-8'))

  for i, sample in enumerate(dataset):
    print("{} ({} from {})".format(sample['imgFn'], i+1, len(dataset)))
    img = cv2.imread(sample['imgFn'])
    dets = dn.detect(net, meta, dn.nparray_to_image(img))
    dets = convertWithThreshold(dets, thrs)
    if verbose:
      print('\t{}'.format(sample['objs']))
      print('\t{}'.format(dets))
    estimateDets(dets, sample['objs'], classnames, generalEstimation)
    for classname, estimation in perClassEstimations.items(): estimateDets(dets, sample['objs'], (classname,), estimation)
    estimationTable.analyseDetections(dets, sample['objs'])

  print("General estimation results:")
  generalEstimation.printStats()
  print("Per class estimation results:")
  for classname, estimation in perClassEstimations.items():
    print("-----  " + classname + "  -----")
    estimation.printStats()
  print('')
  estimationTable.printTable()
  print("Used thresholds:")
  print(thrs)