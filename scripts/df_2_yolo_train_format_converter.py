"""
Convert dataset represented in format provided by Deep Fashion into Yolo format:

Input for the script:
1) Deep Fashion dataset root (which includes 'Img', 'Anno' and 'Eval' subfolders);
2) txt file with labelnames (one of yolo format should be taken).
3) Root folder where to store result train/val/test datasets 
   (with subfolders 'labels' and 'images', file yolo.txt).

Outputs of the script:
1) train, val and test folders with dataset represented in yolo format.
The script must be run onto the instance where training will be run.
"""

import os, sys, argparse
import itertools
import cv2
from random import shuffle
from shutil import copyfile

def addBackground(image, imgFolder, bgFns, bgCounter):
  newFn = lambda fn: '.'.join(fn.split('.')[:-1]) + '_backgrounded.jpg'
  tries = 100
  bgImg = cv2.imread(bgFns[bgCounter])
  while bgImg is None and tries > 0:
    tries -= 1
    bgCounter += 1
    if bgCounter >= len(bgFns): bgCounter = 0
    bgImg = cv2.imread(bgFns[bgCounter])
  fn  = os.path.join(imgFolder, image['fn'])
  img = cv2.imread(fn)
  if img is None: 
    print("Cannot read image to add background " + image['fn'])
    return bgCounter
  newSize = (2 * img.shape[1], 2 * img.shape[0])
  bgImg = cv2.resize(bgImg, newSize)
  if len(img.shape) != len(bgImg.shape):
    print("Shapes of images are different img {} vs bg {}".format(img.shape, bgImg.shape))
    return bgCounter
  l, t, r, b = (int(el) for el in image['box'])
  if l <= 0: l = 1
  if t <= 0: t = 1
  if r >= img.shape[1]: r = img.shape[1] - 1
  if b >= img.shape[1]: b = img.shape[0] - 1
  if len(img.shape) == 3: bgImg[t:b, l:r, :] = img[t:b, l:r, :]
  else:                   bgImg[t:b, l:r]    = img[t:b, l:r]
  image['fn'] = newFn(image['fn'])
  cv2.imwrite(os.path.join(imgFolder, image['fn']), bgImg)
  bgCounter += 1
  if bgCounter >= len(bgFns): bgCounter = 0
  return bgCounter

def analyseAnnotations(images, offset):
  stats = {}
  counter = 0
  for image in images:
    cat = image['category']
    typ = image['clothes_type']
    fn  = os.path.join(imgFolder, image['fn'])
    img = cv2.imread(fn)
    if not cat is None:
      if not cat in stats: stats[cat] = {'presented':0,'boxCorrect':0,'boxInNPix':0}
      stats[cat]['presented'] += 1
      if not img is None: 
        stats[cat]['boxCorrect'] += isBBoxCorrect(image['box'],img.shape)
        stats[cat]['boxInNPix']  += isBBoxCorrect(image['box'],img.shape, offset)
    if not typ is None:
      if not typ in stats: stats[typ] = {'presented':0,'boxCorrect':0,'boxInNPix':0}
      stats[typ]['presented'] += 1
      if not img is None:
        stats[typ]['boxCorrect'] += isBBoxCorrect(image['box'],img.shape)
        stats[typ]['boxInNPix']  += isBBoxCorrect(image['box'],img.shape, offset)
    #if not isBBoxCorrect(image['box'],img.shape): show(img, image['box'])
    counter += 1
  return stats, counter

def convertBBox(box, w, h):
    box = [float(el) for el in box]
    dw, dh = 1./w, 1./h
    x = (box[0] + box[2])/2.0
    y = (box[1] + box[3])/2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x, y = x*dw, y*dh
    w, h = w*dw, h*dh
    return " ".join([str(a) for a in (x,y,w,h)])

def convert_annotation(annFn, yoloAnnFn, classes):
  tree=ET.parse(open(annFn))
  root = tree.getroot()
  size = root.find('size')
  w = int(size.find('width').text)
  h = int(size.find('height').text)
  add2Lst = False
  f = open(yoloAnnFn, 'w')
  for obj in root.iter('object'):
    difficult = obj.find('difficult').text
    cls = obj.find('name').text
    if cls not in classes or int(difficult) == 1:
      continue
    cls_id = classes.index(cls)
    xmlbox = obj.find('bndbox')
    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
    bb = convert((w,h), b)
    f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    add2Lst = True
  f.close()
  return add2Lst


def enrichWithAttr(images, attrAnnFn, attributes):
  attrs = {}
  with open(attrAnnFn) as f:
    for line in itertools.islice(f, 2, len(images)+2):
      els = line.split()
      fn  = els[0]
      a   = [attributes.get(ind) for ind, el in enumerate(els[1:]) if el == '1']
      a   = [el for el in a if el is not None]
      attrs[fn] = a
  for image in images: image['attributes'] = attrs.get(image['fn'])

def enrichWithCategory(images, catAnnFn, categories):
  cats = {}
  with open(catAnnFn) as f:
    for line in itertools.islice(f, 2, len(images)+2):
      fn, catInd = line.split()
      d = categories.get(int(catInd))
      if d is None: continue
      cats[fn] = d
  for image in images:
    d = cats.get(image['fn'])
    if d is None: 
      image['category']     = None
      image['clothes_type'] = None
      continue
    image['category']     = d['category']
    image['clothes_type'] = d['clothes_type']

def enrichWithEval(images, evalLabelFn):
  labels = {}
  with open(evalLabelFn) as f:
    for line in itertools.islice(f, 2, len(images)+2):
      fn, label = line.split()
      labels[fn] = label
  for image in images:
    image['evalLabel'] = labels.get(image['fn'])

def isBBoxCorrect(box, img_shape, offset=0):
  l,r,t,b = (int(el) for el in box)
  h, w    = img_shape[:2]
  if l <    offset or t <    offset: return False
  if r >= w-offset or b >= h-offset: return False
  return True

def makeFolder(path):
  if os.path.exists(path): return path
  os.makedirs(path)
  return path

def notFullData(image):
  if image.get('clothes_type') is None: return True
  if image.get('category')     is None: return True
  if image.get('evalLabel')    is None: return True
  if image.get('attributes')   is None: return True
  return False

def readAttrNames(attrNamesFn):
  lines = open(attrNamesFn).read().splitlines()[2:]
  return {ind:line.split()[0] for ind, line in enumerate(lines)}

def readBBoxes(bboxAnnFn, limit):
  def line2dict(line):
    name, l, t, r, b = line.split()
    return {'fn':name, 'box':(l,t,r,b)}
  images = []
  with open(bboxAnnFn) as f:
    if limit is None:
      for line in f.read().splitlines()[2:]:            images.append(line2dict(line))
    else:
      for line in itertools.islice(f, 2, limit+2):  images.append(line2dict(line))
  return images

def readCatNames(catNamesFn):
  clothes_types = {'1':'upper_body', '2':'lower_body', '3':'full_body'}
  lines = open(catNamesFn).read().splitlines()[2:]
  return {ind+1:{'category':line.split()[0],'clothes_type':clothes_types[line.split()[1]]}
          for ind, line in enumerate(lines)}

def put2Dataset(image, dataset, names, folder):
  convertImgFn = lambda fn: '_'.join(fn.split('/')[1:])
  toAnn = lambda cls_ind, box_str: "{} {}".format(cls_ind, box_str)

  imgFolder = makeFolder(os.path.join(folder, 'images'))
  annFolder = makeFolder(os.path.join(folder, 'labels'))
  srcImgFn  = os.path.join(dataset, image['fn'])
  if not os.path.isfile(srcImgFn):  
    print("No file " + srcImgFn)
    return None
  imgName   = convertImgFn(image['fn'])
  dstImgFn  = os.path.join(imgFolder, imgName)
  annName   = '.'.join(imgName.split('.')[:-1]) + '.txt'
  img = cv2.imread(srcImgFn)
  if img is None: 
    print("Cannot read file " + srcImgFn)
    return None
  h, w = img.shape[:2]
  copyfile(srcImgFn, dstImgFn)
  box_str = convertBBox(image['box'], w, h)
  with open(os.path.join(annFolder, annName), 'w') as f:
    cls_ind = names.get(image['clothes_type'])
    if cls_ind is not None: f.write(toAnn(cls_ind, box_str) + '\n')
    cls_ind = names.get(image['category'])
    if cls_ind is not None: f.write(toAnn(cls_ind, box_str) + '\n')
    for attribute in image['attributes']:
      cls_ind = names.get(attribute)
      if cls_ind is not None: f.write(toAnn(cls_ind, box_str) + '\n')
  return dstImgFn

def saveList(lst, folder):
  path = os.path.join(folder, 'yolo.txt')
  count = 0
  with open(path, 'w') as f:
    for l in lst:
      f.write(l + '\n')
    count = len(lst)
  return path, count

def scan_fns(folder):
  paths = []
  els = [os.path.join(folder, fn) for fn in os.listdir(folder)]
  for path in els:
    if  os.path.isfile(path): paths.append(path)
    elif os.path.isdir(path): paths += scan_fns(path)
  return paths

def show(img, box):
  box = tuple([int(el) for el in box])
  cv2.rectangle(img, box[:2], box[2:], (0,0,255), 2)
  cv2.imshow('win', img)
  cv2.waitKey()
  cv2.destroyAllWindows()

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset', type=str, required=True, help="Deep Fashion root folder (with 'Anno', 'Eval', 'Img'")
  parser.add_argument('-n', '--names',   type=str, required=True, help="Yolo file with labelnames of interest")
  parser.add_argument('-b', '--background',type=str, default=None,help="To add background around boxes specify root folder with images to use for background")
  parser.add_argument('-o', '--output',  type=str, required=True, help="Output root folder")
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()
  names = {name:ind for ind, name in enumerate(open(args.names).read().splitlines())}

  imgFolder   = os.path.join(args.dataset, 'Img')
  bboxAnnFn   = os.path.join(args.dataset, 'Anno', 'list_bbox.txt')
  lmAnnFn     = os.path.join(args.dataset, 'Anno', 'list_landmarks.txt')
  catAnnFn    = os.path.join(args.dataset, 'Anno', 'list_category_img.txt')
  attrAnnFn   = os.path.join(args.dataset, 'Anno', 'list_attr_img.txt')
  catNamesFn  = os.path.join(args.dataset, 'Anno', 'list_category_cloth.txt')
  attrNamesFn = os.path.join(args.dataset, 'Anno', 'list_attr_cloth.txt')
  evalLabelFn = os.path.join(args.dataset, 'Eval', 'list_eval_partition.txt')

  bgFns = [] if args.background is None else scan_fns(args.background)

  images = readBBoxes(bboxAnnFn, None)
  #enrichWithGenCategoryAndLandmarks(images, lmAnnFn)
  enrichWithCategory(images,  catAnnFn, readCatNames(catNamesFn))

  """  
  stats, counter = analyseAnnotations(images, -2)
  print("DF stats:")
  for key in sorted(stats.keys()):
    print(key, stats[key]['presented'], stats[key]['boxCorrect'], stats[key]['boxInNPix'])
  print('Total images amount:', counter)
  quit()
  """

  enrichWithAttr(images, attrAnnFn, readAttrNames(attrNamesFn))  
  enrichWithEval(images, evalLabelFn)
  print("Loaded {} images".format(len(images)))

  trainFolder = makeFolder(os.path.join(args.output, 'train'))
  valFolder   = makeFolder(os.path.join(args.output, 'val'))
  testFolder  = makeFolder(os.path.join(args.output, 'test'))
  train, val, test = [], [], []
  counter    = 0
  bgCounter = 0
  for image in images:
    if notFullData(image): continue
    
    if len(bgFns) > 0: bgCounter = addBackground(image, imgFolder, bgFns, bgCounter)

    if   image['evalLabel'] == 'train': 
      path = put2Dataset(image, imgFolder, names, trainFolder)
      if path is not None: train.append(path)
    elif image['evalLabel'] == 'val':
      path = put2Dataset(image, imgFolder, names, valFolder)
      if path is not None: val.append(path)
    elif image['evalLabel'] == 'test':
      path = put2Dataset(image, imgFolder, names, testFolder)
      if path is not None: test.append(path)
    counter += 1
  print("{} images had full data".format(counter))

  path, count = saveList(train, trainFolder)
  print("{} images of train set has been saved to {}".format(count, path))
  path, count = saveList(val,   valFolder)
  print("{} images of val set has been saved to {}".format(count, path))
  path, count = saveList(test,  testFolder)
  print("{} images of test set has been saved to {}".format(count, path))


"""
def enrichWithGenCategoryAndLandmarks(images, lmAnnFn):
  lms = {}
  with open(lmAnnFn) as f:
    for line in itertools.islice(f, 4, len(images)+4):
      l = [[0,0,0]] * 6
      name, cl_type, l[0][0], l[0][1], l[0][2], \
                     l[1][0], l[1][1], l[1][2], \
                     l[2][0], l[2][1], l[2][2], \
                     l[3][0], l[3][1], l[3][2], \
                     l[4][0], l[4][1], l[4][2], \
                     l[5][0], l[5][1], l[5][2]  = line.split()
      lms[name] = {'clothes_type':cl_type, 'landmarks':l}
  for image in images:
    d = lms.get(image['fn'])
    image['clothes_type'] = d['clothes_type']
    image['landmarks']    = d['landmarks']

"""


