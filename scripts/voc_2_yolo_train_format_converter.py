"""
Convert dataset represented in VOC format into Yolo format:
Input for the script:
1) folders with xml annotations and images;
2) txt file with labelnames (one of yolo format should be taken).
Output of the script:
1) folder with yolo annotations;
2) list of filepaths of yolo format.
The script must be run onto the instance where training will be run.
"""

import xml.etree.ElementTree as ET
import os, argparse
#voc_classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

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

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset', type=str, required=True, help="VOC root folder (with 'annotations', 'images', etc.")
  parser.add_argument('-n', '--names',   type=str, required=True, help="Yolo file with labelnames of interest")
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()
  names = [el for el in os.listdir(os.path.join(args.dataset, 'images'))]
  labels = open(args.names).read().splitlines()
  annFolder = os.path.join(args.dataset, 'annotations')
  imgFolder = os.path.join(args.dataset, 'images')
  yoloAnnFolder = os.path.join(args.dataset, 'labels')
  if not os.path.exists(yoloAnnFolder): os.makedirs(yoloAnnFolder)
  yoloLstFn = os.path.join(args.dataset, 'yolo.txt')

  f = open(yoloLstFn, 'w')
  for name in names:
    nameCore = '.'.join(name.split('.')[:-1])
    annFn     = os.path.join(annFolder,     nameCore + '.xml')
    yoloAnnFn = os.path.join(yoloAnnFolder, nameCore + '.txt')
    if convert_annotation(annFn, yoloAnnFn, labels):
      path = os.path.join(imgFolder, name)
      f.write(os.path.join(imgFolder, name) + '\n')
  f.close()
  print("Yolo annotations have been saved to " + yoloAnnFolder)
  print("Yolo list of filenames has been saved to " + yoloLstFn)