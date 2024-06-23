import os, sys
add_path = lambda path: sys.path.insert(0, path) if path not in sys.path else None
add_path(os.path.join(os.path.dirname(__file__), '..', 'python'))

import argparse
import darknet as dn
import cv2
import time

class YoloV3:
  def __init__(self, cfg, weights, labels, gpuId=0):
    dn.set_gpu(gpuId)
    self.net  = dn.load_net(cfg.encode('utf-8'), weights.encode('utf-8'), 0)
    self.meta = dn.load_names(labels.encode('utf-8'))

  def detect(self, img, threshold=.5):
    dets = dn.detect(self.net, self.meta, dn.nparray_to_image(img), threshold)
    return self._fromYoloFormat(dets)

  @staticmethod
  def drawBoxes(img, dets):
    for det in dets:
      cv2.rectangle(img, (det['topleft']['x'],    det['topleft']['y']),
                         (det['bottomright']['x'],det['bottomright']['y']),
                         (0,0,255), 2)
      YoloV3.drawText(img, det['label'], det['topleft']['x'], det['bottomright']['y'])
      print(det['label'], det['confidence'])

  @staticmethod
  def drawText(img, text, l, b):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = (0,0,255)
    lineType               = 2
    cv2.putText(img, text, (l,b), font, fontScale, fontColor, lineType)

  @staticmethod
  def _fromYoloFormat(yoloDets):
    boxesInfo = []
    for det in yoloDets: # convert to better format
      cls   = det[0].decode("utf-8")
      score = det[1]
      cx, cy = det[2][0], det[2][1]
      w,  h  = det[2][2], det[2][3]
#      print(cx, cy, w, h)
      magicOffset = 0  # -190 #-208   # TODO: why 190 ?
      l, t = int(cx - w/2.) + magicOffset, int(cy - h/2.) + magicOffset
      r, b = int(l + w), int(t + h)
#      box   = [l, t, r, b]
#      print(box)
      boxesInfo.append( {"label": cls, "confidence": score,
                         "topleft":     {"x": l, "y": t},
                         "bottomright": {"x": r, "y": b}} )
    return boxesInfo

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--model', type=str, default='weights/', help="Yolo weights filepath")
  parser.add_argument('-c', '--cfg',   type=str, default='cfg/',     help="Yolo cfg filepath")
  parser.add_argument('-n', '--names', type=str, default='data/',    help="Yolo .names filepath")
  parser.add_argument('-t', '--threshold', type=float, default=0.5,  help="Yolo threshold")
  parser.add_argument('-i', '--images',type=str, nargs='*', required=True, help="Images to classify")
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()
  yolo = YoloV3(args.cfg, args.model, args.names)
  win  = 'win'
  cv2.namedWindow(win)
  for i, fn in enumerate(args.images):
    print("Processing {} image from {}".format(i+1, len(args.images)))
    img  = cv2.imread(fn)
    st = time.time()
    dets = yolo.detect(img, args.threshold)
    timing = time.time() - st
    print("\tprocessed in {} msec".format(1000.*timing))
    yolo.drawBoxes(img, dets)
    cv2.imshow(win, img)
    key = cv2.waitKey()
    if key == 27: break
