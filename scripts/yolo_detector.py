import sys
sys.path.insert(0, "/data/darknet/python")

import darknet as dn

class YoloDetector:
  def __init__(self, cfg, weights, labels, gpuId=0):
    dn.set_gpu(gpuId)
    self.net  = dn.load_net(cfg.encode('utf-8'), weights.encode('utf-8'), 0)
    self.meta = dn.load_names(labels.encode('utf-8'))

  def __call__(self, img, threshold):
    im = dn.nparray_to_image(img)
    print("Created np image")
    dets = dn.detect(self.net, self.meta, im, threshold)
    return self._fromYoloFormat(dets)

  @staticmethod
  def _fromYoloFormat(yoloDets):
    dets = []
    for det in yoloDets: # convert to better format
      label   = str(det[0].decode("utf-8"))
      score   = float(det[1])
      cx, cy  = int(det[2][0]), int(det[2][1])
      w,  h   = int(det[2][2]), int(det[2][3])
      l, t = int(cx - w/2.), int(cy - h/2.)
      r, b = int(l + w), int(t + h)
      dets.append({"label": label, "confidence": score, "box": {"left": l, "top": t, "right": r, "bottom": b}})
    return dets
