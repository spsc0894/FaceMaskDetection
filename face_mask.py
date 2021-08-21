import os
import numpy as np
# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')
import os
import cv2
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time
from albumentations.pytorch import ToTensorV2
from threading import Thread


def get_object_detection_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

model=get_object_detection_model(4).cuda()
model.load_state_dict(torch.load('FaceMaskDetection_FRCNN.pt'))


mapping=['_','with_mask','without_mask','mask_weared_incorrect']

def draw_boxes(img,boxes,labels,scores,fps):
    for i,b in enumerate(boxes):
        label=mapping[labels[i]]
        if label=='with_mask':
            color=(0,255,0)
        elif label=='without_mask':
            color=(0, 0, 255)
        else:
            color=(255, 255, 0)
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), color, 2)
        cv2.putText(img, label + str(int(scores[i] * 100)) + '%', (b[0], b[1]), 1, 1, color, 1)
    #cv2.putText(img, 'FPS:' + str(int(fps)) , (18,70), 1, 2, (0,255,255), 1)
    return img


def start_detections(model):
    start_time=0.0
    end_time=0.0
    with torch.no_grad():
      model.eval()
      cap=cv2.VideoCapture(0)
      while cap.isOpened():
          ret,frame=cap.read()
          frame=cv2.resize(frame,(480,480),interpolation=cv2.INTER_AREA)
          img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
          img = img / 255.0
          img=ToTensorV2()(image=img)['image']
          inp=img.type(torch.FloatTensor).unsqueeze(0)
          outputs = model(inp.cuda())
          boxes = outputs[0]['boxes'].data.cpu().numpy()
          scores = outputs[0]['scores'].data.cpu().numpy()
          boxes = boxes[scores>=0.9].astype(np.int32)
          scores = scores[scores >= 0.9]
          labels= outputs[0]['labels']
          end_time=time.time()
          fps=1/(end_time-start_time)
          start_time=end_time
          output=draw_boxes(frame,boxes,labels,scores,fps)
          cv2.imshow("output",output)
          if cv2.waitKey(10) & 0xFF ==ord('q'):
              break
      cap.release()
      cv2.destroyAllWindows()
start_detections(model)