# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import cv2
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

##
import torch
import matplotlib
from moviepy.editor import *

from .utils import *



def tracker(predictor, cfg, frames, view_predicted_frames = False):
  """
  track all object in the frames
  ---
  input: 
  predictor: the predictor we used for object tracking
  cfg: predictor configuration
  frames: list of images 
  view_predicted_frames: bool, tell weather to view the predicted frames of not.

  output:
  predicted_frames: list of predicted frames
  """

  # all the colors
  All_Colors = list(matplotlib.colors.cnames.keys())
  predicted_frames = []

  # start by the first frame
  im = cv2.imread(frames[0])

  outputs = predictor(im)

  previous_boxes = outputs['instances'].pred_boxes
  pred_classes = outputs['instances'].pred_classes
  pred_masks = outputs['instances'].pred_masks

  # label to class name 
  thing_classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes

  classes = list(map(lambda x: thing_classes[x], pred_classes.cpu().numpy()))
  num_of_classes = len(classes)
  previous_colors = All_Colors[-1:-1*(num_of_classes)*3:-3]

  # visualizer
  v = Visualizer(im[:, :, ::-1], scale=1.2)
  out = v.overlay_instances(boxes=previous_boxes.tensor.cpu().numpy(), masks=pred_masks.cpu().numpy(), labels=classes, assigned_colors=previous_colors)
  if view_predicted_frames:
    cv2_imshow(out.get_image()[:, :, ::-1])

  # append the predicted frames including the boxes
  predicted_frames.append(out.get_image()[:, :, ::-1])


  for im_path in frames[1: ]:

    im = cv2.imread(im_path)

    outputs = predictor(im)

    current_boxes = outputs['instances'].pred_boxes
    # scores = outputs['instances'].scores
    pred_classes = outputs['instances'].pred_classes
    pred_masks = outputs['instances'].pred_masks

    # thing_classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    classes = list(map(lambda x: thing_classes[x], pred_classes.cpu().numpy()))
    # num_of_classes = len(classes)
    # colors = All_Colors[5:(num_of_classes+5)*2:2]


    IOUs = detectron2.structures.pairwise_iou(current_boxes, previous_boxes)

    idx_matches = torch.argmax(IOUs, dim=1)

    for i,idx in enumerate(idx_matches):
      if IOUs[i][idx] < 0.4:
        idx_matches[i] = -1

    filtered_colors = list(filter(lambda x: x not in previous_colors, All_Colors))
    current_colors = []
    for idx in idx_matches:
      if idx == -1:
        filtered_colors = list(filter(lambda x: x not in current_colors, filtered_colors))
        current_colors.append(filtered_colors[-3])
      else:
        current_colors.append(previous_colors[idx])

    # update previous boxes and colors
    previous_boxes = current_boxes
    previous_colors = current_colors 

    # visualizer
    v = Visualizer(im[:, :, ::-1], scale=1.2)
    out = v.overlay_instances(boxes=previous_boxes.tensor.cpu().numpy(), masks=pred_masks.cpu().numpy(), labels=classes, assigned_colors=previous_colors)
    if view_predicted_frames:
      cv2_imshow(out.get_image()[:, :, ::-1])

    # append
    predicted_frames.append(out.get_image()[:, :, ::-1])

  return predicted_frames

def object_traking(folder_path, 
                   original_video_path, 
                   resulted_video_path, 
                   view_predicted_frames = False, 
                    ):
  
  """
  simple object tracker
  ---
  input: 
  folder_path: str, the path for the images folder
  original_video_path: str, the path in which we will save our original video
  resulted_video_path: str, the path in which we will save our resulted video 
  view_predicted_frames: bool, show the predicted frames while creating the video

 
  """

  # run the model
  seg_predictor, cfg = getModel()

  # fetch images paths
  # folder_path = '/content/clip/'
  images = fetch_frames(folder_path) 

  # save video of the original frames
  # original_video_path = 'original_vedio.avi'
  img_array = images_to_avi_video(images, original_video_path)

  # predict the new frames
  predicted_frames = tracker(seg_predictor, cfg, images, view_predicted_frames = view_predicted_frames)

  # save video of the predicted frames
  # resulted_video_path = 'resulted_video.avi'
  height, width, layers = predicted_frames[0].shape
  size = (width,height)
  out = cv2.VideoWriter(resulted_video_path ,cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
  # create the vedio'resulted_vedio.avi'
  for i in range(len(predicted_frames)):
      out.write(predicted_frames[i])
  # release the vedio
  out.release()
