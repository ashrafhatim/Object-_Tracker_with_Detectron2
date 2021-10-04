# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import cv2
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

##
import glob
from moviepy.editor import *


def getModel(COCOinit = True):
  """
  get the trained model
  ---
  input: 
  COCOinit: bool, true for cocoinit inistialisation

  output:
  seg_predictor: the trained model
  cfg: the trained model setting
  """
  cfg = get_cfg()
  # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
  cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
  # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
  if COCOinit:
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
  seg_predictor = DefaultPredictor(cfg)

  return seg_predictor, cfg

def fetch_frames(folder_path):
  """
  fetch the images paths form the memory
  ---
  input: 
  folder_path: str, the folde path

  output:
  list[str]: list of the images paths
  """
  # fetch the frames with glop, put the frames in the correct order
  images = glob.glob(folder_path +'*.jpg')
  images.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]) )
  return images


def images_to_avi_video(images, path):
  """
  create an avi video from the images
  ---
  input: 
  images: list of the images paths
  path: the path to used to savee the video 

  output:
  img_array: list of the images
  """
  img_array = []
  # append frames to im_array
  for filename in images:
      img = cv2.imread(filename)
      height, width, layers = img.shape
      size = (width,height)
      img_array.append(img)
  # video writer object
  out = cv2.VideoWriter(path ,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
  # create the vedio'resulted_vedio.avi'
  for i in range(len(img_array)):
      out.write(img_array[i])
  # release the vedio
  out.release()

  return img_array