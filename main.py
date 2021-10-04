# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()
from moviepy.editor import *

from src.tracker import object_traking
from src.utils import *
from src.config import args


object_traking(folder_path= args.imgs_path, 
               original_video_path= 'original_vedio.avi', 
               resulted_video_path= 'resulted_video.avi', 
               view_predicted_frames = False
               )
 

 # visualise the original vedio
clip=VideoFileClip("./videos/" + 'original_vedio.avi')
clip.ipython_display(width=500)


# visualise the resulted video
clip=VideoFileClip('resulted_video.avi')
clip.ipython_display(width=500)