# Object_Tracker_with_Detectron2
In-class project of the Computer Vision Part 2 course during [AMMI](https://aimsammi.org/) master 2021.
## The project summary
Make use of the pretrained "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" model form detectron2 library to implement object traicking system. 
## Dependencies
```
!pip install pyyaml==5.1
!pip install torch==1.8.1
!pip install torchvision==0.9.1
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
```
## Test the model 
```
!python main.py
```
## Sample of the result
#### Original video
![alt text](https://github.com/ashrafhatim/Object-_Tracker_with_Detectron2/blob/main/videos/original.mp4)
#### Resulted video
![alt text](https://github.com/ashrafhatim/Object-_Tracker_with_Detectron2/blob/main/videos/resulted.mp4)
## Different data
Change the data in data folder, the inputs is the frames (.jpg) of the video.
