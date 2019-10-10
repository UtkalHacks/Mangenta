import tensorflow as tf
from utils import backbone
from api import object_counting_api

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

input_video = 1

detection_graph, category_index = backbone.set_model('ssdlite_mobilenet_v2_coco_2018_05_09')

targeted_objects = "person"
fps = 30
width = 854
height = 480
is_color_recognition_enabled = 0

object_counting_api.video(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height) # counting all the objects
