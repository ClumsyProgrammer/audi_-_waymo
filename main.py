
import display_sample
import preprocessing
import unet_unit
# import voxelnet_unit
import traffic_unit

# Preprocess data for Neural Networks
# preprocessing.preprocess()

# data_directory = "/home/katerina/Desktop/camera_lidar_semantic_bboxes_preprocessing/"
# images_directory = "/home/katerina/Desktop/camera_lidar_semantic_bboxes_preprocessing/images/"
# masks_directory = "/home/katerina/Desktop/camera_lidar_semantic_bboxes_preprocessing/masks/"
# lidar_directory = "/home/katerina/Desktop/camera_lidar_semantic_bboxes_preprocessing/lidar/"
# boxes_directory = "/home/katerina/Desktop/camera_lidar_semantic_bboxes_preprocessing/boxes/"
# output_directory = "/home/katerina/Desktop/camera_lidar_semantic_bboxes_preprocessing/output"

# Show example of data
# display_sample.display_sample_of_dataset()

# 1. Segmentation with UNet

# unet_unit.unet_unit()

# 2. Object Detection with VoxelNet

# voxelnet_unit.voxelnet_unit()

# 3. Traffic prediction

traffic_unit.predict_traffic()
