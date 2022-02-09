
# configuration file
import cv2
import json
import open3d as o3
import transformations
import numpy as np
EPSILON = 1.0e-10 # norm should not be small
from os.path import join
import glob
import matplotlib.pylab as pt




def display_sample_of_dataset():

    # load the configuration file
    with open("/home/katerina/Desktop/camera_lidar_semantic_bboxes_preprocessing/cams_lidars.json", 'r') as f:
        config = json.load(f)



    # show examples of data
    root_path = '/home/katerina/Desktop/camera_lidar_semantic_bboxes_preprocessing/'
    # get the list of files in lidar directory
    file_names = sorted(glob.glob(join(root_path, '*/lidar/cam_front_center/*.npz')))
    # select the lidar point cloud
    file_name_lidar = file_names[0]
    # read the lidar data
    lidar_front_center = np.load(file_name_lidar)
    # create o3 object
    pcd_front_center = transformations.create_open3d_pc(lidar_front_center)
    o3.visualization.draw_geometries([pcd_front_center])

    # find the corresponding image
    seq_name = file_name_lidar.split('/')[5]
    file_name_image = transformations.extract_image_file_name_from_lidar_file_name(file_name_lidar)
    file_name_image = join(root_path, seq_name, 'camera/cam_front_center/', file_name_image)
    image_front_center = cv2.imread(file_name_image)
    image_front_center = cv2.cvtColor(image_front_center, cv2.COLOR_BGR2RGB)

    # display image from front center camera
    pt.fig = pt.figure(figsize=(15, 15))
    pt.imshow(image_front_center)
    pt.axis('off')
    pt.title('front center')
    pt.show()

    # display undistorted image
    undist_image_front_center = transformations.undistort_image(image_front_center, config,'front_center')
    pt.fig = pt.figure(figsize=(15, 15))
    pt.imshow(undist_image_front_center)
    pt.axis('off')
    pt.title('front center undistorted')
    pt.show()

    # find corresponding label
    file_name_semantic_label = transformations.extract_semantic_file_name_from_image_file_name(file_name_image)
    file_name_semantic_label = join(root_path, seq_name, 'label/cam_front_center/', file_name_semantic_label)
    label_front_center = cv2.imread(file_name_semantic_label)
    label_front_center = cv2.cvtColor(label_front_center, cv2.COLOR_BGR2RGB)
    pt.fig = pt.figure(figsize=(15, 15))
    pt.imshow(label_front_center)
    pt.axis('off')
    pt.title('front center label')
    pt.show()

    # display undistorted label
    undist_label_front_center = transformations.undistort_image(label_front_center, config,'front_center')
    pt.fig = pt.figure(figsize=(15, 15))
    pt.imshow(undist_label_front_center)
    pt.axis('off')
    pt.title('front center label undistorted')
    pt.show()

    # display point cloud with colors of image
    rows = (lidar_front_center['row'] + 0.5).astype(int)
    cols = (lidar_front_center['col'] + 0.5).astype(int)
    colours = image_front_center[rows, cols, :] / 255.0
    pcd_color = o3.geometry.PointCloud()
    pcd_color.points = o3.utility.Vector3dVector(lidar_front_center['points'])
    pcd_color.colors = o3.utility.Vector3dVector(colours)
    o3.visualization.draw_geometries([pcd_color])

    # display point cloud with colors of label
    rows = (lidar_front_center['row'] + 0.5).astype(int)
    cols = (lidar_front_center['col'] + 0.5).astype(int)
    colours = undist_label_front_center[rows, cols, :] / 255.0
    pcd_label = o3.geometry.PointCloud()
    pcd_label.points = o3.utility.Vector3dVector(lidar_front_center['points'])
    pcd_label.colors = o3.utility.Vector3dVector(colours)
    o3.visualization.draw_geometries([pcd_label])

    # 3D bounding boxes
    file_name_bboxes = transformations.extract_bboxes_file_name_from_image_file_name(file_name_image)
    file_name_bboxes = join(root_path, seq_name, 'label3D/cam_front_center/', file_name_bboxes)
    boxes = transformations.read_bounding_boxes(file_name_bboxes)
    points = transformations.get_points(boxes[0])

    # reflectance coloring
    pcd_front_center = transformations.create_open3d_pc(lidar_front_center)
    entities_to_draw = []
    entities_to_draw.append(pcd_front_center)

    for bbox in boxes:
        linesets = transformations._get_bboxes_wire_frames([bbox], color=(255, 0, 0))
        entities_to_draw.append(linesets[0])

    o3.visualization.draw_geometries(entities_to_draw)

    # label coloring
    pcd_lidar_colored = transformations.create_open3d_pc(lidar_front_center, undist_label_front_center)
    entities_to_draw = []
    entities_to_draw.append(pcd_lidar_colored)

    for bbox in boxes:
        linesets = transformations._get_bboxes_wire_frames([bbox], color=(255, 0, 0))
        entities_to_draw.append(linesets[0])

    o3.visualization.draw_geometries(entities_to_draw)







