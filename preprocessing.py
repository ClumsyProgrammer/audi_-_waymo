import cv2
import transformations
import json
import pprint
import numpy as np
from colormap import rgb2hex
import time
from multiprocessing import Pool
from pathlib import Path

EPSILON = 1.0e-10  # norm should not be small


# call preprocessing.preprocess() from main.py


def rgb_to_class(undist_image, classes):
    height, width, channels = undist_image.shape
    label = np.zeros([height, width])
    pprint.pprint(label.shape)

    for i in range(height):
        for j in range(width):
            pixel_val = rgb2hex(undist_image[i, j, 0], undist_image[i, j, 1],
                                undist_image[i, j, 2])
            for num, (col, cl) in enumerate(classes.items()):
                if pixel_val.lower() == col.lower():
                    label[i, j] = num
    return label


def mask_to_class(path):
    start1 = time.time()

    with open("/home/katerina/Desktop/camera_lidar_semantic_bboxes_preprocessing/cams_lidars.json", 'r') as f1:
        config = json.load(f1)
    with open("/home/katerina/Desktop/camera_lidar_semantic_bboxes_preprocessing/class_list.json", 'r') as f2:
        classes = json.load(f2)
    masks_directory = "/home/katerina/Desktop/camera_lidar_semantic_bboxes_preprocessing/masks/"

    # because path is object not string
    path_in_str = str(path)
    new_file_name = path_in_str.split('/')
    seq_number = new_file_name[5]
    new_file_name = path_in_str.split('_')
    new_file_name_image = masks_directory + seq_number + '_' + new_file_name[10]
    pprint.pprint(new_file_name_image)
    image_front_center = cv2.imread(path_in_str)
    image_front_center = cv2.cvtColor(image_front_center, cv2.COLOR_BGR2RGB)
    undist_image = transformations.undistort_image(image_front_center, config, 'front_center')

    label = rgb_to_class(undist_image, classes)

    pprint.pprint(label)
    np.save(new_file_name_image, label)
    end1 = time.time()
    print("[INFO] total time taken to create one mask: {:.2f}s".format(end1 - start1))


def preprocess():
    with open("/home/katerina/Desktop/camera_lidar_semantic_bboxes_preprocessing/cams_lidars.json", 'r') as f1:
        config = json.load(f1)

    data_directory = "/home/katerina/Desktop/camera_lidar_semantic_bboxes_preprocessing/"
    images_directory = "/home/katerina/Desktop/camera_lidar_semantic_bboxes_preprocessing/images/"
    masks_directory = "/home/katerina/Desktop/camera_lidar_semantic_bboxes_preprocessing/masks/"
    lidar_directory = "/home/katerina/Desktop/camera_lidar_semantic_bboxes_preprocessing/lidar/"
    boxes_directory = "/home/katerina/Desktop/camera_lidar_semantic_bboxes_preprocessing/boxes/"
    output_directory = "/home/katerina/Desktop/camera_lidar_semantic_bboxes_preprocessing/output"

    pprint.pprint('____________________________________________________________________________________')

    # Undistort and save all images of camera
    start = time.time()

    pathlist = sorted(Path(data_directory).glob('*/camera/*/*.png'))
    for path in pathlist:
        # because path is object not string
        path_in_str = str(path)

        new_file_name = path_in_str.split('/')
        seq_number = new_file_name[5]
        new_file_name = path_in_str.split('_')
        new_file_name_image = images_directory + seq_number + '_' + new_file_name[10]
        pprint.pprint(new_file_name_image)
        image_front_center = cv2.imread(path_in_str)
        image_front_center = cv2.cvtColor(image_front_center, cv2.COLOR_BGR2RGB)
        undist_image = transformations.undistort_image(image_front_center, config, 'front_center')
        undist_image = cv2.cvtColor(undist_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(new_file_name_image, undist_image)

    end = time.time()
    print("[INFO] total time taken to undistort the images: {:.2f}s".format(end - start))

    pprint.pprint('____________________________________________________________________________________')

    # Undistort and save all images of masks in Class Code in .npy
    start = time.time()

    pathlist = sorted(Path(data_directory).glob('*/label/*/*.png'))
    # for path in pathlist:
    p = Pool(12)
    p.map(mask_to_class, pathlist)

    end = time.time()
    print("[INFO] total time taken to convert masks to classes: {:.2f}s".format(end - start))

    pprint.pprint('____________________________________________________________________________________')

    # Color point clouds with classes' number
    start = time.time()

    # get the list of files in lidar directory
    pathlist = sorted(Path(data_directory).glob('*/lidar/*/*.npz'))
    for path in pathlist:
        # because path is object not string
        path_in_str = str(path)
        # pprint.pprint(path_in_str)
        new_file_name = path_in_str.split('/')
        seq_number = new_file_name[5]
        new_file_name = path_in_str.split('_')
        new_file_name_lidar = lidar_directory + seq_number + '_' + new_file_name[10]
        new_file_name_lidar = new_file_name_lidar.replace(".npz", "")
        pprint.pprint(new_file_name_lidar)

        lidar = np.load(path_in_str)

        file_name_image = masks_directory + seq_number + '_' + new_file_name[10]
        file_name_image = file_name_image.replace(".npz", ".png.npy")
        pprint.pprint(file_name_image)

        label = np.load(file_name_image)

        pprint.pprint(lidar['row'].shape)
        pprint.pprint(lidar['col'].shape)
        pprint.pprint(lidar['points'].shape)

        rows = (lidar['row'] + 0.5).astype(np.int)
        cols = (lidar['col'] + 0.5).astype(np.int)

        labels = label[rows, cols]

        # pprint.pprint(labels.shape)
        # pprint.pprint(labels.size)

        class_lidar = np.column_stack((lidar['points'], labels))
        # pprint.pprint(class_lidar.shape)
        # pprint.pprint(class_lidar)

        # for the point cloud only points and class codes are used
        np.save(new_file_name_lidar, class_lidar)

    end = time.time()
    print("[INFO] total time taken to get point clouds: {:.2f}s".format(end - start))

    pprint.pprint('____________________________________________________________________________________')

    # Prepare Bounding Boxes
    start = time.time()

    # get the list of files in bboxes directory
    pathlist = sorted(Path(data_directory).glob('*/label3D/*/*.json'))
    for path in pathlist:
        # because path is object not string
        path_in_str = str(path)
        # pprint.pprint(path_in_str)
        new_file_name = path_in_str.split('/')
        seq_number = new_file_name[5]
        new_file_name = path_in_str.split('_')
        new_file_name_boxes = boxes_directory + seq_number + '_' + new_file_name[10]
        new_file_name_boxes = new_file_name_boxes.replace(".json", "")
        pprint.pprint(new_file_name_boxes)

        boxes = transformations.read_bounding_boxes(path_in_str)
        # pprint.pprint(boxes)

        frame_boxes = []
        for bbox in boxes:

            angle = (bbox['rot_angle']*bbox['axis'][2])
            box = np.asarray([bbox['center'][0], bbox['center'][1], bbox['center'][2],
                              bbox['size'][0], bbox['size'][1], bbox['size'][2], angle])
            frame_boxes.append(box)

        pprint.pprint(frame_boxes)
        np.save(new_file_name_boxes, frame_boxes)

    end = time.time()
    print("[INFO] total time taken to prepare bounding boxes: {:.2f}s".format(end - start))

    pprint.pprint('____________________________________________________________________________________')
