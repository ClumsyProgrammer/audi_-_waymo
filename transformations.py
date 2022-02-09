# Transformation Functions provided with the a2d2 dataset

import numpy as np
import json
import numpy.linalg as la
import open3d as o3
import cv2
# matplotlib inline
import matplotlib.pylab as pt


EPSILON = 1.0e-10  # norm should not be small


def get_axes_of_a_view(view):
    x_axis = view['x-axis']
    y_axis = view['y-axis']

    x_axis_norm = la.norm(x_axis)
    y_axis_norm = la.norm(y_axis)

    if x_axis_norm < EPSILON or y_axis_norm < EPSILON:
        raise ValueError("Norm of input vector(s) too small.")

    # normalize the axes
    x_axis = x_axis / x_axis_norm
    y_axis = y_axis / y_axis_norm

    # make a new y-axis which lies in the original x-y plane, but is orthogonal to x-axis
    y_axis = y_axis - x_axis * np.dot(y_axis, x_axis)

    # create orthogonal z-axis
    z_axis = np.cross(x_axis, y_axis)

    # calculate and check y-axis and z-axis norms
    y_axis_norm = la.norm(y_axis)
    z_axis_norm = la.norm(z_axis)

    if (y_axis_norm < EPSILON) or (z_axis_norm < EPSILON):
        raise ValueError("Norm of view axis vector(s) too small.")

    # make x/y/z-axes orthonormal
    y_axis = y_axis / y_axis_norm
    z_axis = z_axis / z_axis_norm

    return x_axis, y_axis, z_axis


def get_origin_of_a_view(view):
    return view['origin']


def get_transform_to_global(view):
    # get axes
    x_axis, y_axis, z_axis = get_axes_of_a_view(view)

    # get origin
    origin = get_origin_of_a_view(view)
    transform_to_global = np.eye(4)

    # rotation
    transform_to_global[0:3, 0] = x_axis
    transform_to_global[0:3, 1] = y_axis
    transform_to_global[0:3, 2] = z_axis

    # origin
    transform_to_global[0:3, 3] = origin

    return transform_to_global


def get_transform_from_global(view):
    # get transform to global
    transform_to_global = get_transform_to_global(view)
    trans = np.eye(4)
    rot = np.transpose(transform_to_global[0:3, 0:3])
    trans[0:3, 0:3] = rot
    trans[0:3, 3] = np.dot(rot, -transform_to_global[0:3, 3])

    return trans


def get_rot_from_global(view):
    # get transform to global
    transform_to_global = get_transform_to_global(view)
    # get rotation
    rot = np.transpose(transform_to_global[0:3, 0:3])

    return rot


def get_rot_to_global(view):
    # get transform to global
    transform_to_global = get_transform_to_global(view)
    # get rotation
    rot = transform_to_global[0:3, 0:3]

    return rot


def rot_from_to(src, target):
    rot = np.dot(get_rot_from_global(target), get_rot_to_global(src))

    return rot


def transform_from_to(src, target):
    transform = np.dot(get_transform_from_global(target),
                       get_transform_to_global(src))

    return transform


# Create array of RGB colour values from the given array of reflectance values
def colours_from_reflectances(reflectances):
    return np.stack([reflectances, reflectances, reflectances], axis=1)


def create_open3d_pc(lidar, cam_image=None):
    # create open3d point cloud
    pcd = o3.geometry.PointCloud()

    # assign point coordinates
    pcd.points = o3.utility.Vector3dVector(lidar['points'])

    # assign colours
    if cam_image is None:
        median_reflectance = np.median(lidar['reflectance'])
        colours = colours_from_reflectances(lidar['reflectance']) / (median_reflectance * 5)

        # clip colours for visualisation on a white background
        colours = np.clip(colours, 0, 0.75)
    else:
        rows = (lidar['row'] + 0.5).astype(np.int)
        cols = (lidar['col'] + 0.5).astype(np.int)
        colours = cam_image[rows, cols, :] / 255.0

    pcd.colors = o3.utility.Vector3dVector(colours)

    return pcd


def project_lidar_from_to(lidar, src_view, target_view):
    lidar = dict(lidar)
    trans = transform_from_to(src_view, target_view)
    points = lidar['points']
    points_hom = np.ones((points.shape[0], 4))
    points_hom[:, 0:3] = points
    points_trans = (np.dot(trans, points_hom.T)).T
    lidar['points'] = points_trans[:, 0:3]

    return lidar





def extract_image_file_name_from_lidar_file_name(file_name_lidar):
    file_name_image = file_name_lidar.split('/')
    file_name_image = file_name_image[-1].split('.')[0]
    file_name_image = file_name_image.split('_')
    file_name_image = file_name_image[0] + '_' + \
                        'camera_' + \
                        file_name_image[2] + '_' + \
                        file_name_image[3] + '.png'

    return file_name_image


def undistort_image(image, config, cam_name):
    if cam_name in ['front_left', 'front_center',
                    'front_right', 'side_left',
                    'side_right', 'rear_center']:
        # get parameters from config file
        intr_mat_undist = \
            np.asarray(config['cameras'][cam_name]['CamMatrix'])
        intr_mat_dist = \
            np.asarray(config['cameras'][cam_name]['CamMatrixOriginal'])
        dist_parms = \
            np.asarray(config['cameras'][cam_name]['Distortion'])
        lens = config['cameras'][cam_name]['Lens']

        if (lens == 'Fisheye'):
            return cv2.fisheye.undistortImage(image, intr_mat_dist,
                                              D=dist_parms, Knew=intr_mat_undist)
        elif (lens == 'Telecam'):
            return cv2.undistort(image, intr_mat_dist,
                                 distCoeffs=dist_parms, newCameraMatrix=intr_mat_undist)
        else:
            return image
    else:
        return image






def read_image_info(file_name):
        with open(file_name, 'r') as f:
            image_info = json.load(f)

        return image_info


def plot_lidar_id_vs_delat_t(image_info, lidar):
    timestamps_lidar = lidar['timestamp']
    timestamp_camera = image_info['cam_tstamp']
    time_diff_in_sec = (timestamps_lidar - timestamp_camera) / (1e6)
    lidar_ids = lidar ['lidar_id']
    pt.fig = pt.figure(figsize=(15, 5))
    pt.plot(time_diff_in_sec, lidar_ids, 'go', ms=2)
    pt.grid(True)
    ticks = np.arange(len(image_info['lidar_ids'].keys()))
    ticks_name = []
    for key in ticks:
        ticks_name.append(image_info['lidar_ids'][str(key)])
    pt.yticks(ticks, tuple(ticks_name))
    pt.ylabel('LiDAR sensor')
    pt.xlabel('delta t in sec')
    pt.title(image_info['cam_name'])
    pt.show()


def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v

    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6

    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q


def map_lidar_points_onto_image(image_orig, lidar, pixel_size=3, pixel_opacity=1):
    image = np.copy(image_orig)

    # get rows and cols
    rows = (lidar['row'] + 0.5).astype(np.int)
    cols = (lidar['col'] + 0.5).astype(np.int)

    # lowest distance values to be accounted for in colour code
    MIN_DISTANCE = np.min(lidar['distance'])
    # largest distance values to be accounted for in colour code
    MAX_DISTANCE = np.max(lidar['distance'])

    # get distances
    distances = lidar['distance']
    # determine point colours from distance
    colours = (distances - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
    colours = np.asarray([np.asarray(hsv_to_rgb(0.75 * c, \
                                                np.sqrt(pixel_opacity), 1.0)) for c in colours])
    pixel_rowoffs = np.indices([pixel_size, pixel_size])[0] - pixel_size // 2
    pixel_coloffs = np.indices([pixel_size, pixel_size])[1] - pixel_size // 2
    canvas_rows = image.shape[0]
    canvas_cols = image.shape[1]
    for i in range(len(rows)):
        pixel_rows = np.clip(rows[i] + pixel_rowoffs, 0, canvas_rows - 1)
        pixel_cols = np.clip(cols[i] + pixel_coloffs, 0, canvas_cols - 1)
        image[pixel_rows, pixel_cols, :] = \
            (1. - pixel_opacity) * \
            np.multiply(image[pixel_rows, pixel_cols, :], \
                        colours[i]) + pixel_opacity * 255 * colours[i]
    return image.astype(np.uint8)





def extract_semantic_file_name_from_image_file_name(file_name_image):
    file_name_semantic_label = file_name_image.split('/')
    file_name_semantic_label = file_name_semantic_label[-1].split('.')[0]
    file_name_semantic_label = file_name_semantic_label.split('_')
    file_name_semantic_label = file_name_semantic_label[0] + '_' + \
                               'label_' + \
                               file_name_semantic_label[2] + '_' + \
                               file_name_semantic_label[3] + '.png'

    return file_name_semantic_label


def skew_sym_matrix(u):
    return np.array([[    0, -u[2],  u[1]],
                     [ u[2],     0, -u[0]],
                     [-u[1],  u[0],    0]])

def axis_angle_to_rotation_mat(axis, angle):
    return np.cos(angle) * np.eye(3) + \
        np.sin(angle) * skew_sym_matrix(axis) + \
        (1 - np.cos(angle)) * np.outer(axis, axis)



def read_bounding_boxes(file_name_bboxes):
    # open the file
    with open(file_name_bboxes, 'r') as f:
        bboxes = json.load(f)

    boxes = []  # a list for containing bounding boxes
    # print(bboxes.keys())

    for bbox in bboxes.keys():
        bbox_read = {}  # a dictionary for a given bounding box
        bbox_read['class'] = bboxes[bbox]['class']
        bbox_read['truncation'] = bboxes[bbox]['truncation']
        bbox_read['occlusion'] = bboxes[bbox]['occlusion']
        bbox_read['alpha'] = bboxes[bbox]['alpha']
        bbox_read['top'] = bboxes[bbox]['2d_bbox'][0]
        bbox_read['left'] = bboxes[bbox]['2d_bbox'][1]
        bbox_read['bottom'] = bboxes[bbox]['2d_bbox'][2]
        bbox_read['right'] = bboxes[bbox]['2d_bbox'][3]
        bbox_read['center'] = np.array(bboxes[bbox]['center'])
        bbox_read['size'] = np.array(bboxes[bbox]['size'])
        bbox_read['rot_angle'] = np.array(bboxes[bbox]['rot_angle'])
        bbox_read['axis'] = np.array(bboxes[bbox]['axis'])
        angle = bboxes[bbox]['rot_angle']
        axis = np.array(bboxes[bbox]['axis'])
        bbox_read['rotation'] = axis_angle_to_rotation_mat(axis, angle)
        boxes.append(bbox_read)

    return boxes


def extract_bboxes_file_name_from_image_file_name(file_name_image):
    file_name_bboxes = file_name_image.split('/')
    file_name_bboxes = file_name_bboxes[-1].split('.')[0]
    file_name_bboxes = file_name_bboxes.split('_')
    file_name_bboxes = file_name_bboxes[0] + '_' + \
                       'label3D_' + \
                       file_name_bboxes[2] + '_' + \
                       file_name_bboxes[3] + '.json'

    return file_name_bboxes


def get_points(bbox):
    half_size = bbox['size'] / 2.

    if half_size[0] > 0:
        # calculate unrotated corner point offsets relative to center
        brl = np.asarray([-half_size[0], +half_size[1], -half_size[2]])
        bfl = np.asarray([+half_size[0], +half_size[1], -half_size[2]])
        bfr = np.asarray([+half_size[0], -half_size[1], -half_size[2]])
        brr = np.asarray([-half_size[0], -half_size[1], -half_size[2]])
        trl = np.asarray([-half_size[0], +half_size[1], +half_size[2]])
        tfl = np.asarray([+half_size[0], +half_size[1], +half_size[2]])
        tfr = np.asarray([+half_size[0], -half_size[1], +half_size[2]])
        trr = np.asarray([-half_size[0], -half_size[1], +half_size[2]])

        # rotate points
        points = np.asarray([brl, bfl, bfr, brr, trl, tfl, tfr, trr])
        points = np.dot(points, bbox['rotation'].T)

        # add center position
        points = points + bbox['center']

    return points



# Create or update open3d wire frame geometry for the given bounding boxes
def _get_bboxes_wire_frames(bboxes, linesets=None, color=None):
    num_boxes = len(bboxes)

    # initialize linesets, if not given
    if linesets is None:
        linesets = [o3.geometry.LineSet() for _ in range(num_boxes)]

    # set default color
    if color is None:
        # color = [1, 0, 0]
        color = [0, 0, 1]

    assert len(linesets) == num_boxes, "Number of linesets must equal number of bounding boxes"

    # point indices defining bounding box edges
    lines = [[0, 1], [1, 2], [2, 3], [3, 0],
             [0, 4], [1, 5], [2, 6], [3, 7],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [5, 2], [1, 6]]

    # loop over all bounding boxes
    for i in range(num_boxes):
        # get bounding box corner points
        points = get_points(bboxes[i])
        # update corresponding Open3d line set
        colors = [color for _ in range(len(lines))]
        line_set = linesets[i]
        line_set.points = o3.utility.Vector3dVector(points)
        line_set.lines = o3.utility.Vector2iVector(lines)
        line_set.colors = o3.utility.Vector3dVector(colors)

    return linesets






































