
# https://github.com/skyhehe123/VoxelNet-pytorch

# voxelnet


import torch.utils.data as data
import time

import torch.optim as optim
import torch.nn.init as init
import torch.backends.cudnn
from __future__ import division
import numpy as np
import math
import mayavi.mlab as mlab
import cv2
import cv2


from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import os
import cv2
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import torch.backends.cudnn
import cv2
import matplotlib.pyplot as plt
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


# voxel configuration

class config:

    # classes
    class_list = ['Car', 'Bicycle', 'Pedestrian', 'Truck',
                  'Small vehicles', 'Traffic signal', 'Traffic sign',
                  'Utility vehicle', 'Sidebars', 'Speed bumper',
                  'Curbstone', 'Solid line', 'Irrelevant signs',
                  'Road blocks', 'Tractor', 'Non-drivable street',
                  'Zebra crossing', 'Obstacles / trash', 'Poles',
                  'RD restricted area', 'Animals', 'Grid structure',
                  'Signal corpus', 'Drivable cobblestone', 'Electronic traffic',
                  'Slow drive area', 'Nature object', 'Parking area',
                  'Sidewalk', 'Ego car', 'Painted driv. instr.',
                  'Traffic guide obj.', 'Dashed line', 'RD normal street',
                  'Sky', 'Buildings', 'Blurred area', 'Rain dirt']

    # batch size
    N = 2

    # maxiumum number of points per voxel
    T = 35

    # voxel size
    vd = 0.4
    vh = 0.2
    vw = 0.2

    # points cloud range
    xrange = (0, 100)
    yrange = (-50, 50)
    zrange = (-5, 5)

    # voxel grid
    W = math.ceil((xrange[1] - xrange[0]) / vw)
    H = math.ceil((yrange[1] - yrange[0]) / vh)
    D = math.ceil((zrange[1] - zrange[0]) / vd)

    # iou threshold
    pos_threshold = 0.6
    neg_threshold = 0.45

    #   anchors: (200, 176, 2, 7) x y z h w l r
    x = np.linspace(xrange[0]+vw, xrange[1]-vw, W/2)
    y = np.linspace(yrange[0]+vh, yrange[1]-vh, H/2)
    cx, cy = np.meshgrid(x, y)
    # all is (w, l, 2)
    cx = np.tile(cx[..., np.newaxis], 2)
    cy = np.tile(cy[..., np.newaxis], 2)
    cz = np.ones_like(cx) * -1.0
    w = np.ones_like(cx) * 1.6
    l = np.ones_like(cx) * 3.9
    h = np.ones_like(cx) * 1.56
    r = np.ones_like(cx)
    r[..., 0] = 0
    r[..., 1] = np.pi/2
    anchors = np.stack([cx, cy, cz, h, w, l, r], axis=-1)

    anchors_per_position = 2

    # non-maximum suppression
    nms_threshold = 0.1
    score_threshold = 0.96


# _____________________________________________________________________________________________________ #


class A2D2Dataset(data.Dataset):

    def __init__(self, root='./KITTI',set='train',type='velodyne_train'):
        self.type = type
        self.root = root
        self.data_path = os.path.join(root, 'training')
        self.lidar_path = os.path.join(self.data_path, "crop/")
        self.image_path = os.path.join(self.data_path, "image_2/")
        self.calib_path = os.path.join(self.data_path, "calib/")
        self.label_path = os.path.join(self.data_path, "label_2/")

        with open(os.path.join(self.data_path, '%s.txt' % set)) as f:
            self.file_list = f.read().splitlines()

        self.T = config.T
        self.vd = config.vd
        self.vh = config.vh
        self.vw = config.vw
        self.xrange = config.xrange
        self.yrange = config.yrange
        self.zrange = config.zrange
        self.anchors = config.anchors.reshape(-1,7)
        self.feature_map_shape = (int(config.H / 2), int(config.W / 2))
        self.anchors_per_position = config.anchors_per_position
        self.pos_threshold = config.pos_threshold
        self.neg_threshold = config.neg_threshold

    def cal_target(self, gt_box3d):
        # Input:
        #   labels: (N,)
        #   feature_map_shape: (w, l)
        #   anchors: (w, l, 2, 7)
        # Output:
        #   pos_equal_one (w, l, 2)
        #   neg_equal_one (w, l, 2)
        #   targets (w, l, 14)
        # attention: cal IoU on birdview

        anchors_d = np.sqrt(self.anchors[:, 4] ** 2 + self.anchors[:, 5] ** 2)

        pos_equal_one = np.zeros((*self.feature_map_shape, 2))
        neg_equal_one = np.zeros((*self.feature_map_shape, 2))
        targets = np.zeros((*self.feature_map_shape, 14))

        gt_xyzhwlr = box3d_corner_to_center_batch(gt_box3d)

        anchors_corner = anchors_center_to_corner(self.anchors)

        anchors_standup_2d = corner_to_standup_box2d_batch(anchors_corner)
        # BOTTLENECK
        gt_standup_2d = corner_to_standup_box2d_batch(gt_box3d)

        iou = bbox_overlaps(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )

        id_highest = np.argmax(iou.T, axis=1)  # the maximum anchor's ID
        id_highest_gt = np.arange(iou.T.shape[0])
        mask = iou.T[id_highest_gt, id_highest] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]
        # find anchor iou > cfg.XXX_POS_IOU
        id_pos, id_pos_gt = np.where(iou > self.pos_threshold)
        # find anchor iou < cfg.XXX_NEG_IOU
        id_neg = np.where(np.sum(iou < self.neg_threshold,
                                 axis=1) == iou.shape[1])[0]

        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])
        # TODO: uniquify the array in a more scientific way
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()
        # cal the target and set the equal one
        index_x, index_y, index_z = np.unravel_index(
            id_pos, (*self.feature_map_shape, self.anchors_per_position))
        pos_equal_one[index_x, index_y, index_z] = 1
        # ATTENTION: index_z should be np.array

        targets[index_x, index_y, np.array(index_z) * 7] = \
            (gt_xyzhwlr[id_pos_gt, 0] - self.anchors[id_pos, 0]) / anchors_d[id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 1] = \
            (gt_xyzhwlr[id_pos_gt, 1] - self.anchors[id_pos, 1]) / anchors_d[id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 2] = \
            (gt_xyzhwlr[id_pos_gt, 2] - self.anchors[id_pos, 2]) / self.anchors[id_pos, 3]
        targets[index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
            gt_xyzhwlr[id_pos_gt, 3] / self.anchors[id_pos, 3])
        targets[index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
            gt_xyzhwlr[id_pos_gt, 4] / self.anchors[id_pos, 4])
        targets[index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
            gt_xyzhwlr[id_pos_gt, 5] / self.anchors[id_pos, 5])
        targets[index_x, index_y, np.array(index_z) * 7 + 6] = (
                gt_xyzhwlr[id_pos_gt, 6] - self.anchors[id_pos, 6])
        index_x, index_y, index_z = np.unravel_index(
            id_neg, (*self.feature_map_shape, self.anchors_per_position))
        neg_equal_one[index_x, index_y, index_z] = 1
        # to avoid a box be pos/neg in the same time
        index_x, index_y, index_z = np.unravel_index(
            id_highest, (*self.feature_map_shape, self.anchors_per_position))
        neg_equal_one[index_x, index_y, index_z] = 0

        return pos_equal_one, neg_equal_one, targets

    def preprocess(self, lidar):

        # shuffling the points
        np.random.shuffle(lidar)

        voxel_coords = ((lidar[:, :3] - np.array([self.xrange[0], self.yrange[0], self.zrange[0]])) / (
                        self.vw, self.vh, self.vd)).astype(np.int32)

        # convert to  (D, H, W)
        voxel_coords = voxel_coords[:,[2,1,0]]
        voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords, axis=0, \
                                                  return_inverse=True, return_counts=True)

        voxel_features = []

        for i in range(len(voxel_coords)):
            voxel = np.zeros((self.T, 7), dtype=np.float32)
            pts = lidar[inv_ind == i]
            if voxel_counts[i] > self.T:
                pts = pts[:self.T, :]
                voxel_counts[i] = self.T
            # augment the points
            voxel[:pts.shape[0], :] = np.concatenate((pts, pts[:, :3] - np.mean(pts[:, :3], 0)), axis=1)
            voxel_features.append(voxel)
        return np.array(voxel_features), voxel_coords

    def __getitem__(self, i):

        lidar_file = self.lidar_path + '/' + self.file_list[i] + '.bin'
        calib_file = self.calib_path + '/' + self.file_list[i] + '.txt'
        label_file = self.label_path + '/' + self.file_list[i] + '.txt'
        image_file = self.image_path + '/' + self.file_list[i] + '.png'

        calib = load_kitti_calib(calib_file)
        Tr = calib['Tr_velo2cam']
        gt_box3d = load_kitti_label(label_file, Tr)
        lidar = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)


        if self.type == 'velodyne_train':
            image = cv2.imread(image_file)

            # data augmentation
            lidar, gt_box3d = aug_data(lidar, gt_box3d)

            # specify a range
            lidar, gt_box3d = get_filtered_lidar(lidar, gt_box3d)

            # voxelize
            voxel_features, voxel_coords = self.preprocess(lidar)

            # bounding-box encoding
            pos_equal_one, neg_equal_one, targets = self.cal_target(gt_box3d)

            return voxel_features, voxel_coords, pos_equal_one, neg_equal_one, targets, image, calib, self.file_list[i]

        elif self.type == 'velodyne_test':
            NotImplemented

        else:
            raise ValueError('the type invalid')


    def __len__(self):
        return len(self.file_list)





# _____________________________________________________________________________________________________ #


















# conv2d + bn + relu
class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, activation=True, batch_norm=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x=self.bn(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x


# conv3d + bn + relu
class Conv3d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, batch_norm=True):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)

        return F.relu(x, inplace=True)


# Fully Connected Network
class FCN(nn.Module):

    def __init__(self, cin, cout):
        super(FCN, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        # KK is the stacked k across batch
        kk, t, _ = x.shape
        x = self.linear(x.view(kk*t, -1))
        x = F.relu(self.bn(x))
        return x.view(kk, t, -1)


# Voxel Feature Encoding layer
class VFE(nn.Module):

    def __init__(self,cin,cout):
        super(VFE, self).__init__()
        assert cout % 2 == 0
        self.units = cout // 2
        self.fcn = FCN(cin,self.units)

    def forward(self, x, mask):
        # point-wise feature
        pwf = self.fcn(x)
        # locally aggregated feature
        laf = torch.max(pwf, 1)[0].unsqueeze(1).repeat(1, config.T, 1)
        # point-wise concat feature
        pwcf = torch.cat((pwf, laf), dim=2)
        # apply mask
        mask = mask.unsqueeze(2).repeat(1, 1, self.units * 2)
        pwcf = pwcf * mask.float()

        return pwcf


# Stacked Voxel Feature Encoding
class SVFE(nn.Module):

    def __init__(self):
        super(SVFE, self).__init__()
        self.vfe_1 = VFE(7, 32)
        self.vfe_2 = VFE(32, 128)
        self.fcn = FCN(128, 128)

    def forward(self, x):
        mask = torch.ne(torch.max(x, 2)[0], 0)
        x = self.vfe_1(x, mask)
        x = self.vfe_2(x, mask)
        x = self.fcn(x)
        # element-wise max pooling
        x = torch.max(x, 1)[0]
        return x


# Convolutional Middle Layer
class CML(nn.Module):
    def __init__(self):
        super(CML, self).__init__()
        self.conv3d_1 = Conv3d(128, 64, 3, s=(2, 1, 1), p=(1, 1, 1))
        self.conv3d_2 = Conv3d(64, 64, 3, s=(1, 1, 1), p=(0, 1, 1))
        self.conv3d_3 = Conv3d(64, 64, 3, s=(2, 1, 1), p=(1, 1, 1))

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        return x


# Region Proposal Network
class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        self.block_1 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_1 += [Conv2d(128, 128, 3, 1, 1) for _ in range(3)]
        self.block_1 = nn.Sequential(*self.block_1)

        self.block_2 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_2 += [Conv2d(128, 128, 3, 1, 1) for _ in range(5)]
        self.block_2 = nn.Sequential(*self.block_2)

        self.block_3 = [Conv2d(128, 256, 3, 2, 1)]
        self.block_3 += [nn.Conv2d(256, 256, 3, 1, 1) for _ in range(5)]
        self.block_3 = nn.Sequential(*self.block_3)

        self.deconv_1 = nn.Sequential(nn.ConvTranspose2d(256, 256, 4, 4, 0), nn.BatchNorm2d(256))
        self.deconv_2 = nn.Sequential(nn.ConvTranspose2d(128, 256, 2, 2, 0), nn.BatchNorm2d(256))
        self.deconv_3 = nn.Sequential(nn.ConvTranspose2d(128, 256, 1, 1, 0), nn.BatchNorm2d(256))

        self.score_head = Conv2d(768, config.anchors_per_position, 1, 1, 0, activation=False, batch_norm=False)
        self.reg_head = Conv2d(768, 7 * config.anchors_per_position, 1, 1, 0, activation=False, batch_norm=False)

    def forward(self,x):
        x = self.block_1(x)
        x_skip_1 = x
        x = self.block_2(x)
        x_skip_2 = x
        x = self.block_3(x)
        x_0 = self.deconv_1(x)
        x_1 = self.deconv_2(x_skip_2)
        x_2 = self.deconv_3(x_skip_1)
        x = torch.cat((x_0,x_1,x_2),1)
        return self.score_head(x),self.reg_head(x)


class VoxelNet(nn.Module):

    def __init__(self):
        super(VoxelNet, self).__init__()
        self.svfe = SVFE()
        self.cml = CML()
        self.rpn = RPN()

    def voxel_indexing(self, sparse_features, coords):
        dim = sparse_features.shape[-1]

        dense_feature = Variable(torch.zeros(dim, config.N, config.D, config.H, config.W).cuda())

        dense_feature[:, coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = sparse_features

        return dense_feature.transpose(0, 1)

    def forward(self, voxel_features, voxel_coords):

        # feature learning network
        vwfs = self.svfe(voxel_features)
        vwfs = self.voxel_indexing(vwfs, voxel_coords)

        # convolutional middle network
        cml_out = self.cml(vwfs)

        # region proposal network

        # merge the depth and feature dim into one, output probability score map and regression map
        psm, rm = self.rpn(cml_out.view(config.N, -1, config.H, config.W))

        return psm, rm


# loss function

class VoxelLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(VoxelLoss, self).__init__()
        self.smoothl1loss = nn.SmoothL1Loss(size_average=False)
        self.alpha = alpha
        self.beta = beta

    def forward(self, rm, psm, pos_equal_one, neg_equal_one, targets):

        p_pos = F.sigmoid(psm.permute(0,2,3,1))
        rm = rm.permute(0,2,3,1).contiguous()
        rm = rm.view(rm.size(0),rm.size(1),rm.size(2),-1,7)
        targets = targets.view(targets.size(0),targets.size(1),targets.size(2),-1,7)
        pos_equal_one_for_reg = pos_equal_one.unsqueeze(pos_equal_one.dim()).expand(-1,-1,-1,-1,7)

        rm_pos = rm * pos_equal_one_for_reg
        targets_pos = targets * pos_equal_one_for_reg

        cls_pos_loss = -pos_equal_one * torch.log(p_pos + 1e-6)
        cls_pos_loss = cls_pos_loss.sum() / (pos_equal_one.sum() + 1e-6)

        cls_neg_loss = -neg_equal_one * torch.log(1 - p_pos + 1e-6)
        cls_neg_loss = cls_neg_loss.sum() / (neg_equal_one.sum() + 1e-6)

        reg_loss = self.smoothl1loss(rm_pos, targets_pos)
        reg_loss = reg_loss / (pos_equal_one.sum() + 1e-6)
        conf_loss = self.alpha * cls_pos_loss + self.beta * cls_neg_loss
        return conf_loss, reg_loss






# _____________________________________________________________________________________________________ #



















# _____________________________________________________________________________________________________ #








# data augmentation



def draw_polygon(img, box_corner, color = (255, 255, 255),thickness = 1):

    tup0 = (box_corner[0, 1],box_corner[0, 0])
    tup1 = (box_corner[1, 1],box_corner[1, 0])
    tup2 = (box_corner[2, 1],box_corner[2, 0])
    tup3 = (box_corner[3, 1],box_corner[3, 0])
    cv2.line(img, tup0, tup1, color, thickness, cv2.LINE_AA)
    cv2.line(img, tup1, tup2, color, thickness, cv2.LINE_AA)
    cv2.line(img, tup2, tup3, color, thickness, cv2.LINE_AA)
    cv2.line(img, tup3, tup0, color, thickness, cv2.LINE_AA)
    return img


def point_transform(points, tx, ty, tz, rx=0, ry=0, rz=0):
    # Input:
    #   points: (N, 3)
    #   rx/y/z: in radians
    # Output:
    #   points: (N, 3)
    N = points.shape[0]
    points = np.hstack([points, np.ones((N, 1))])
    mat1 = np.eye(4)
    mat1[3, 0:3] = tx, ty, tz
    points = np.matmul(points, mat1)
    if rx != 0:
        mat = np.zeros((4, 4))
        mat[0, 0] = 1
        mat[3, 3] = 1
        mat[1, 1] = np.cos(rx)
        mat[1, 2] = -np.sin(rx)
        mat[2, 1] = np.sin(rx)
        mat[2, 2] = np.cos(rx)
        points = np.matmul(points, mat)
    if ry != 0:
        mat = np.zeros((4, 4))
        mat[1, 1] = 1
        mat[3, 3] = 1
        mat[0, 0] = np.cos(ry)
        mat[0, 2] = np.sin(ry)
        mat[2, 0] = -np.sin(ry)
        mat[2, 2] = np.cos(ry)
        points = np.matmul(points, mat)
    if rz != 0:
        mat = np.zeros((4, 4))
        mat[2, 2] = 1
        mat[3, 3] = 1
        mat[0, 0] = np.cos(rz)
        mat[0, 1] = -np.sin(rz)
        mat[1, 0] = np.sin(rz)
        mat[1, 1] = np.cos(rz)
        points = np.matmul(points, mat)
    return points[:, 0:3]


def box_transform(boxes_corner, tx, ty, tz, r=0):
    # boxes_corner (N, 8, 3)
    for idx in range(len(boxes_corner)):
        boxes_corner[idx] = point_transform(boxes_corner[idx], tx, ty, tz, rz=r)
    return boxes_corner


def cal_iou2d(box1_corner, box2_corner):
    box1_corner = np.reshape(box1_corner, [4, 2])
    box2_corner = np.reshape(box2_corner, [4, 2])
    box1_corner = ((config.W, config.H)-(box1_corner - (config.xrange[0], config.yrange[0])) / (config.vw, config.vh)).astype(np.int32)
    box2_corner = ((config.W, config.H)-(box2_corner - (config.xrange[0], config.yrange[0])) / (config.vw, config.vh)).astype(np.int32)

    buf1 = np.zeros((config.H, config.W, 3))
    buf2 = np.zeros((config.H, config.W, 3))
    buf1 = cv2.fillConvexPoly(buf1, box1_corner, color=(1,1,1))[..., 0]
    buf2 = cv2.fillConvexPoly(buf2, box2_corner, color=(1,1,1))[..., 0]

    indiv = np.sum(np.absolute(buf1-buf2))
    share = np.sum((buf1 + buf2) == 2)
    if indiv == 0:
        return 0.0 # when target is out of bound
    return share / (indiv + share)


def aug_data(lidar, gt_box3d_corner):
    np.random.seed()

    choice = np.random.randint(1, 10)

    if choice >= 7:
        for idx in range(len(gt_box3d_corner)):
            # TODO: precisely gather the point
            is_collision = True
            _count = 0
            while is_collision and _count < 100:
                t_rz = np.random.uniform(-np.pi / 10, np.pi / 10)
                t_x = np.random.normal()
                t_y = np.random.normal()
                t_z = np.random.normal()

                # check collision
                tmp = box_transform(
                    gt_box3d_corner[[idx]], t_x, t_y, t_z, t_rz)
                is_collision = False
                for idy in range(idx):
                    iou = cal_iou2d(tmp[0, :4, :2], gt_box3d_corner[idy, :4, :2])
                    if iou > 0:
                        is_collision = True
                        _count += 1
                        break
            if not is_collision:
                box_corner = gt_box3d_corner[idx]
                minx = np.min(box_corner[:, 0])
                miny = np.min(box_corner[:, 1])
                minz = np.min(box_corner[:, 2])
                maxx = np.max(box_corner[:, 0])
                maxy = np.max(box_corner[:, 1])
                maxz = np.max(box_corner[:, 2])
                bound_x = np.logical_and(
                    lidar[:, 0] >= minx, lidar[:, 0] <= maxx)
                bound_y = np.logical_and(
                    lidar[:, 1] >= miny, lidar[:, 1] <= maxy)
                bound_z = np.logical_and(
                    lidar[:, 2] >= minz, lidar[:, 2] <= maxz)
                bound_box = np.logical_and(
                    np.logical_and(bound_x, bound_y), bound_z)
                lidar[bound_box, 0:3] = point_transform(
                    lidar[bound_box, 0:3], t_x, t_y, t_z, rz=t_rz)
                gt_box3d_corner[idx] = box_transform(
                    gt_box3d_corner[[idx]], t_x, t_y, t_z, t_rz)

        gt_box3d = gt_box3d_corner

    elif choice < 7 and choice >= 4:
        # global rotation
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        lidar[:, 0:3] = point_transform(lidar[:, 0:3], 0, 0, 0, rz=angle)
        gt_box3d = box_transform(gt_box3d_corner, 0, 0, 0, r=angle)

    else:
        # global scaling
        factor = np.random.uniform(0.95, 1.05)
        lidar[:, 0:3] = lidar[:, 0:3] * factor
        gt_box3d = gt_box3d_corner * factor

    return lidar, gt_box3d


# Utilities







def get_filtered_lidar(lidar, boxes3d=None):

    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]

    filter_x = np.where((pxs >= config.xrange[0]) & (pxs < config.xrange[1]))[0]
    filter_y = np.where((pys >= config.yrange[0]) & (pys < config.yrange[1]))[0]
    filter_z = np.where((pzs >= config.zrange[0]) & (pzs < config.zrange[1]))[0]
    filter_xy = np.intersect1d(filter_x, filter_y)
    filter_xyz = np.intersect1d(filter_xy, filter_z)

    if boxes3d is not None:
        box_x = (boxes3d[:, :, 0] >= config.xrange[0]) & (boxes3d[:, :, 0] < config.xrange[1])
        box_y = (boxes3d[:, :, 1] >= config.yrange[0]) & (boxes3d[:, :, 1] < config.yrange[1])
        box_z = (boxes3d[:, :, 2] >= config.zrange[0]) & (boxes3d[:, :, 2] < config.zrange[1])
        box_xyz = np.sum(box_x & box_y & box_z,axis=1)

        return lidar[filter_xyz], boxes3d[box_xyz>0]

    return lidar[filter_xyz]

def lidar_to_bev(lidar):

    X0, Xn = 0, config.W
    Y0, Yn = 0, config.H
    Z0, Zn = 0, config.D

    width  = Yn - Y0
    height   = Xn - X0
    channel = Zn - Z0  + 2

    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]
    prs = lidar[:, 3]

    qxs=((pxs-config.xrange[0])/config.vw).astype(np.int32)
    qys=((pys-config.yrange[0])/config.vh).astype(np.int32)
    qzs=((pzs-config.zrange[0])/config.vd).astype(np.int32)

    print('height,width,channel=%d,%d,%d' % (height, width, channel))
    top = np.zeros(shape=(height, width, channel), dtype=np.float32)
    mask = np.ones(shape=(height, width, channel-1), dtype=np.float32) * -5

    for i in range(len(pxs)):
        top[-qxs[i], -qys[i], -1] = 1 + top[-qxs[i], -qys[i], -1]
        if pzs[i] > mask[-qxs[i], -qys[i], qzs[i]]:
            top[-qxs[i], -qys[i], qzs[i]] = max(0, pzs[i]-config.zrange[0])
            mask[-qxs[i], -qys[i], qzs[i]]=pzs[i]
        if pzs[i] > mask[-qxs[i], -qys[i], -1]:
            mask[-qxs[i], -qys[i], -1] = pzs[i]
            top[-qxs[i], -qys[i], -2] = prs[i]

    top[:, :, -1] = np.log(top[:, :, -1]+1)/math.log(64)

    if 1:
        # top_image = np.sum(top[:,:,:-1],axis=2)
        density_image = top[:, :, -1]
        density_image = density_image-np.min(density_image)
        density_image = (density_image/np.max(density_image)*255).astype(np.uint8)
        # top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)

    return top, density_image


def draw_lidar(lidar, is_grid=False, is_axis=True, is_top_region=True, fig=None):

    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]
    prs = lidar[:, 3]

    if fig is None: fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))

    mlab.points3d(
        pxs, pys, pzs, prs,
        mode='point',  # 'point'  'sphere'
        colormap='gnuplot',  # 'bone',  #'spectral',  #'copper',
        scale_factor=1,
        figure=fig)

    # draw grid
    if is_grid:
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        for y in np.arange(-50,50,1):
            x1,y1,z1 = -50, y, 0
            x2,y2,z2 =  50, y, 0
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

        for x in np.arange(-50,50,1):
            x1,y1,z1 = x,-50, 0
            x2,y2,z2 = x, 50, 0
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

    # draw axis
    if is_grid:
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        axes=np.array([
            [2.,0.,0.,0.],
            [0.,2.,0.,0.],
            [0.,0.,2.,0.],
        ],dtype=np.float64)
        fov=np.array([  ##<todo> : now is 45 deg. use actual setting later ...
            [20., 20., 0.,0.],
            [20.,-20., 0.,0.],
        ],dtype=np.float64)


        mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
        mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)

    # draw top_image feature area
    if is_top_region:
        x1 = config.xrange[0]
        x2 = config.xrange[1]
        y1 = config.yrange[0]
        y2 = config.yrange[1]
        mlab.plot3d([x1, x1], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x2, x2], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y1, y1], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y2, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

    mlab.orientation_axes()
    mlab.view(azimuth=180,elevation=None,distance=50,focalpoint=[ 12.0909996 , -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991

    return fig


def draw_gt_boxes3d(gt_boxes3d, fig, color=(1,0,0), line_width=2):

    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]

        for k in range(0,4):

            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+3)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

    mlab.view(azimuth=180,elevation=None,distance=50,focalpoint=[ 12.0909996 , -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991


def project_velo2rgb(velo,calib):
    T=np.zeros([4,4],dtype=np.float32)
    T[:3,:]=calib['Tr_velo2cam']
    T[3,3]=1
    R=np.zeros([4,4],dtype=np.float32)
    R[:3,:3]=calib['R0']
    R[3,3]=1
    num=len(velo)
    projections = np.zeros((num,8,2),  dtype=np.int32)
    for i in range(len(velo)):
        box3d=np.ones([8,4],dtype=np.float32)
        box3d[:,:3]=velo[i]
        M=np.dot(calib['P2'],R)
        M=np.dot(M,T)
        box2d=np.dot(M,box3d.T)
        box2d=box2d[:2,:].T/box2d[2,:].reshape(8,1)
        projections[i] = box2d
    return projections


def draw_rgb_projections(image, projections, color=(255,255,255), thickness=2, darker=1):

    img = image.copy()*darker
    num=len(projections)
    forward_color=(255,255,0)
    for n in range(num):
        qs = projections[n]
        for k in range(0,4):
            i,j=k,(k+1)%4

            cv2.line(img, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

            i,j=k+4,(k+1)%4 + 4
            cv2.line(img, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

            i,j=k,k+4
            cv2.line(img, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

        cv2.line(img, (qs[3,0],qs[3,1]), (qs[7,0],qs[7,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[7,0],qs[7,1]), (qs[6,0],qs[6,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[6,0],qs[6,1]), (qs[2,0],qs[2,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[2,0],qs[2,1]), (qs[3,0],qs[3,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[3,0],qs[3,1]), (qs[6,0],qs[6,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[2,0],qs[2,1]), (qs[7,0],qs[7,1]), forward_color, thickness, cv2.LINE_AA)

    return img


def _quantize_coords(x, y):
    xx = config.H - int((y - config.yrange[0]) / config.vh)
    yy = config.W - int((x - config.xrange[0]) / config.vw)
    return xx, yy


def  draw_polygons(image, polygons,color=(255,255,255), thickness=1, darken=1):

    img = image.copy() * darken
    for polygon in polygons:
        tup0, tup1, tup2, tup3 = [_quantize_coords(*tup) for tup in polygon]
        cv2.line(img, tup0, tup1, color, thickness, cv2.LINE_AA)
        cv2.line(img, tup1, tup2, color, thickness, cv2.LINE_AA)
        cv2.line(img, tup2, tup3, color, thickness, cv2.LINE_AA)
        cv2.line(img, tup3, tup0, color, thickness, cv2.LINE_AA)
    return img


def draw_rects(image, rects, color=(255,255,255), thickness=1, darken=1):

    img = image.copy() * darken
    for rect in rects:
        tup0,tup1 = [_quantize_coords(*tup) for tup in list(zip(rect[0::2], rect[1::2]))]
        cv2.rectangle(img, tup0, tup1, color, thickness, cv2.LINE_AA)
    return img


def load_kitti_calib(calib_file):
    """
    load projection matrix
    """
    with open(calib_file) as fi:
        lines = fi.readlines()
        assert (len(lines) == 8)

    obj = lines[0].strip().split(' ')[1:]
    P0 = np.array(obj, dtype=np.float32)
    obj = lines[1].strip().split(' ')[1:]
    P1 = np.array(obj, dtype=np.float32)
    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)
    obj = lines[6].strip().split(' ')[1:]
    Tr_imu_to_velo = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


def angle_in_limit(angle):
    # To limit the angle in -pi/2 - pi/2
    limit_degree = 5
    while angle >= np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
    if abs(angle + np.pi / 2) < limit_degree / 180 * np.pi:
        angle = np.pi / 2
    return angle


def box3d_cam_to_velo(box3d, Tr):

    def project_cam2velo(cam, Tr):
        T = np.zeros([4, 4], dtype=np.float32)
        T[:3, :] = Tr
        T[3, 3] = 1
        T_inv = np.linalg.inv(T)
        lidar_loc_ = np.dot(T_inv, cam)
        lidar_loc = lidar_loc_[:3]
        return lidar_loc.reshape(1, 3)

    def ry_to_rz(ry):
        angle = -ry - np.pi / 2

        if angle >= np.pi:
            angle -= np.pi
        if angle < -np.pi:
            angle = 2*np.pi + angle

        return angle

    h,w,l,tx,ty,tz,ry = [float(i) for i in box3d]
    cam = np.ones([4, 1])
    cam[0] = tx
    cam[1] = ty
    cam[2] = tz
    t_lidar = project_cam2velo(cam, Tr)

    Box = np.array([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                    [0, 0, 0, 0, h, h, h, h]])

    rz = ry_to_rz(ry)

    rotMat = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz), np.cos(rz), 0.0],
        [0.0, 0.0, 1.0]])

    velo_box = np.dot(rotMat, Box)

    cornerPosInVelo = velo_box + np.tile(t_lidar, (8, 1)).T

    box3d_corner = cornerPosInVelo.transpose()

    return box3d_corner.astype(np.float32)


def anchors_center_to_corner(anchors):
    N = anchors.shape[0]
    anchor_corner = np.zeros((N, 4, 2))
    for i in range(N):
        anchor = anchors[i]
        translation = anchor[0:3]
        h, w, l = anchor[3:6]
        rz = anchor[-1]
        Box = np.array([
            [-l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2]])
        # re-create 3D bounding box in velodyne coordinate system
        rotMat = np.array([
            [np.cos(rz), -np.sin(rz)],
            [np.sin(rz), np.cos(rz)]])
        velo_box = np.dot(rotMat, Box)
        cornerPosInVelo = velo_box + np.tile(translation[:2], (4, 1)).T
        box2d = cornerPosInVelo.transpose()
        anchor_corner[i] = box2d
    return anchor_corner


def corner_to_standup_box2d_batch(boxes_corner):
    # (N, 4, 2) -> (N, 4) x1, y1, x2, y2
    N = boxes_corner.shape[0]
    standup_boxes2d = np.zeros((N, 4))
    standup_boxes2d[:, 0] = np.min(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 1] = np.min(boxes_corner[:, :, 1], axis=1)
    standup_boxes2d[:, 2] = np.max(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 3] = np.max(boxes_corner[:, :, 1], axis=1)
    return standup_boxes2d


def box3d_corner_to_center_batch(box3d_corner):
    # (N, 8, 3) -> (N, 7)
    assert box3d_corner.ndim == 3
    batch_size = box3d_corner.shape[0]

    xyz = np.mean(box3d_corner[:, :4, :], axis=1)

    h = abs(np.mean(box3d_corner[:, 4:, 2] - box3d_corner[:, :4, 2], axis=1, keepdims=True))

    w = (np.sqrt(np.sum((box3d_corner[:, 0, [0, 1]] - box3d_corner[:, 1, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 2, [0, 1]] - box3d_corner[:, 3, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 4, [0, 1]] - box3d_corner[:, 5, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 6, [0, 1]] - box3d_corner[:, 7, [0, 1]]) ** 2, axis=1, keepdims=True))) / 4

    l = (np.sqrt(np.sum((box3d_corner[:, 0, [0, 1]] - box3d_corner[:, 3, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 1, [0, 1]] - box3d_corner[:, 2, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 4, [0, 1]] - box3d_corner[:, 7, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 5, [0, 1]] - box3d_corner[:, 6, [0, 1]]) ** 2, axis=1, keepdims=True))) / 4

    theta = (np.arctan2(box3d_corner[:, 2, 1] - box3d_corner[:, 1, 1],
                        box3d_corner[:, 2, 0] - box3d_corner[:, 1, 0]) +
             np.arctan2(box3d_corner[:, 3, 1] - box3d_corner[:, 0, 1],
                        box3d_corner[:, 3, 0] - box3d_corner[:, 0, 0]) +
             np.arctan2(box3d_corner[:, 2, 0] - box3d_corner[:, 3, 0],
                        box3d_corner[:, 3, 1] - box3d_corner[:, 2, 1]) +
             np.arctan2(box3d_corner[:, 1, 0] - box3d_corner[:, 0, 0],
                        box3d_corner[:, 0, 1] - box3d_corner[:, 1, 1]))[:, np.newaxis] / 4

    return np.concatenate([xyz, h, w, l, theta], axis=1).reshape(batch_size, 7)


def get_anchor3d(anchors):
    num = anchors.shape[0]
    anchors3d = np.zeros((num,8,3))
    anchors3d[:, :4, :2] = anchors
    anchors3d[:, :, 2] = config.z_a
    anchors3d[:, 4:, :2] = anchors
    anchors3d[:, 4:, 2] = config.z_a + config.h_a
    return anchors3d


def load_kitti_label(label_file, Tr):

    with open(label_file, 'r') as f:
        lines = f.readlines()

    gt_boxes3d_corner = []

    num_obj = len(lines)

    for j in range(num_obj):
        obj = lines[j].strip().split(' ')

        obj_class = obj[0].strip()
        if obj_class not in config.class_list:
            continue

        box3d_corner = box3d_cam_to_velo(obj[8:], Tr)

        gt_boxes3d_corner.append(box3d_corner)

    gt_boxes3d_corner = np.array(gt_boxes3d_corner).reshape(-1, 8, 3)

    return gt_boxes3d_corner



















def test():
    import os
    import glob
    import matplotlib.pyplot as plt

    lidar_path = os.path.join('./data/KITTI/training', "crop/")
    image_path = os.path.join('./data/KITTI/training', "image_2/")
    calib_path = os.path.join('./data/KITTI/training', "calib/")
    label_path = os.path.join('./data/KITTI/training', "label_2/")

    file=[i.strip().split('/')[-1][:-4] for i in sorted(os.listdir(label_path))]

    i = 2600

    lidar_file = lidar_path + '/' + file[i] + '.bin'
    calib_file = calib_path + '/' + file[i] + '.txt'
    label_file = label_path + '/' + file[i] + '.txt'
    image_file = image_path + '/' + file[i] + '.png'

    image = cv2.imread(image_file)
    print("Processing: ", lidar_file)
    lidar = np.fromfile(lidar_file, dtype=np.float32)
    lidar = lidar.reshape((-1, 4))

    calib = load_kitti_calib(calib_file)
    gt_box3d = load_kitti_label(label_file, calib['Tr_velo2cam'])

    # augmentation
    #lidar, gt_box3d = aug_data(lidar, gt_box3d)

    # filtering
    lidar, gt_box3d = get_filtered_lidar(lidar, gt_box3d)

    # view in point cloud

    # fig = draw_lidar(lidar, is_grid=False, is_top_region=True)
    # draw_gt_boxes3d(gt_boxes3d=gt_box3d, fig=fig)
    # mlab.show()

    # view in image

    # gt_3dTo2D = project_velo2rgb(gt_box3d, calib)
    # img_with_box = draw_rgb_projections(image,gt_3dTo2D, color=(0,0,255),thickness=1)
    # plt.imshow(img_with_box[:,:,[2,1,0]])
    # plt.show()

    # view in bird-eye view

    top_new, density_image=lidar_to_bev(lidar)
    # gt_box3d_top = corner_to_standup_box2d_batch(gt_box3d)
    # density_with_box = draw_rects(density_image,gt_box3d_top)
    density_with_box = draw_polygons(density_image,gt_box3d[:,:4,:2])
    plt.imshow(density_with_box,cmap='gray')
    plt.show()


# Train Network

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight.data)
        m.bias.data.zero_()


def detection_collate(batch):
    voxel_features = []
    voxel_coords = []
    pos_equal_one = []
    neg_equal_one = []
    targets = []

    images = []
    calibs = []
    ids = []
    for i, sample in enumerate(batch):
        voxel_features.append(sample[0])

        voxel_coords.append(
            np.pad(sample[1], ((0, 0), (1, 0)),
                mode='constant', constant_values=i))

        pos_equal_one.append(sample[2])
        neg_equal_one.append(sample[3])
        targets.append(sample[4])

        images.append(sample[5])
        calibs.append(sample[6])
        ids.append(sample[7])
    return np.concatenate(voxel_features), \
           np.concatenate(voxel_coords), \
           np.array(pos_equal_one),\
           np.array(neg_equal_one),\
           np.array(targets),\
           images, calibs, ids


















# Test utilities



def delta_to_boxes3d(deltas, anchors):
    # Input:
    #   deltas: (N, w, l, 14)
    #   feature_map_shape: (w, l)
    #   anchors: (w, l, 2, 7)

    # Ouput:
    #   boxes3d: (N, w*l*2, 7)
    N = deltas.shape[0]
    deltas = deltas.view(N, -1, 7)
    anchors = torch.FloatTensor(anchors)
    boxes3d = torch.zeros_like(deltas)

    if deltas.is_cuda:
        anchors = anchors.cuda()
        boxes3d = boxes3d.cuda()

    anchors_reshaped = anchors.view(-1, 7)

    anchors_d = torch.sqrt(anchors_reshaped[:, 4]**2 + anchors_reshaped[:, 5]**2)

    anchors_d = anchors_d.repeat(N, 2, 1).transpose(1,2)
    anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

    boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + anchors_reshaped[..., [0, 1]]
    boxes3d[..., [2]] = torch.mul(deltas[..., [2]], anchors_reshaped[...,[3]]) + anchors_reshaped[..., [2]]

    boxes3d[..., [3, 4, 5]] = torch.exp(
        deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]

    boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

    return boxes3d


def detection_collate(batch):
    lidars = []
    images = []
    calibs = []

    targets = []
    pos_equal_ones=[]
    ids = []
    for i, sample in enumerate(batch):
        lidars.append(sample[0])
        images.append(sample[1])
        calibs.append(sample[2])
        targets.append(sample[3])
        pos_equal_ones.append(sample[4])
        ids.append(sample[5])
    return lidars,images,calibs,\
           torch.cuda.FloatTensor(np.array(targets)), \
           torch.cuda.FloatTensor(np.array(pos_equal_ones)),\
           ids


def box3d_center_to_corner_batch(boxes_center):
    # (N, 7) -> (N, 8, 3)
    N = boxes_center.shape[0]
    ret = torch.zeros((N, 8, 3))
    if boxes_center.is_cuda:
        ret = ret.cuda()

    for i in range(N):
        box = boxes_center[i]
        translation = box[0:3]
        size = box[3:6]
        rotation = [0, 0, box[-1]]

        h, w, l = size[0], size[1], size[2]
        trackletBox = torch.FloatTensor([  # in velodyne coordinates around zero point and without orientation yet
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [0, 0, 0, 0, h, h, h, h]])
        if boxes_center.is_cuda:
            trackletBox = trackletBox.cuda()
        # re-create 3D bounding box in velodyne coordinate system
        yaw = rotation[2]
        rotMat = torch.FloatTensor([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]])
        if boxes_center.is_cuda:
            rotMat = rotMat.cuda()

        cornerPosInVelo = torch.mm(rotMat, trackletBox) + translation.repeat(8, 1).t()
        box3d = cornerPosInVelo.transpose(0,1)
        ret[i] = box3d

    return ret


def box3d_corner_to_top_batch(boxes3d, use_min_rect=True):
    # [N,8,3] -> [N,4,2] -> [N,8]
    box3d_top=[]

    num = len(boxes3d)
    for n in range(num):
        b  = boxes3d[n]
        x0 = b[0,0]
        y0 = b[0,1]
        x1 = b[1,0]
        y1 = b[1,1]
        x2 = b[2,0]
        y2 = b[2,1]
        x3 = b[3,0]
        y3 = b[3,1]
        box3d_top.append([x0,y0,x1,y1,x2,y2,x3,y3])

    if use_min_rect:
        box8pts = torch.FloatTensor(np.array(box3d_top))
        if boxes3d.is_cuda:
            box8pts = box8pts.cuda()
        min_rects = torch.zeros((box8pts.shape[0], 4))
        if boxes3d.is_cuda:
            min_rects = min_rects.cuda()
        # calculate minimum rectangle
        min_rects[:, 0] = torch.min(box8pts[:, [0, 2, 4, 6]], dim=1)[0]
        min_rects[:, 1] = torch.min(box8pts[:, [1, 3, 5, 7]], dim=1)[0]
        min_rects[:, 2] = torch.max(box8pts[:, [0, 2, 4, 6]], dim=1)[0]
        min_rects[:, 3] = torch.max(box8pts[:, [1, 3, 5, 7]], dim=1)[0]
        return min_rects

    return box3d_top


def draw_boxes(reg, prob, images, calibs, ids, tag):
    prob = prob.view(config.N, -1)
    batch_boxes3d = delta_to_boxes3d(reg, config.anchors)
    mask = torch.gt(prob, config.score_threshold)
    mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

    for batch_id in range(config.N):
        boxes3d = torch.masked_select(batch_boxes3d[batch_id], mask_reg[batch_id]).view(-1, 7)
        scores = torch.masked_select(prob[batch_id], mask[batch_id])

        image = images[batch_id]
        calib = calibs[batch_id]
        id = ids[batch_id]

        if len(boxes3d) != 0:

            boxes3d_corner = box3d_center_to_corner_batch(boxes3d)
            boxes2d = box3d_corner_to_top_batch(boxes3d_corner)
            boxes2d_score = torch.cat((boxes2d, scores.unsqueeze(1)), dim=1)

            # NMS
            keep = pth_nms(boxes2d_score, config.nms_threshold)
            boxes3d_corner_keep = boxes3d_corner[keep]
            print("No. %d objects detected" % len(boxes3d_corner_keep))

            rgb_2D = project_velo2rgb(boxes3d_corner_keep, calib)
            img_with_box = draw_rgb_projections(image, rgb_2D, color=(0, 0, 255), thickness=1)
            cv2.imwrite('results/%s_%s.png' % (id,tag), img_with_box)

        else:
            cv2.imwrite('results/%s_%s.png' % (id,tag), image)
            print("No objects detected")














def voxelnet_unit():

    torch.backends.cudnn.enabled = True

    # dataset
    dataset = A2D2Dataset(cfg=config, root='./data/KITTI', set='train')
    data_loader = data.DataLoader(dataset, batch_size=config.N, num_workers=4, collate_fn=detection_collate, shuffle=True, \
                                  pin_memory=False)

    # network
    net = VoxelNet()
    net.cuda()

    net.train()

    # initialization
    print('Initializing weights...')
    net.apply(weights_init)

    # define optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # define loss function
    criterion = VoxelLoss(alpha=1.5, beta=1)

    # training process
    batch_iterator = None
    epoch_size = len(dataset) // config.N
    print('Epoch size', epoch_size)
    for iteration in range(10000):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)

        voxel_features, voxel_coords, pos_equal_one, neg_equal_one, targets, images, calibs, ids = next(batch_iterator)

        # wrapper to variable
        voxel_features = Variable(torch.cuda.FloatTensor(voxel_features))
        pos_equal_one = Variable(torch.cuda.FloatTensor(pos_equal_one))
        neg_equal_one = Variable(torch.cuda.FloatTensor(neg_equal_one))
        targets = Variable(torch.cuda.FloatTensor(targets))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        t0 = time.time()
        psm, rm = net(voxel_features, voxel_coords)

        # calculate loss
        conf_loss, reg_loss = criterion(rm, psm, pos_equal_one, neg_equal_one, targets)
        loss = conf_loss + reg_loss

        # backward
        loss.backward()
        optimizer.step()

        t1 = time.time()

        print('Timer: %.4f sec.' % (t1 - t0))
        print('iter ' + repr(iteration) + ' || Loss: %.4f || Conf Loss: %.4f || Loc Loss: %.4f' % \
              (loss.data[0], conf_loss.data[0], reg_loss.data[0]))

        # visualization
        # draw_boxes(rm, psm, ids, images, calibs, 'pred')
        draw_boxes(targets.data, pos_equal_one.data, images, calibs, ids, 'true')















