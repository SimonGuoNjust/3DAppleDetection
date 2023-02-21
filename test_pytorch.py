#!/usr/bin/python3.6

import argparse
import copy
import os
import queue
import time
import sys
from sklearn import linear_model, datasets

from scipy import optimize as sci_optimize
from pathlib import Path
import pyrealsense2 as rs2
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
sys.path.insert(0,'apple3ddetection/thirdparty/yolov5')
from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import matplotlib as mpl
# import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.datasets import letterbox
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import threading
from mmdet.apis import init_detector, inference_detector
from apple3ddetection.RGBDetetor.yolo import YoloDetector
from configs.yolo import yolo_config
import mmcv
mpl.use('TkAgg')



class pt_Vis():
    def __init__(self,name='20m test',width=800,height=600,json='./viewpoint.json'):
        self.vis=o3d.visualization.Visualizer()
        self.vis.create_window(window_name=name,width=width,height=height)
        self.axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
        self.vis.set_full_screen()
        # 可视化参数
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([125, 125, 125])
        opt.point_size = 1
        opt.show_coordinate_frame = True

        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)

        # 读取viewpoint参数
        # self.param=o3d.io.read_pinhole_camera_parameters(json)
        # self.ctr=self.vis.get_view_control()
        # self.ctr.convert_from_pinhole_camera_parameters(self.param)
        # print('viewpoint json loaded!')

    def __del__(self):
        self.vis.destroy_window()

    def update(self,pcd):
        '''

        :param pcd: PointCLoud()
        :return:
        '''

        # param = self.ctr.convert_to_pinhole_camera_parameters()

        # o3d.io.write_pinhole_camera_parameters('./new.json',param)
        # print('viewpoint json saved!')

        self.pcd.points = pcd.points
        self.pcd = pcd

        self.pcd.colors = pcd.colors

        self.vis.clear_geometries()
        self.vis.update_geometry(self.pcd)          # 更新点云

        # self.vis.remove_geometry(self.pcd)          # 删除vis中的点云
        self.vis.add_geometry(self.pcd)             # 增加vis中的点云

        # 设置viewpoint
        # self.ctr.convert_from_pinhole_camera_parameters(self.param)

        self.vis.poll_events()
        self.vis.update_renderer()
        # self.vis.run()

    def capture_screen(self,fn, depth = False):
        if depth:
            self.vis.capture_depth_image(fn, False)
        else:
            self.vis.capture_screen_image(fn, False)


def project_image_to_rect(uv_depth, P):
    '''
    :param box: nx3 first two channels are uv in image coord, 3rd channel
                is depth in rect camera coord
    :param P: 3x3 or 3x4
    :return: nx3 points in rect camera coord
    '''
    c_u = P[0,2]
    c_v = P[1,2]
    f_u = P[0, 0]
    f_v = P[1, 1]
    if P.shape[1] == 4:
        b_x = P[0, 3] / (-f_u)  # relative
        b_y = P[1, 3] / (-f_v)
    else:
        b_x = 0
        b_y = 0
    n = uv_depth.shape[0]
    x = ((uv_depth[:, 0] - c_u) * uv_depth[:, 2]) / f_u + b_x
    y = ((uv_depth[:, 1] - c_v) * uv_depth[:, 2]) / f_v + b_y
    pts_3d_rect = np.zeros((n, 3), dtype=uv_depth.dtype)
    pts_3d_rect[:, 0] = x
    pts_3d_rect[:, 1] = y
    pts_3d_rect[:, 2] = uv_depth[:, 2]
    return pts_3d_rect


def get_max_min_range(box, depth):
    xmin, ymin, xmax, ymax = box
    depth_crop = depth[int(ymin):int(ymax),int(xmin):int(xmax)]
    try:
        max_depth = np.max(depth_crop[np.nonzero(depth_crop)])
        min_depth = np.min(depth_crop[np.nonzero(depth_crop)])
    except Exception as e:
        return None, None

    return max_depth, min_depth


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc, box3d)
    return box3d_roi_inds


def get_frustum_lines(frustums):
    lines = list()
    for i in range(len(frustums)):
        offset = i * 8
        lines += [[0 + offset,2 + offset],[2 + offset,1 + offset],[1 + offset,3 + offset],[3 + offset,0 + offset],
                  [4 + offset,6 + offset],[6 + offset,5 + offset],[5 + offset,7 + offset],[7 + offset,4 + offset],
                  [0 + offset,4 + offset],[1 + offset,5 + offset],[2 + offset,6 + offset],[3 + offset,7 + offset]]
    lines_pcd_lines = o3d.utility.Vector2iVector(lines)
    color_ = [[1, 0, 0] for i in range(len(lines))]
    lines_pcd_colors = o3d.utility.Vector3dVector(color_)  # 线条颜色
    points = list()
    for frustum in frustums:
        points += frustum
    lines_pcd_points = o3d.utility.Vector3dVector(points)
    return lines_pcd_lines, lines_pcd_points, lines_pcd_colors


def sphere_surface(points, no_continue=False, type='custom'):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # 计算拟合矩阵A，数组b的参数
    xm = np.mean(x)
    ym = np.mean(y)
    zm = np.mean(z)
    if type == 'custom':
        """球面拟合"""

        xym = np.mean(x * y)
        xzm = np.mean(x * z)
        yzm = np.mean(y * z)
        x2m = np.mean(x * x)
        y2m = np.mean(y * y)
        z2m = np.mean(z * z)
        x2ym = np.mean(x * x * y)
        x2zm = np.mean(x * x * z)
        y2xm = np.mean(y * y * x)
        y2zm = np.mean(y * y * z)
        z2xm = np.mean(z * z * x)
        z2ym = np.mean(z * z * y)
        x3m = np.mean(x * x * x)
        y3m = np.mean(y * y * y)
        z3m = np.mean(z * z * z)

        A = np.array([[x2m - xm * xm, xym - xm * ym, xzm - xm * zm], [xym - xm * ym, y2m - ym * ym, yzm - ym * zm],
                      [xzm - xm * zm, yzm - ym * zm, z2m - zm * zm]])
        b = 0.5 * np.array(
            [x3m - xm * x2m + y2xm - xm * y2m + z2xm - xm * z2m, x2ym - x2m * ym + y3m - ym * y2m + z2ym - ym * z2m,
             x2zm - x2m * zm + y2zm - y2m * zm + z3m - z2m * zm])
        # 求解球心
        s = np.linalg.solve(A, b)
        # 求解半径
        R = np.sqrt(x2m - 2 * s[0] * xm + s[0] ** 2 + y2m - 2 * s[1] * ym + s[1] ** 2 + z2m - 2 * s[2] * zm + s[2] ** 2)

        # 计算误差
        dis = np.sqrt(np.sum(np.power(points-s,2),axis=1))
        err = np.abs(dis-R)
        mean_error = np.mean(err)
        std_error = np.std(err)
        if no_continue:
            return s,R,mean_error
        non_outliers = np.where(np.abs((err-mean_error)/ std_error) < 5)[0].tolist()
        # print(outliers)
        if len(points)-len(non_outliers) > 10:
            # print(points.shape)
            new_points = points[non_outliers,:]
            # print(np.where(err>0.9*(err_range[1]-err_range[0])))
            # print(new_points.shape)
            # input()
            s, R,_ = sphere_surface(new_points,no_continue=False)
    else:
        r0 = 10
        tparas = sci_optimize.leastsq(spherrors, np.asarray([xm, ym, zm, r0]),points,full_output=1)
        paras = tparas[0]
        s = paras[0:3]
        R = paras[3]
        mean_error = np.mean(errors)
        if type == 'sci':
            return s, R, mean_error
        elif type == 'ransac':
            return s, R, errors


def ransac_shape(pc : o3d.geometry.PointCloud, d_threshhold=0.1, max_iters=20, stop_score=1):
    max_inlier = 0
    for iter in  range(max_iters):
        tmp_pc = pc.random_down_sample(5 / len(pc.points))
        s,R,errors = sphere_surface(np.asarray(tmp_pc.points),type='ransac')
        inliers_mask = np.asarray(errors) < 0.1
        print(inliers_mask)
        # outliers_mask = np.logical_not(inliers_mask)
        inliers_num = np.sum(inliers_mask)
        print(inliers_num)
        if inliers_num > max_inlier:
            best_mask = inliers_mask
            best_errors = np.mean(errors)
        elif inliers_num == max_inlier and inliers_num != 0:
            m_error = np.mean(errors)
            if m_error < best_errors:
                best_mask = inliers_mask
                best_errors = np.mean(errors)
        if best_errors < stop_score:
            break
    tmp_pc = pc.select_by_index(best_mask)
    s, R, errors = sphere_surface(np.asarray(tmp_pc.points), type='sci')
    return s, R, errors


def spherrors(para, points):
    """球面拟合误差"""
    x0, y0, z0, r0 = para
    # error = (points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2 + (points[:, 2] - z0) ** 2 - r0 ** 2
    return (points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2 + (points[:, 2] - z0) ** 2 - r0 ** 2

def get_sphere_for_show(params):
    mesh = o3d.geometry.TriangleMesh()
    for s,R in params:
        if R < 1:
            rs = 0.001
        else:
            rs = 1
        fr = mesh.create_torus(torus_radius=R,tube_radius=3 * rs).translate(s)
        fr.paint_uniform_color([0.9, 0.1, 0.1])
        rotate_1 = o3d.geometry.get_rotation_matrix_from_axis_angle([0,np.pi / 2,0])
        sr = mesh.create_torus(torus_radius=R, tube_radius=3 * rs).translate(s).rotate(rotate_1,s)
        sr.paint_uniform_color([0.9, 0.1, 0.1])
        rotate_2 = o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi /2,0,0])
        tr = mesh.create_torus(torus_radius=R, tube_radius=3 * rs).translate(s).rotate(rotate_2,s)
        tr.paint_uniform_color([0.9, 0.1, 0.1])
        center = mesh.create_sphere(radius=4 * rs).translate(s)
        center.paint_uniform_color([0.9, 0.1, 0.1])
        mesh = mesh + fr + sr + tr + center
    return mesh

def lineseg_dist(p, a, b):

    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, 0])

    # perpendicular distance component
    c = np.cross(p - a, d)

    return np.hypot(h, np.linalg.norm(c))






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r'yolov5/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--offline', nargs='+', type=str, default=None, help='model.pt path(s)')
    parser.add_argument('--bagfile', nargs='+', type=str, default=r'C:\Users\think\Documents\20221212_190754.bag', help='.bag path(s)')
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--only1', action='store_true')
    opt = parser.parse_args()
    # print(opt)
    config_file = 'mmdetection/configs/cascade_rcnn/my_apple_config.py'
    checkpoint_file = 'epoch_12.pth'
    time_base = time.time()

    print("=" * 8 + "\b" + "RealSense Configuration" + "\b" + "=" * 8)
    pipeline = rs.pipeline()
    config = rs.config()
    if opt.replay:
        if isinstance(opt.bagfile, list):
            config.enable_device_from_file(opt.bagfile[0])
        else:
            config.enable_device_from_file(opt.bagfile)

    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

    profile_ = pipeline.start(config)
    align = rs.align(rs.stream.color)

    print(f"Done --- {time.time() - time_base:.4f}s.")
    yolodetector = YoloDetector(yolo_config)
    yolodetector.timecnt(True)
    yolodetector.verbose(True)
    yolodetector.show(True)
    print("=" * 8 + "\b" + "YOLO Initialization" + "\b" + "=" * 8)
    # yolo_detector = Detector(opt)

    print(f"Done --- {time.time() - time_base:.4f}s.")
    # model = init_detector(config_file, checkpoint_file, device='cuda:0')
    print("=" * 8 + "\b" + "Rendering Configuration" + "\b" + "=" * 8)

    pc = rs.pointcloud()
    pcd = o3d.geometry.PointCloud()
    lines_pcd = o3d.geometry.LineSet()

    # vis = o3d.visualization.Visualizer()
    # vis.create_window(width=1920, height=1080)  # 创建窗口
    # # vis.set_full_screen(True)
    # render_option = vis.get_render_option()  # 渲染配置
    # render_option.background_color = np.array([255, 255, 255])  # 设置点云渲染参数，背景颜色
    # render_option.point_size = 1.0  # 设置渲染点的大小
    # param = o3d.io.read_pinhole_camera_parameters('./viewpoint1.json')
    # ctr = vis.get_view_control()
    # ctr.convert_from_pinhole_camera_parameters(param)
    # vis.add_geometry(pcd)

    print(f"Done --- {time.time() - time_base:.4f}s.")
    if not opt.replay:
        for i in range(100):
            frames = pipeline.wait_for_frames()
    # print(f"Ready! Press any key to begin!")
    # input()

    frame = 1
    depth_scale = 1000

    while True:
        # color_image = cv2.imread(fr"D:\Files\dev\datasets\test\color_shot_{frame % 15}.png")
        # depth_image = cv2.imread(fr"D:\Files\dev\datasets\test\depth_frame_{frame % 15}.png")
        # o3d_color_image = o3d.io.read_image(fr"D:\Files\dev\datasets\test\color_shot_{frame % 15}.png")
        # o3d_depth_image = o3d.io.read_image(fr"D:\Files\dev\datasets\test\depth_frame_{frame % 15}.png")
        time_0 = time.time()
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        dprofile = aligned_depth_frame.get_profile()
        cprofile = color_frame.get_profile()

        cvsprofile = rs.video_stream_profile(cprofile)
        dvsprofile = rs.video_stream_profile(dprofile)


        color_intrin = cvsprofile.get_intrinsics()
        depth_intrin = dvsprofile.get_intrinsics()
        K = [[617.187, 0, 429.1957], [0,617.428,241.077], [0, 0, 1]]


        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        # o_color_image = color_image.copy()
        # print(depth_image.tolist())
        # color_image[np.where(depth_image > 1500)] = 0
        # cv2.imshow('0',color_image)
        # cv2.waitKey()
        # color_image = cv2.imread(r"color_shot_0.png",cv2.COLOR_BGR2RGB)
        # align_depth_image = depth_image = cv2.imread(r"depth_colormap_0.png",cv2.COLOR_RGB2GRAY)

        time1 = time.time()
        print(f"Image Processing Done in {time1 - time_0:.4f}s.", end='')
        with torch.no_grad():
            yolo_result = yolodetector.detect(color_image)
            # labeled_img, yolo_result = yolo_detector.one_img_result(color_image)
            # bbox_result, segment_result = inference_detector(model, o_color_image)
            # model.show_result(o_color_image, (bbox_result, segment_result),show=True)
            # bbox_result, segment_result = inference_detector(model, color_image)
            # print(bbox_result,segment_result)
            # model.show_result(color_image, (bbox_result,segment_result),show=True)
            # yolo_result = bbox_result[0]
            # yolo_result = yolo_result[np.where(yolo_result[:,-1]>0.65)[0].tolist()][:,:-1]
            # segment_result = np.array(segment_result[0])
            # segment_result = segment_result[np.where(yolo_result[:,-1]>0.65)[0].tolist()]
            # print(yolo_result)
            # input()

        time2 = time.time()
        if len(yolo_result) != 0:
            print(f"YOLO Detecting Done in {time2 - time1:.4f}s, {len(yolo_result)} apples detected.", end='')
        else:
            print(f"YOLO Detecting Done in {time2 - time1:.4f}s,  No Apple detected.")
            continue

        # process yolo_result

        final_3d_frustums = list()
        sub_images = list()
        for box in yolo_result:
            # sub_images.append(color_image[int(box[1]):int(box[3]),int(box[0]):int(box[2]),:])
            box = box[:4]
            mind, maxd = get_max_min_range(box, depth_image)
            print(mind,maxd)
            if mind is None or maxd is None:
                continue
            box3d_8_3 = list()
            for d in [mind, maxd]:
                for i in [0, 2]:
                    for j in [1, 3]:
                        box3d_8_3.append(rs.rs2_deproject_pixel_to_point(depth_intrin, (box[i], box[j]), d))
            final_3d_frustums.append(box3d_8_3)
        # sub_images_for_show = list()
        # for image in sub_images:
        #     print(sub_images[0].shape)
        #     sub_images_for_show.append(cv2.resize(image, (sub_images[0].shape[0],sub_images[0].shape[1])))
        #
        # sub_images_for_show = np.hstack(sub_images_for_show)
        # cv2.imshow('x',sub_images_for_show)

        time3 = time.time()
        print(f"Frustum Converting Done in {time3 - time2:.4f}s.", end='')

        color = color_image.reshape(-1,3)[:,[2,1,0]] / 255
        # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color_image,o3d_depth_image,depth_scale=depth_scale,convert_rgb_to_intensity=False)
        # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        #     rgbd_image,
        #     o3d.camera.PinholeCameraIntrinsic(848,480,617.187,617.428,429.1957,241.077))
        # o3d.visualization.draw_geometries([pcd])
        pc.map_to(color_frame)
        points = pc.calculate(aligned_depth_frame)
        vtx = np.asanyarray(points.get_vertices())
        vtx = np.asarray(vtx.tolist()) * 1000
        #
        new_vtx = np.concatenate((vtx,color),axis=1)

        pc_remain = list()

        pcd.points = o3d.utility.Vector3dVector(new_vtx[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(new_vtx[:, 3:])
        # all_mask = segment_result[0][0]
        # for mask in segment_result[0][1:]:
        #     all_mask = np.bitwise_or(all_mask, mask)
        # remain_index = np.flatnonzero(all_mask)
        # pcd_croped = o3d.geometry.PointCloud()
        # pcd_croped.colors = o3d.utility.Vector3dVector([pcd.colors[inx] for inx in remain_index])
        # pcd_croped.points = o3d.utility.Vector3dVector([pcd.points[inx] for inx in remain_index])
        # labels = pcd_croped.cluster_dbscan(eps=80, min_points=20, print_progress=False)
        # labels = np.asarray(labels)
        # # all_sum = [[label, np.sum(labels == label)] for label in np.unique(labels) if label != -1]
        # for label in np.unique(labels):
        #     pcd_ = pcd_croped.select_by_index(np.where(labels==label)[0].tolist())
        #     o3d.visualization.draw_geometries([pcd_])

        # remain_index = np.flatnonzero(segment_result[0][0])
        # pcd.points = o3d.utility.Vector3dVector([pcd.points[inx] for inx in remain_index])
        # pcd.colors = o3d.utility.Vector3dVector([pcd.colors[inx] for inx in remain_index])
        # o3d.visualization.draw_geometries([pcd])
        # print(np.flatnonzero(segment_result[0][0]))
        # input()
        # pcd.voxel_down_sample(10)
        # pcd.remove_non_finite_points()
        # pcd.remove_statistical_outlier(100,0.2)
        # pcd.remove_radius_outlier(100, 10)
        # lines_pcd.lines, lines_pcd.points, lines_pcd.colors = get_frustum_lines(final_3d_frustums)
        # vis.clear_geometries()
        # vis.add_geometry(pcd)
        # vis.add_geometry(lines_pcd)
        # vis.update_geometry(pcd)
        # vis.update_geometry(lines_pcd)
        # o3d.visualization.draw_geometries([pcd, lines_pcd])
        all_pcd = o3d.geometry.PointCloud()
        locations,radius,errors = list(), list(),list()
        for num_f, frustum in enumerate(final_3d_frustums):
        # depth_remain = np.where(np.asanyarray(rgbd_image.depth).flatten('C') != 0)[0]
        # for num_f in range(len(segment_result)):
        #     pcd_croped = o3d.geometry.PointCloud()
            remain_index = extract_pc_in_box3d(pcd.points, frustum)
            remain_index = np.where(remain_index)[0].tolist()
            # remain_index = np.flatnonzero(segment_result[num_f])
        #     remain_index = [np.where(depth_remain == index)[0][0] for index in remain_index if index in depth_remain]

            # pcd_croped.points = o3d.utility.Vector3dVector([pcd.points[inx] for inx, flag in enumerate(remain_index) if flag])
            # pcd_croped.colors = o3d.utility.Vector3dVector([pcd.colors[inx] for inx, flag in enumerate(remain_index) if flag])

            # pcd_croped.colors = o3d.utility.Vector3dVector([pcd.colors[inx] for inx in remain_index])
            # pcd_croped.points = o3d.utility.Vector3dVector([pcd.points[inx] for inx in remain_index])
            pcd_croped = pcd.select_by_index(remain_index)

            # o_pcd_croped = o3d.geometry.PointCloud()
            # o_pcd_croped.colors = o3d.utility.Vector3dVector(pcd_croped.colors)
            # o_pcd_croped.points = o3d.utility.Vector3dVector(np.asarray(pcd_croped.points) + 700)
            # o3d.visualization.draw_geometries([pcd_croped,o_pcd_croped])
            # bbox = pcd_croped.get_oriented_bounding_box()
        #     # lines_pcd.points = o3d.utility.Vector3dVector(frustum)
        #     o3d.visualization.draw_geometries([pcd_croped])
            pcd_croped, _ = pcd_croped.remove_statistical_outlier(600, 0.05)
            # pcd_croped, _ = pcd_croped.remove_radius_outlier(250, 60)
            eps, min_points = 30, 20
            labels = pcd_croped.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
            if all(np.asarray(labels) == -1):
                continue
            labels = np.asarray(labels) + 1
            all_sum = np.bincount(labels)
            max_label = np.argmax(all_sum) - 1
            pcd_croped = pcd_croped.select_by_index(np.where((labels-1) == max_label)[0].tolist())
            # print(len(pcd_croped.points))
            # pcd_croped_ = pcd_croped.select_by_index(np.where(labels==max_label)[0].tolist(),invert=True)
            # pcd_croped_.paint_uniform_color((1,0,0))
            # if input() == str(1):
            # o3d.visualization.draw_geometries([o_pcd_croped,pcd_croped])
            # if len(pcd_croped.points) < 500:
            #     continue
            pc_remain.append(pcd_croped)

        #     # centroid, _ = pcd_croped.compute_mean_and_covariance()
        #     # center_point = o3d.geometry.PointCloud()
        #     # center_point.points = o3d.utility.Vector3dVector(centroid)
        #     # sub_pc.points = o3d.utility.Vector3dVector([i for index, i in enumerate(pcd_croped.points) if labels[index] == label])
        #     # sub_pc.colors = o3d.utility.Vector3dVector([i for index, i in enumerate(pcd_croped.colors) if labels[index] == label])
        #     # points = np.asarray([[i[0],i[1],i[2]] for i in pcd_croped.points])
        #         # 计算点集的x,y,z均值，用于拟合参数初值
        #     s,R,mean_error = ransac_shape(pcd_croped)
            s,R,mean_error = sphere_surface(np.asarray(pcd_croped.points),type='sci')
            # print(pcd_croped.points[0])
        #     if R > 50:
        #         o3d.visualization.draw_geometries([pcd_croped])
            locations.append(s)
            radius.append(R)
            errors.append(mean_error)
            # if R > 70:
            #     print(R)
                # s, _ = pcd_croped.compute_mean_and_covariance()
                # pts = np.array(pcd_croped.points)
                # tmp = np.sort(pts[:,0]-s[0],axis=0)
                # print(tmp)
                # pts_r = np.min(pts[:])
                # # print(pts_r)
                # input()
                # r_2d = (yolo_result[num_f][2] - yolo_result[num_f][0]) + (yolo_result[num_f][3] - yolo_result[num_f][1])
                # r_2d = r_2d / 2
                # r_3d = r_2d / s[2] * 641


                # print(s,R)
            # o3d.visualization.draw_geometries([pcd_croped])
                # sphere = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(sub_pc,0.7)
            # pc_remain.append(copy.copy(pcd_croped))
            # pc_remain.append(center_point)

        meshes_for_show = get_sphere_for_show(zip(locations,radius))
        # print(errors)
        mean_error_ = np.mean(errors)
        std_error_ = np.std(errors)
        for index in np.where((errors-mean_error_)/std_error_ > 2)[0].tolist():
            print(index)
            pc_remain[index].paint_uniform_color((1,0,0))
        # o3d.io.write_point_cloud(f"yolov5/output/{frame}-point_cloud.pcd", pcd)
        # o3d.io.write_triangle_mesh(f"yolov5/output/{frame}-mesh.obj", meshes_for_show)
        # o3d.io.write_line_set(f"yolov5/output/{frame}-lineset.ply", lines_pcd)
        o3d.visualization.draw_geometries(pc_remain+[pcd,lines_pcd,meshes_for_show])
        # o3d.visualization.draw_geometries(pc_remain)

        # o3d.visualization.draw_geometries([pcd,lines_pcd]+pc_remain)
        # # for pc_ in pc_remain:
        # #     vis.add_geometry(pc_)  # 添加点云
        # # vis.add_geometry(pcd)
        #
        # # vis.run()
        # # vis.run()
        # # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        # # open3d.io.write_pinhole_camera_parameters(f'./viewpoint1.json', param)
        # # input()
        # # vis.poll_events()
        # # vis.update_renderer()
        #
        # # ctr.convert_from_pinhole_camera_parameters(param)
        # # input()
        # # o3d.visualization.draw_geometries([pcd, lines_pcd])
        time4 = time.time()
        print(f"Rendering Done in {time4 - time3:.4f}s.", end='\r')

        if opt.only1:
            break
        frame += 1
        # cv2.namedWindow('yolo_result')
        #
        # cv2.imshow('yolo_result', labeled_img)
        #
        # if cv2.waitKey() == ord('q'):
        #     break

    # pcd = o3d.io.read_point_cloud(os.path.join(os.path.split(img_name)[0],os.path.split(img_name)[-1].split('.')[0]) + '.pcd', format='auto')
    # pcd.remove_radius_outlier(16, 0.05)
    # o3d.visualization.draw_geometries([pcd])
