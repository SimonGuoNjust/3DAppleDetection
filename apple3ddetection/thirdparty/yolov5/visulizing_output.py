import open3d as o3d
import queue
import threading
import os
import time

def visulizing_pc(show_q):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)
    pointcloud = o3d.geometry.PointCloud()
    meshes = o3d.geometry.TriangleMesh()
    lineset = o3d.geometry.LineSet()
    to_reset = True
    vis.add_geometry(pointcloud)
    vis.add_geometry(meshes)
    vis.add_geometry(lineset)

    # param = o3d.io.read_pinhole_camera_parameters('./viewpoint1.json')
    # ctr = vis.get_view_control()
    # ctr.convert_from_pinhole_camera_parameters(param)
    while True:
        try:
            all_for_show = show_q.get()
            if all_for_show == -1:
                break
            pcd = all_for_show[0]
            pointcloud.points = o3d.utility.Vector3dVector(pcd.points)
            pointcloud.colors = o3d.utility.Vector3dVector(pcd.colors)
            mesh = all_for_show[1]
            meshes.clear()
            meshes += mesh
            l = all_for_show[2]
            lineset.points = l.points
            lineset.lines = l.lines
            # vis.update_geometry()
            # 注意，如果使用的是open3d 0.8.0以后的版本，这句话应该改为下面格式
            vis.update_geometry(pointcloud)
            vis.update_geometry(meshes)
            vis.update_geometry(lineset)
            if to_reset:
                vis.reset_view_point(True)
                to_reset = False
            vis.poll_events()
            vis.update_renderer()
            # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            # o3d.io.write_pinhole_camera_parameters(f'./viewpoint1.json', param)
        except Exception as e:
            print(e)
            continue



if __name__ == '__main__':
    stop_q = queue.Queue(1)
    show_q = queue.Queue(1)
    visual = threading.Thread(target=visulizing_pc, args=(show_q,))
    visual.start()

    input_dir = r"yolov5\output"
    frame = 0
    while True:
        try:
            time.sleep(0.03)
            pcd_name = os.path.join(input_dir, str(frame)+"-point_cloud.pcd")
            mesh_name = os.path.join(input_dir, str(frame)+"-mesh.obj")
            lineset_name = os.path.join(input_dir, str(frame)+"-lineset.ply")
            print(frame)
            # 获取雷达数据
            pcd = o3d.io.read_point_cloud(pcd_name)
            mesh = o3d.io.read_triangle_mesh(mesh_name)
            lineset = o3d.io.read_line_set(lineset_name)

            if show_q.full():
                show_q.get()
            show_q.put([pcd,mesh,lineset])
            frame += 1  # 迭代读取下一张图片
            frame %= 9  # 由于文件夹中最多只有98图片，超出了，又会回到0,循环
        except KeyboardInterrupt:
            if show_q.full():
                show_q.get()
            show_q.put(-1)
            break
