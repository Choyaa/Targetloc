import cv2
import numpy as np
import os
from osgeo import gdal
import time
from pathlib import Path
import torch
import pyproj
import matplotlib
from lib.read_model import parse_pose_list, parse_intrinsic_list
from typing import Dict
from scipy.ndimage import map_coordinates
#import matplotlib.pyplot as plt
from .transform import qvec2rotmat, get_CRS
#matplotlib.use('TKAgg')


#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class TargetLocation():
    def __init__(self, config: Dict, use_dsm = False):
        self.config = config
        self.clicked_points = []
        
        if use_dsm:
            DSM_path = self.config["ray_casting"]["DSM_path"]
            DSM_npy_path = self.config["ray_casting"]["DSM_npy_path"]
            geotransform_path = self.config["ray_casting"]["geotransform_path"]

            self.num_sample = self.config["ray_casting"]["num_sample"]
            self.area_minZ = self.config["ray_casting"]["area_minZ"]
            
            # open DSM map
            cur_dir = os.path.abspath(os.path.dirname(__file__))
            print("DEBUG: DSM_path ", cur_dir + '/../' + DSM_path)
            dsm_path = cur_dir + '/../' + DSM_path
            npy_path = cur_dir + '/../' + DSM_npy_path
            geotransform_path = cur_dir + "/../" + geotransform_path

            if os.path.isfile(npy_path):
                self.area = np.load(npy_path)
                self.geotransform = np.load(geotransform_path)
            else:
                self.area, self.geo_transform = self.dsm2npy(dsm_path, npy_path, geotransform_path)
                print("DEBUG: self.area_minZ", self.area_minZ)


    @staticmethod
    def dsm2npy(dsm_path, npy_path, geotransform_path):
        DSM_map = gdal.Open(dsm_path)
        band = DSM_map.GetRasterBand(1) # 读取波段1
        geotransform = DSM_map.GetGeoTransform() # 读取左上角坐标等
        area = band.ReadAsArray()
        del DSM_map
        mask = np.ma.masked_values(area, -9999)
        area_minZ = mask.min()
        print("DEBUG: area_minZ, area_avgZ, area_maxZ ", area_minZ, mask.mean(), mask.max())

        np.save(npy_path, area)
        np.save(geotransform_path, geotransform)
        return area, geotransform

    def dms_to_dd(self, d, m, s):
        return d + (m / 60) + (s / 3600)

    def dd_to_dms(self, DD):
        degrees = int(DD)
        minutes = int((DD - degrees) * 60)
        seconds = (DD - degrees - minutes / 60) * 3600
        seconds = round(seconds, 2)
        dms = f"{degrees}\u00B0{minutes}\u2032{seconds}\u2033"
        return dms, degrees, minutes, seconds



    def interpolate_along_line(self, area, x, y, num_points):
        # 构造一个网格坐标系
        # xx, yy = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        
        # 根据采样点的坐标，从图像中取得对应的像素值
        sample_values = map_coordinates(area, [y, x], order=1)
        # sample_values = np.max(sample_values, axis=1).reshape(-1, 1)

        # 将采样点对应的像素值重新排列成num_samples长度的数组
        sample_array = sample_values.reshape((num_points,))

        return sample_array

    def pixel_to_world_coordinate(self, K, R, t, u, v):
        # 将2D像素坐标转换为相机坐标系下的坐标
        p_camera = np.array([[u], [v], [1]])
        p_camera = np.linalg.inv(K).dot(p_camera)

        # 将相机坐标系下的坐标转换为世界坐标系下的坐标
        p_world = R.dot(p_camera) + t

        return p_world

    def get_index_array(self, dataset):
        # 读取tif图像
        ds = dataset

        # 获取地理信息
        geotransform = ds.GetGeoTransform()
        x_origin, x_pixel_size, _, y_origin, _, y_pixel_size = geotransform

        # 获取图像大小
        rows, cols = ds.RasterYSize, ds.RasterXSize

        # 生成x坐标数组和y坐标数组
        x = np.arange(cols) * x_pixel_size + x_origin
        y = np.arange(rows) * y_pixel_size + y_origin

        # index_array = [x, y]
        # 生成索引数组
        # xx, yy = np.meshgrid(x, y)
        # index_array = np.stack([yy.ravel(), xx.ravel()], axis=1)
        return x, y

    def line_equation_3d(self, point1, point2):
        """
        求两点所在直线的方程

        :param point1: 第一个点的坐标，形如 [x1, y1, z1]
        :param point2: 第二个点的坐标，形如 [x2, y2, z2]
        :return: 直线方程的系数，形如 [a, b, c, d]，表示 ax + by + cz + d = 0
        """
        # 将点转换成向量形式
        p1 = np.array(point1)
        p2 = np.array(point2)

        # 求解直线方向向量
        direction = p2 - p1

        # 求解直线方程系数
        a, b, c = direction
        d = -(a * p1[0] + b * p1[1] + c * p1[2])

        return [a, b, c, d]


    def line_equation_2d(self, x1, y1, x2, y2):
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        return [A, B, C]

    def line_equation(self, A, B, Z):
        """
        计算射线和投影直线的方程
        :param A: A点坐标 (x,y,z)
        :param B: B点坐标 (x1,y1,z1)
        :param Z: 平面Z的值
        :return: 射线方程和投影直线方程
        """
        # 计算射线方程
        x, y, z = A
        x1, y1, z1 = B
        t = np.array([x1 - x, y1 - y, z1 - z])
        ray = lambda k: np.array([x, y, z]) + k * t

        # 计算投影直线方程
        k = (Z - z) / t[2]
        projection = ray(k)[:2]

        return ray, projection

    def intersection(self, ray_eqn, Z):
        """
        计算射线与平面Z的交点
        :param ray_eqn: 射线方程
        :param Z: 平面Z的值
        :return: 交点坐标
        """
        # 计算k值
        k = (Z - ray_eqn(0)[2]) / (ray_eqn(1)[2] - ray_eqn(0)[2])
        
        # 计算交点坐标
        intersection_point = ray_eqn(k)
        
        return intersection_point

    def geo_coords_to_array_index(self, x, y, geotransform):
        # (685659.0, 0.5, 0.0, 3255878.0, 0.0, -0.5) 隔离点
        x_origin, x_pixel_size, _, y_origin, _, y_pixel_size = geotransform
        col = ((x - x_origin) / x_pixel_size).astype(int)
        row = ((y - y_origin) / y_pixel_size).astype(int)
        
        return row, col

    def sample_points_on_line(self, line_equation, num_sample, x_minmax):
        # 根据斜截式计算 y 范围
        A, B, C = line_equation[0], line_equation[1], line_equation[2]
        x_min, x_max = x_minmax[0], x_minmax[1]

        # 在 y 范围内均匀采样 num_points 个点
        x = np.linspace(x_min, x_max, num_sample)
        y = (-A/B)*x - (C/B)

        return x, y

    def find_z(self, ray_eqn, points):
        """
        计算投影直线上的点在射线方程中对应的Z值
        :param ray_eqn: 射线方程
        :param points: 投影直线上的点的平面坐标 (x,y)
        :return: Z值列表
        """
        # 计算k值
        k = (points[0] - ray_eqn(0)[0]) / (ray_eqn(1)[0] - ray_eqn(0)[0])
        # z_values = (-d-a*x-b*y)/c
        
        # 计算Z值
        z_values = [ray_eqn(k_i)[2] for k_i in k]
        
        return z_values

    def caculate_predictXYZ(self, K, pose, objPixelCoords):
        R = pose[:3, :3]
        t = pose[:3, 3].reshape([3,1])

        target = self.pixel_to_world_coordinate(K,R,t,objPixelCoords[0],objPixelCoords[1])

        ray_eqn, projection_eqn = self.line_equation(t, target, self.area_minZ)
        line2D_abcd = self.line_equation_2d(t[0], t[1], target[0], target[1])
        

        intersection_point = self.intersection(ray_eqn, self.area_minZ)
        x_minmax = [t[0],intersection_point[0]]

        # 直线采样
        x, y = self.sample_points_on_line(line2D_abcd, self.num_sample, x_minmax)

        # DSM采样,先将xy对应的地理信息坐标索引求到
        row, col = self.geo_coords_to_array_index(x, y, self.geotransform)
        sampleHeight = self.interpolate_along_line(self.area, col ,row, self.num_sample)

        #得到三维直线上的z值
        z_values = self.find_z(ray_eqn,[x, y])
        z_values = torch.tensor(np.asarray(z_values))
        z_values = z_values.squeeze()

        #寻找最近点
        sampleHeight = torch.tensor(sampleHeight)
        abs_x = torch.abs(z_values - sampleHeight)
        min_val, min_idx = torch.min(abs_x, dim=0)
        # print(min_val)

        # print ("Resulting time cost: ",time.time()-start_time_2,"s") 

        # import pdb; pdb.set_trace()
        
        return [x[min_idx], y[min_idx], z_values[min_idx]]
    def get_intrinsic(self, camera):
            """
            计算35mm等效焦距和内参矩阵。
            
            参数:
            image_width_px -- 图像的宽度（像素）
            image_height_px -- 图像的高度（像素）
            sensor_width_mm -- 相机传感器的宽度（毫米）
            sensor_height_mm -- 相机传感器的高度（毫米）
            
            返回:
            K -- 内参矩阵，形状为3x3
            """
            image_width_px, image_height_px, sensor_width_mm, sensor_height_mm, f_mm = camera
            # 计算焦距在x和y方向上的比率
            focal_ratio_x = f_mm / sensor_width_mm
            focal_ratio_y = f_mm / sensor_height_mm
            
            # 计算内参矩阵中的焦距和主点坐标
            fx = image_width_px * focal_ratio_x
            fy = image_height_px * focal_ratio_y
            cx = image_width_px / 2
            cy = image_height_px / 2
            
            # 构建内参矩阵 K
            K = [[fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]]
            
            return K
    def get_query_intrinsic(self, camera):
        """
        计算35mm等效焦距和内参矩阵。
        
        参数:
        image_width_px -- 图像的宽度（像素）
        image_height_px -- 图像的高度（像素）
        sensor_width_mm -- 相机传感器的宽度（毫米）
        sensor_height_mm -- 相机传感器的高度（毫米）
        
        返回:
        K -- 内参矩阵，形状为3x3
        """
        image_width_px, image_height_px, sensor_width_mm, sensor_height_mm, f_mm = camera
        # 计算内参矩阵中的焦距
        focal_ratio_x = f_mm / sensor_width_mm
        focal_ratio_y = f_mm / sensor_height_mm
        
        fx = image_width_px * focal_ratio_x
        fy = image_height_px * focal_ratio_y

        # 计算主点坐标
        cx = image_width_px / 2
        cy = image_height_px / 2

        # 构建内参矩阵 K
        K = [[fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]]
        
        return K
    # def get_query_intrinsic(self, camera):
    #         """
    #         计算35mm等效焦距和内参矩阵。
            
    #         参数:
    #         image_width_px -- 图像的宽度（像素）
    #         image_height_px -- 图像的高度（像素）
    #         sensor_width_mm -- 相机传感器的宽度（毫米）
    #         sensor_height_mm -- 相机传感器的高度（毫米）
            
    #         返回:
    #         K -- 内参矩阵，形状为3x3
    #         """
    #         image_width_px, image_height_px, fx, fy, cx, cy = camera
    #         # 计算内参矩阵中的焦距和主点坐标
            
    #         # 构建内参矩阵 K
    #         K = [[fx, 0, cx],
    #             [0, fy, cy],
    #             [0, 0, 1]]
            
    #         return K 
    def get_pose(self, ret):
        q, t = ret['qvec'], ret['tvec']
        # Convert the quaternion to a rotation matrix
        R = np.asmatrix(qvec2rotmat(q)).transpose()
        
        # Initialize a 4x4 identity matrix
        T = np.identity(4)
        
        # Set the rotation and translation components
        T[0:3, 0:3] = R
        T[0:3, 3] = -R.dot(t)
        
        return T

    def estimate_target_location(self, ret, clicked_points, query_camera, exp_place):
        if ret['success']:
            K_w2c = self.get_query_intrinsic(query_camera)
            T_c2w = self.get_pose(ret)
            
            predict_xyz = self.caculate_predictXYZ(K_w2c, T_c2w, clicked_points)

            wgs84, cgcs2000 = get_CRS(exp_place)
            
            transformer = pyproj.Transformer.from_crs(cgcs2000, wgs84,always_xy=True)
            lon, lat = transformer.transform(predict_xyz[0], predict_xyz[1])
            dms_longitude, d_lon, m_lon, s_lon = self.dd_to_dms(lon[0])
            dms_latitude, d_lat, m_lat, s_lat = self.dd_to_dms(lat[0])
            # wgs84_res  = [dms_longitude, dms_latitude, predict_xyz[2].item()]
            wgs84_res = [lon[0], lat[0], predict_xyz[2].item()]
            
            # cv2.circle(query_image, (clicked_points[0], clicked_points[1]), 30, (0, 0, 255), -1)
            # cv2.putText(query_image, f"({d_lon}' {m_lon}' {s_lon}'', {d_lat}' {m_lat}' {s_lat}'')", \
            #     (clicked_points[0] + 60, clicked_points[1] + 60), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, (0, 0, 255), 10)
            # image_rgb = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
            # plt.figure(figsize=(15, 12))
            # plt.imshow(image_rgb)
            # plt.title("Image with object coordinate")
            # print('object corrdinate:',  coord)
            
            # plt.axis('off')
            # plt.show()
            cgcs2000_res = [predict_xyz[0].item(), predict_xyz[1].item(), predict_xyz[2].item()]
            # return [predict_xyz[0].item(), predict_xyz[1].item(), predict_xyz[2].item()] # G2000
            return cgcs2000_res, wgs84_res
    def estimate_target_location_byHOMO(self, depth, H, name):   # points[n,2]
        # 验证投影点实验，目前无法投过去
        def project_points(points, K, R, T):
            
            world_points = np.array(points)
            num = world_points.shape[0]
            world_points_h = np.hstack((world_points, np.ones((num, 1))))
            extrinsic = np.hstack((R, T.reshape(-1, 1)))
            
            came_p_h = extrinsic @ world_points_h.T
            image_p_h = K @ came_p_h
            image_p = image_p_h[:2]/image_p_h[2]
            
            return image_p.T
        
        
        R_w2c, t_w2c = self.get_pose_w2c([self.translation, self.euler_angles])
        render_K = self.get_intrinsic(self.render_camera)
        # points3D = [[40965443, 2644720, 34]]
        points3D = [[40965477.517, 2644873.607, 34]]   # 23 49 00 117 45 33
        
        img_2d = project_points(points3D, render_K, R_w2c, t_w2c)
        # breakpoint()
        # 2d to 3d
        for i in range(10):            
            self.renderer.update_pose(self.translation, self.euler_angles)  # 每一次渲染都会
        color_image = self.renderer.get_color_image()
        cv2.imwrite("28/loca_results/valid_pose/test.jpg", color_image)
        img1 = cv2.imread("28/loca_results/valid_pose/test.jpg")
        cv2.circle(img1, (int(img_2d[0][0]), int(img_2d[0][1])), 10, (255,0,0),-1)
        cv2.imwrite("28/loca_results/valid_pose/test_point1.jpg", img1)
        
        
        # breakpoint()
        color_image = self.renderer.get_color_image()
        depth_image = self.renderer.get_depth_image()
        # breakpoint()
        
        K_c2w = torch.tensor(render_K).inverse()
        render_T = torch.tensor(self.get_pose([self.translation, self.euler_angles])).float()
        R = render_T[:3, :3]
        t = render_T[:3, 3]
        # img_2d = [[int(img_2d[0][1]), int(img_2d[0][0])]]
        img_2d = torch.tensor(img_2d)
        if img_2d.shape[-1] != 3:
            points_2D = torch.cat([img_2d, torch.ones_like(img_2d[ :, [0]])], dim=-1)
            points_2D = points_2D.T  
        t = torch.unsqueeze(t,-1).repeat(1, points_2D.shape[-1])
        
        depth_test = depth_image[int(img_2d[0][0].numpy()), int(img_2d[0][1].numpy())]
        Points_3D = R @ K_c2w @ (torch.tensor([depth_test]) * points_2D.float()) + t  
        # breakpoint()
      
        
        height, width = depth.shape[:2]
        rows,cols = np.arange(height), np.arange(width)
        Y, X = np.meshgrid(rows, cols, indexing='xy')
        points_ref = torch.tensor((np.stack([Y, X], axis=-1)).reshape(-1, 2))
        
        points_query = (np.stack([Y, X], axis=-1)).reshape(-1, 2)
        points_homogeneous = np.hstack((points_query, np.ones((points_query.shape[0], 1))))
        transformed_points_homogeneous = H.dot(points_homogeneous.T).T
        transformed_points = transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, 2, np.newaxis]
        transformed_points = transformed_points.reshape(height, width, 2)
        
        
        render_K = torch.tensor(self.get_intrinsic(self.render_camera)).float()
        K_c2w = render_K.inverse()
        
        # import matplotlib.pyplot as plt
        # depth_img = depth.reshape(height, width).numpy()
        # min_pixel = np.min(depth_img)
        # max_pixel = np.max(depth_img)
        # nor_depth_img = (depth_img - min_pixel) / (max_pixel - min_pixel)
        # plt.imshow(nor_depth_img, cmap='gray')
        # plt.colorbar()
        # plt.show()
        # Get render pose
        render_T = torch.tensor(self.get_pose([self.translation, self.euler_angles])).float()
        R = render_T[:3, :3]
        t = render_T[:3, 3]
        if points_ref.shape[-1] != 3:
            points_2D = torch.cat([points_ref, torch.ones_like(points_ref[ :, [0]])], dim=-1)
            points_2D = points_2D.T  
        t = torch.unsqueeze(t,-1).repeat(1, points_2D.shape[-1])
        depth_reshape = torch.tensor(depth.reshape(-1))
        Points_3D = R @ K_c2w @ (depth_reshape * points_2D) + t  
        
        import pdb; pdb.set_trace()
        # ya
        depth_900 = self.read_valid_depth([[900., 900.]] , depth)
        Points_3D_ya = R @ K_c2w @ (depth_900[0] * torch.tensor([900,900,1])) + t[:, 900] 
        # reshape
        Points_3D_900 = R @ K_c2w @ (torch.tensor(depth[900, 900]) * torch.tensor([900,900,1])) + t[:, 900] 
         
        # import pdb;pdb.set_trace()
        Points_3D_ref = Points_3D.reshape(3, height, width)
        import pyproj
        
        wgs84 = pyproj.CRS('EPSG:4326')
        cgcs2000 = pyproj.CRS('EPSG:4529')
        transformer = pyproj.Transformer.from_crs(cgcs2000, wgs84, always_xy=True)
        ref_lon, ref_lat = transformer.transform(Points_3D_ref[0], Points_3D_ref[1])
        
        Points_3D_query = np.zeros(Points_3D_ref.shape) # [3, h, w]
        Points_latlon_query = np.zeros(Points_3D_ref.shape)
        Points_latlon_ref = np.zeros(Points_3D_ref.shape)
        
        Points_latlon_ref[0] = ref_lon
        Points_latlon_ref[1] = ref_lat
        Points_latlon_ref[2] = Points_3D_ref[2]
        # import pdb;pdb.set_trace()
        for i in tqdm(range(width)):
            for j in range(height):
                # import pdb;pdb.set_trace()
                if transformed_points[i][j][0] > 0 and transformed_points[i][j][0] < width and transformed_points[i][j][1] > 0 and transformed_points[i][j][1] < height:
                    cgcs_x = Points_3D_ref[0,int(transformed_points[i][j][0]), int(transformed_points[i][j][1])]
                    cgcs_y = Points_3D_ref[1,int(transformed_points[i][j][0]), int(transformed_points[i][j][1])]
                    cgcs_z = Points_3D_ref[2,int(transformed_points[i][j][0]), int(transformed_points[i][j][1])]
                    # lon, lat, alt = transform.cgcs2000towgs84(np.array([[cgcs_x.numpy(), cgcs_y.numpy(), cgcs_z.numpy()]]))
                    Points_3D_query[0, i, j] = cgcs_x
                    Points_3D_query[1, i, j] = cgcs_y
                    Points_3D_query[2, i, j] = cgcs_z
                    Points_latlon_query[2, i, j] = cgcs_z
                else:
                    break
        
        lon, lat = transformer.transform(Points_3D_query[0], Points_3D_query[1])
        Points_latlon_query[0] = lon
        Points_latlon_query[1] = lat
        
        temp_name = name + '.npy'
        save_path = str(self.outputs/"position"/temp_name)
        np.save(save_path, Points_latlon_query)
        print("-------------------Save Position------------------------")
        # import pdb;pdb.set_trace()
        
        return Points_latlon_query[:, height//2, width//2]
if __name__ == "__main__":
    
    config = {
    "render2loc": {
        "datasets": "/home/ubuntu/Documents/code/Render2loc/datasets/demo8",
        "image_path":"images/images_upright/query",
        "query_camera": "queries/query_intrinsics.txt",
        "query_pose": "results/1_estimated_pose.txt",
        "ray_casting": {
            "object_name": "pedestrian1",
            "num_sample": 100,
            "DSM_path": "/home/ubuntu/Documents/code/Render2loc/datasets/demo7/texture_model/dsm/DSM_merge.tif",
            "DSM_npy_path": "/mnt/sda/feicuiwan/DSM_array.npy",
            "area_minZ": 20.580223,
            "write_path": "./predictXYZ.txt"
    }
    }
    }
    # main(config)
