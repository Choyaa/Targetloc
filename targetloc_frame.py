import os
import sys

from lib import localize_render2loc_effloftr
cur_dir = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(cur_dir, "3DTilesRender/build/"))
import glob

import argparse
from pathlib import Path
import json
import copy
import yaml
import time
import pyproj
import copy
#import immatch
import cv2
import numpy as np
from lib import (
    eval,
    match_effloftr,
    match_roma,
    undistort,
    transform,
    read_EXIF,
    extract_global_features,
    pairs_from_retrieval
)
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import math
from lib.base64_img import encode_base64
from lib.transform import convert_euler_to_matrix, wgs84tocgcs2000, get_rotation_enu_in_ecef
from lib.strategies import get_protocol
from utils.osg import osg_render
from lib.transform import qvec2rotmat, rotmat2qvec, ECEF_to_WGS84, WGS84_to_ECEF
import os
from lib.read_EXIF import os_mkdir

import logging

formatter = logging.Formatter(
    fmt='[%(asctime)s %(name)s %(levelname)s] %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False
import warnings
 
warnings.filterwarnings("ignore")
class Render2Loc:
    def __init__(self, config_file='configs/config_local.json'): 
        # Load configuration from the provided JSON file
        with open(config_file) as fp:
            self.config = json.load(fp)
        
        # Define paths for datasets and results
        self.dataset = Path(os.path.join(cur_dir, self.config['render2loc']['datasets']))
        self.images = self.dataset / 'images/images_upright'
        self.outputs = self.dataset / self.config['render2loc']['results']
        self.render_camera = self.config['render2loc']['render_camera']
        self.gt_pose = self.dataset / self.config['evaluate']['gt_pose']
        self.engine = self.config['render2loc']['engine']
        self.dev = self.config['render2loc']['dev']
        self.distortion = self.config['render2loc']['distortion']
        self.data = dict()
        self.H_array = np.eye(3)
        self.f_mul = 1.5  # Focal length scaling factor
        # self.resolution_mul = 2.5  # Query image scaling factor (for different use cases)
        self.seeds = {}
        
        # Create directories for prior and result images
        os_mkdir(self.outputs/"prior_images")
        os_mkdir(self.outputs/"result_images")
        
        # Initialize matcher for feature matching
        self.matcher = match_effloftr.ImageMatcher()
        
        scaler_conf = {
            'max_angle': [1, 1, 1],
            'center_std': [50., 50., 50.]
        }
        protocol_conf = {
            'N_steps': 20,
            'n_views': 200,
            'M': 6 
        }
        self.resampler = get_protocol(self.config['render2loc']['sampler_name'], 
                                    self.config['render2loc']['scaler_name'], 
                                    protocol_conf,
                                    scaler_conf, 
                                    multi_seed = False)
        
        # Initialize localizer for render localization
        self.localizer = localize_render2loc_effloftr.QueryLocalizer(config=self.config)
        
        # Load 3D tiles renderer
        self.renderer = osg_render.RenderImageProcessor(self.config, Render2Loc.eglDpy)
        # Update EGL device address if necessary
        # Render2Loc.eglDpy = self.renderer.get_EGLDisplay()

    def receive_from_prior(self, image_path, euler_angles, translation, flight_type = 'M300', focal_len = 4.5):
        self.config['flight_type'] = flight_type
        self.query_path = image_path
        self.image_id = str(image_path.split('/')[-1].split('.')[0]) + '.jpg'
        self.query_image = cv2.imread(self.query_path, cv2.IMREAD_GRAYSCALE)  
        self.euler_angles = euler_angles
        self.translation = translation  # Position
        self.focal_len = focal_len  

    def receive_from_exif(self, image_path, localization_equip = None, flight_type = 'M300'):
        self.config['flight_type'] = flight_type
        self.query_path = image_path
        self.image_id = str(image_path.split('/')[-1].split('.')[0]) + '.jpg'
        self.query_image = cv2.imread(self.query_path, cv2.IMREAD_GRAYSCALE)  
        roll, pitch, yaw, lat, lon, alt, focal_len = read_EXIF.get_dji_exif(image_path, return_focal_lens=True) 
        if self.config['flight_type'] in ["M300", "M3T", "M3P"]:
            # Adjust angles for DJI camera (NED--->ENU)
            self.euler_angles = [90+pitch, roll, -yaw]
   
        self.translation = [lon, lat, alt]  # Location
        self.focal_len = focal_len

        # Set target point if specified
        if localization_equip == 'Z': 
            self.target_point = np.float64([[0, 0]])

    def receive_from_retrieval(self, image_path, euler_angles_list, translation_list, target_point = None, localization_equip=None, flight_type = 'M300', focal_len = 4.5):
        self.config['flight_type'] = flight_type
        self.query_path = image_path
        self.image_id = str(image_path.split('/')[-1]) + '.png'
        self.query_image = cv2.imread(self.query_path, cv2.IMREAD_GRAYSCALE)  
        self.focal_len = focal_len 
        # Handle multiple euler angles and translations
        if len(euler_angles_list) == 1:
            self.euler_angles = euler_angles_list[0]
            self.translation = translation_list[0]
        else:
            for i in range(len(euler_angles_list)):
                self.seeds[i] = {'euler_angles':euler_angles_list[i], 'translation': translation_list[i]}
        # Set target point
        self.target_point = np.float64(target_point)
        if self.target_point.ndim == 1:  # Ensure correct dimensionality
            self.target_point = np.expand_dims(self.target_point, axis=0)

    def update_render_pose(self, ret):
        '''
        input: ret['qvec'], ret['tvec'] in w2c format
        process: transform it into ENU format
        output: euler angles, translation in WGS84 format
        '''
        if ret['success']:
            R_c2w = ret['qvec']
            t_c2w = ret['tvec']
            print(R_c2w, t_c2w)  
            q_w2c = rotmat2qvec(R_c2w.transpose())  # return wxyz (colmap pnp return xyzw)
            t_w2c = np.array(-R_c2w.transpose().dot(t_c2w)) 
            print(q_w2c, t_w2c)   #! pixloc
            
            # Convert ECEF to WGS84
            self.translation = ECEF_to_WGS84(t_c2w)
            lon, lat, _ = self.translation
            # Calculate the rotation matrix from ENU to ECEF
            rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
            rot_ecef_in_enu = rot_ned_in_ecef.T  #! ECEF to WGS84 transformation
            
            # Convert the rotation matrix from ECEF to ENU
            rot_pose_in_enu = np.matmul(rot_ecef_in_enu, R_c2w)
            rot_pose_in_enu_obj = R.from_matrix(rot_pose_in_enu)
            
            self.euler_angles = rot_pose_in_enu_obj.as_euler('xyz', degrees=True)
            euler_angles_in_ned = [self.euler_angles[0], -self.euler_angles[1], -self.euler_angles[2]]
            
            logger.info("Euler angles in degrees(pitch, roll, yaw): %s", self.euler_angles)
            logger.info("Translation in WGS84: %s", self.translation)
            
            # Convert to NED format
            R_c2w = convert_euler_to_matrix(copy.deepcopy(euler_angles_in_ned))
            R_w2c_in_ned = R_c2w.transpose()  # Difference from ENU is negating second and third rows
            t_c2w = wgs84tocgcs2000(self.translation, 0)
            q_w2c = rotmat2qvec(R_w2c_in_ned)  # Return wxyz
            t_w2c = np.array(-R_w2c_in_ned.dot(t_c2w)) 
            
            ret['qvec'] = q_w2c
            ret['tvec'] = t_w2c
            print("qvec,tvec:", ret)

        return ret

        
    def delay_to_load_map(self, config_web):
        # Set camera angles and translation from the provided configuration
        self.euler_angles = config_web['euler_angles']  # Camera angles
        self.translation = config_web['translation']  # Position
        for i in range(100): 
            # Update pose for rendering multiple times
            self.renderer.update_pose(self.translation, self.euler_angles)

    def rendering(self, config_web):
        # Set image ID, query path, and read the query image
        self.image_id = str(config_web['image_id'])
        self.query_path = config_web['query_path']
        self.query_image = cv2.imread(self.query_path, cv2.IMREAD_GRAYSCALE)  # Query image path
        self.euler_angles = config_web['euler_angles']  # Camera angles
        self.translation = config_web['translation']  # Position  
        # Update pose for rendering
        self.renderer.update_pose(self.translation, self.euler_angles)
        cv2.imwrite(str(self.outputs/"normalFOV_images"/self.image_id), self.renderer.get_color_image())

    def generate_query_with_seeds(self):
        translation = self.translation.copy()
        euler_angles = self.euler_angles.copy()
        seeds = {}
        last_step = 1
        # Initialize seed values
        seed_t = [1, 1, 1]
        seed_e = [1, 0, 1]
        for step in range(last_step):
            self.resampler.init_step(step)
            _, render_ts, render_es = self.resampler.resample(self.image_id, translation, euler_angles, seed_e, seed_t, step)
            for index in range(len(render_es)):
                seeds[index] = {'euler_angles': render_es[index], 'translation': render_ts[index]}
            status = self.seed_select(seeds)
            if status is False:
                ValueError('Localization error')

    def seed_select(self, db_name_list):
        best_seed_sum = 0
        best_seed_index = 0

        for key, seed in self.seeds.items():
            self.euler_angles = seed['euler_angles']
            self.translation = seed['translation']
            for _ in range(10):
                self.renderer.update_pose(self.translation, self.euler_angles, ref_camera=self.render_camera)
            color_image = self.renderer.get_color_image()
            
            cv2.imwrite(str(self.outputs/"prior_images"/(str(key) + "_" + self.image_id)), self.renderer.get_color_image())
            
            # Perform image matching to evaluate the best seed
            match_res = self.matcher.match_single_pair(self.image_id.split('.')[0], self.query_image, color_image)
            logger.info("Seed %s: Matches Num: %d", db_name_list[key], len(match_res[0]))
            if len(match_res[0]) > best_seed_sum:
                best_seed_index = key
                best_seed_sum = len(match_res[0])
        
        if best_seed_sum > 100:
            logger.info("Choose the best seed: %s with Euler angles: %s and Translation: %s", db_name_list[key], self.euler_angles, self.translation)
            self.euler_angles = self.seeds[best_seed_index]['euler_angles']
            self.translation = self.seeds[best_seed_index]['translation']
            return True
        else:
            # Raise an error if no valid seed is found
            return ValueError("Could not find best seed")

    def Get_intrisics(self, localization_equip):
        flight_type = self.config['flight_type']  # Flight type: M300, M3T
        self.query_camera = copy.deepcopy(self.config['render2loc'][flight_type]['query_camera_' + localization_equip])
        self.query_camera.append(self.focal_len)

        h, w = self.query_image.shape
        self.resolution_mul = w / self.render_camera[0]
        
        self.query_image, _ = undistort.main(self.query_image, self.query_camera, self.distortion)
        
        self.query_camera = [q / self.resolution_mul for q in self.query_camera]
        self.query_camera[0] = int(self.query_camera[0])
        self.query_camera[1] = int(self.query_camera[1])
        
        self.query_image = cv2.resize(self.query_image, (self.query_camera[0], self.query_camera[1]))
        
        self.render_camera = copy.deepcopy(self.query_camera)
        logger.debug("render_camera in Get_intrisics ", self.render_camera)
        logger.debug("query_camera in Get_intrisics ", self.query_camera)

        use_wide_angle = False
        if use_wide_angle:
            # Adjust for wide-angle cameras
            self.render_camera[-1] = self.render_camera[-1] / self.f_mul
        
        return self.render_camera[0], self.render_camera[1]

    def UAV_localization_with_prior(self, localization_equip=None, iter=1, use_homo=False):
        start_time = time.time()
        
        for i in range(iter):
            logger.info("Translation and Euler angles for iteration %d: %s, %s", i, self.translation, self.euler_angles)
            for _ in range(50): 
                self.renderer.update_pose(self.translation, self.euler_angles, ref_camera=self.render_camera)
                color_image = self.renderer.get_color_image()

            out_path = str(self.outputs/"prior_images"/(localization_equip + '_' + self.image_id))
            cv2.imwrite(out_path, color_image)
            depth_image = self.renderer.get_depth_image() 
            
            # Perform image matching to establish 2D-2D correspondences
            match_res = self.matcher.match_single_pair(self.image_id.split('.')[0], self.query_image, color_image, save_loc_path=self.outputs)
            
            if use_homo:
                self.H_array = self.localizer.findseq7_homography(match_res, color_image, self.image_id.split('.')[0])
            
            # Perform depth map backprojection to establish 2D-3D correspondences
            ret, Point3D, point2d = self.localizer.estimate_query_pose(match_res, [self.translation, self.euler_angles], depth_image, query_camera=self.query_camera, render_camera=self.render_camera, homo=True)
            ret = self.update_render_pose(ret)
            self.render_camera = self.query_camera
        
        if localization_equip == 'W':
            self.render_camera_w = self.query_camera
        end_time = time.time()
        elapsed_time = (end_time - start_time) 
        logger.info(f"Execution time: {elapsed_time} seconds")    

        for i in range(10):            
            self.renderer.update_pose(self.translation, self.euler_angles, ref_camera=self.render_camera)
            cv2.imwrite(str(self.outputs/"result_images"/(self.image_id)), self.renderer.get_color_image())
        
        return ret, self.translation, self.euler_angles, Point3D, point2d, self.resolution_mul

    def homo_estimation(self, Points_3D, points2D, name, localization_equip='W'):
        # Perform homography estimation using 3D points and 2D points
        ret = self.localizer.homo_pnp(Points_3D, points2D, self.query_camera)
        ret = self.update_render_pose(ret)
        
        # Update pose and save the result image
        for i in range(10):            
            self.renderer.update_pose(self.translation, self.euler_angles, ref_camera=self.render_camera)
            cv2.imwrite(str(self.outputs/"result_images"/(name)), self.renderer.get_color_image())
        
        return ret, self.translation, self.euler_angles

    def target_location_with_UAV_pose(self, click_points=None, DSM=None, use_homo=False):
        # If DSM (Digital Surface Model) is not provided, use depth image
        if DSM is None:
            for _ in range(20):  
                self.renderer.update_pose(self.translation, self.euler_angles, ref_camera=self.render_camera_w)
                depth_image = self.renderer.get_depth_image()
            
            # Scale the target point according to the resolution multiplier
            self.target_point = self.target_point / self.resolution_mul
            # Backproject the target point using depth map
            ret = self.localizer.get_target_location(self.target_point, [self.translation, self.euler_angles], depth_image, self.render_camera[:5], self.render_camera)
        return ret

    def eval_position(self, pos_res, pos_RTK):
        # Evaluate the position error by comparing with RTK data
        ecef_res = WGS84_to_ECEF(pos_res)
        ecef_rtk = WGS84_to_ECEF(pos_RTK)

        pos_RTK_ECEF = np.array([ecef_res[0], ecef_res[1]])
        pos_res_ECEF = np.array([ecef_rtk[0], ecef_rtk[1]])
        error = np.sqrt((pos_res_ECEF[0] - pos_RTK_ECEF[0])**2 + (pos_res_ECEF[1] - pos_RTK_ECEF[1])**2)
        return error

        

if __name__ == "__main__":
    use_retrieval = True
    use_seeds = False
    render2loc = Render2Loc('configs/config_local_DJI_feicuiwan.json')
    file_path = "/mnt/sda/CityofStars/Queries/process/single/seq1/"
    db_file_path = "/mnt/sda/CityofStars/Render_db"
    db_pose_path = db_file_path+"/db_pose_osg.txt"
    ref_images = db_file_path+"/images_osg"
    folder_path =file_path+"images/W"
    folder_z_path =file_path+"images/Z"
    
    gt_pose_path = file_path+"poses/gt_pose.txt"
    gt_position_path = file_path + "poses/gt_position.txt"
    save_loc_path = file_path + "poses/estimate_pose.txt"
    save_pos_path = file_path + "poses/estimate_position.txt"
    save_error_path = file_path + "poses/pose_error.txt"
    pos_eval = file_path + "poses/pos_eval.txt"
    estimate_poses = {}
    config_web = {}
    img_list = glob.glob(folder_path + "/*.JPG")
    img_list = np.sort(img_list)
    
    z_image_list = glob.glob(folder_z_path + "/*.JPG")
    z_image_list = np.sort(z_image_list)
    #---db pose dict
    db_poses = {}
    with open(db_pose_path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            tokens = data.split()
            name = os.path.basename(tokens[0])
            e_c2w, t_c2w = np.split(np.array(tokens[1:], dtype=float), [3])
            db_poses[name] = {}
            db_poses[name]['euler_angles'] = e_c2w  # Euler angles from quaternion
            db_poses[name]['translation'] = t_c2w  # Translation vector

    if not use_retrieval:
        # If not using retrieval, use EXIF data from the first image
        roll_value, pitch_value, yaw_value, gps_latitude, gps_longitude, gps_altitude = read_EXIF.get_dji_exif(img_list[0])
        config_web['euler_angles'] = [90 + pitch_value, roll_value, -yaw_value]  # Adjust DJI camera angles
        config_web['translation'] = [gps_longitude, gps_latitude, gps_altitude]
        render2loc.delay_to_load_map(config_web)  # Load map using the configuration
    else:
        gt_poses = {}
        topk = 5
        retrieval_conf = extract_global_features.confs['netvlad']
        global_descriptors = extract_global_features.main(retrieval_conf, Path(ref_images), 'ref_global_descriptors.h5', db_file_path)

    start_time = time.time()
    with open(save_loc_path, 'w') as f, open(save_pos_path, 'w') as f1:
        for i in tqdm(range(len(img_list))):
            image_path = img_list[i]
            
            target_point = [0, 0]
            if not use_retrieval:
                render2loc.receive_from_exif(image_path) 
            else:
                # Perform feature extraction and retrieval
                query_global_descriptors = extract_global_features.main(retrieval_conf, Path(folder_path), 'query_global_descriptors.h5', file_path, image_list=[image_path])
                euler_angles_list, translation_list, db_name_list = pairs_from_retrieval.main(query_global_descriptors, topk, db_poses, db_descriptors=global_descriptors)

                render2loc.receive_from_retrieval(image_path, euler_angles_list, translation_list, target_point)

                if topk > 1:
                    # Select the best seed based on the retrieval results
                    render2loc.seed_select(db_name_list)
            
            render2loc.Get_intrisics()
            
            if use_seeds:
                render2loc.generate_query_with_seeds()

            # Perform UAV localization with prior knowledge
            ret, trans_res, euler_res = render2loc.UAV_localization_with_prior(iter=2)
            
            # Process Z localization equipment
            image_path = z_image_list[i]
            render2loc.receive_from_exif(image_path)
            render2loc.Get_intrisics(localization_equip='Z')
            ret = render2loc.UAV_localization_with_prior(localization_equip='Z', iter=2)
            
            # Target location indication
            if ret['success']:
                qname = image_path.split('/')[-1]
                qvec = ' '.join(map(str, ret['qvec']))
                tvec = ' '.join(map(str, ret['tvec']))
                name = qname.split('/')[-1]
                f.write(f'{name} {qvec} {tvec}\n')

                ecef_res, wgs84_res = render2loc.target_location_with_UAV_pose()
                cgcs2000_res = wgs84tocgcs2000(wgs84_res[0], 0)
                position = ' '.join(map(str, cgcs2000_res))
                f1.write(f'{name} {position}\n')
            else:
                qvec = ' '.join(map(str, [0, 0, 0, 0]))
                tvec = ' '.join(map(str, [0, 0, 0]))
                f.write(f'{name} {qvec} {tvec}\n')
                position = ' '.join(map(str, [0, 0, 0]))
                f1.write(f'{name} {position}\n')

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 50
    logger.info(f"Execution time: {elapsed_time} seconds")

    # Evaluate pose by comparing ground truth and calculated poses
    eval.pose(gt_pose_path, save_loc_path)

    # eval.position(gt_position_path, save_pos_path, pos_eval)