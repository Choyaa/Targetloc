import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import os
from pathlib import Path
from lib.transform import qvec2rotmat,rotmat2qvec, get_CRS

import pyproj
def generate_query_with_seeds(euler_angles, translation, exp_place):
    """
    Preprocess query images by adding seeds for prior pose and adjusting height based on DSM value.
    1. add 15 seeds for prior pose(delta yaw: [0, 30°, -30°], [delta x, delta y]: [[0, 0], [0, 5], [0, -5], [5, 0], [-5, 0]]
    2. height is always inaccurate, set height = dsm value + 1.5
    Args:
        dsm_filepath (str): Path to the DSM file.
        save_filepath (str): Path to save the processed query sequence.
        query_sequence (str): Path to the original query sequence file.
        dev (str): Device type, 'phone' by default.
    
    Returns:
        None
    """
    seeds = {}
    # Define translation deltas for seeds
    delta = [[0, 0, 0], [0, 5, 0], [0, -5, 0], [5, 0, 0], [-5, 0, 0]]
    wgs84, cgcs2000 = get_CRS(exp_place)
    
    transformer = pyproj.Transformer.from_crs(wgs84, cgcs2000, always_xy=True)

    x, y = np.array(transformer.transform(translation[0], translation[1]))  

    t = [x, y, translation[-1]]
    

    index = 0
    # Generate XY seeds
    for i in range(len(delta)):
        # Calculate new x, y with delta
        x, y, _ = t + delta[i]
        
        # Prepare the camera-to-world translation
        transformer_inverse = pyproj.Transformer.from_crs(wgs84, cgcs2000, always_xy=True)
        lon, lat = transformer_inverse.transform(x, y)
        t_c2w = [lon, lat, translation[-1]]
        
        # Generate YAW seeds
        euler_left, euler_right = generate_yaw_seeds(euler_angles)
        
        # Write the processed data to the output file
        for _, seed in enumerate([euler_angles, euler_left, euler_right]):
            seeds[index] = {'euler_angles':seed, 'translation': t_c2w}
            print(t_c2w, seed)
            index += 1
    return seeds
    


def generate_yaw_seeds(euler_xyz):
    """
    Generate two yaw-adjusted quaternions from a given quaternion by adding and subtracting 30 degrees.
    
    Args:
        qvec (list): A list representing the quaternion in the order [x, y, z, w].
    
    Returns:
        tuple: Two tuples, each containing a list that represents a quaternion.
    """
    
    # Create two sets of Euler angles with yaw angles increased and decreased by 30 degrees
    euler_left = euler_xyz.copy()
    euler_right = euler_xyz.copy()
    euler_left[0] += 1
    euler_right[0] -= 1
    

    # Return the two new quaternions
    return euler_left, euler_right   
def generate_angle_seeds(euler_xyz, delta_euler):
    """
    Generate two yaw-adjusted quaternions from a given quaternion by adding and subtracting 30 degrees.
    
    Args:
        qvec (list): A list representing the quaternion in the order [x, y, z, w].
    
    Returns:
        tuple: Two tuples, each containing a list that represents a quaternion.
    """
    
    # Create two sets of Euler angles with yaw angles increased and decreased by 30 degrees
    euler_angles = []
    # for i in range(-1, 2):
    #     for j in range(-1, 2):
    min_euler = -delta_euler
    max_euler = delta_euler + 1
    for i in range(min_euler, max_euler, delta_euler):
        for j in range(min_euler, max_euler, delta_euler):
            euler_angles.append([euler_xyz[0]+i, euler_xyz[1], euler_xyz[2]+j])




    # Return the two new quaternions
    return euler_angles


def main(dsm_file: str, seed_path: Path, prior_path: Path, dsm=False):
    """
    Main function to check if seeds exist and generate them if necessary.

    Args:
        dsm_file (str): Path to the DSM file used for height correction.
        seed_path (Path): Path to the directory where the seeds will be saved.
        prior_path (Path): Path to the directory where prior information is stored.
        dev (str): Optional device identifier. Defaults to an empty string if not provided.
        dsm (bool): Optional set height or a prior height , Defaults to prior height if not provided.
    Returns:
        None
    """
    # If the directory does not exist, generate the query with seeds
    generate_query_with_seeds(dsm_file, seed_path, prior_path, dsm)
