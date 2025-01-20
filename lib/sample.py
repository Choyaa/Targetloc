import torch
from lib.transform import ECEF_to_WGS84, WGS84_to_ECEF
import numpy as np
def get_sampler(sampler_name, scaler_name, scaler_conf=None):    
    if scaler_conf is None:
        scaler_conf = {
            'max_angle': [2, 2, 2],
            'center_std': [5, 5, 5]
        }
    if sampler_name == 'rand':
        sampler_class = RandomGaussianSampler
    elif sampler_name == 'rand_yaw_or_pitch':
        sampler_class = RandomSamplerByAxis
    elif sampler_name == 'rand_yaw_and_pitch':
        sampler_class = RandomDoubleAxisSampler
    elif sampler_name == 'rand_six_axis':
        sampler_class = RandomSixAxisSampler
    elif sampler_name == 'uniform_six_axis':
        sampler_class = UniformSixAxisSampler
    else:
        raise NotImplementedError()
    if scaler_name == 'constant':
        scaler_class = ConstantScaler
    elif scaler_name == 'uniform':
        scaler_class = UniformScaler
    scaler = scaler_class(scaler_conf)
    sampler = sampler_class()
    return sampler, scaler
    
class RandomGaussianSampler():
    def __init__(self):
        pass
    
    def sample_batch(self, n_views, center_std, max_angle, old_t, old_e):
        euler_angles_list = []
        translation_list = []

        for _ in range(n_views):
            new_t, new_e = self.sample(center_std, max_angle, old_t, old_e)

            euler_angles_list.append(new_e)
            translation_list.append(new_t)

        return euler_angles_list, translation_list

    @staticmethod
    def sample(center_std, max_angle, old_t, old_e):

        teta = np.random.rand()*max_angle # sample random angle smaller than theta
        new_e = old_e * teta  # perturb the original pose

        old_xyz = WGS84_to_ECEF(old_t)
        perturb_c = torch.normal(0., center_std)
        new_xyz = old_xyz + np.array(perturb_c) # perturb it 
        new_t = ECEF_to_WGS84(new_xyz)

        return new_t, new_e


class RandomDoubleAxisSampler():
    rotate_axis = {
        'pitch': [1, 0, 0], # x, pitch
        'yaw':   [0, 0, 1] # z, yaw
    }
    
    def __init__(self):
        pass
    
    def sample_batch(self, n_views: int, center_std: torch.tensor, max_angle: torch.tensor, 
                     old_t: np.array, old_R: np.array):
        euler_angles_list = []
        translation_list = []

        for _ in range(n_views):
            # apply yaw first
            ax = self.rotate_axis['yaw']            
            new_t, new_e = self.sample(ax, center_std, float(max_angle[0]), old_t, old_R)

            # apply pitch then
            ax = self.rotate_axis['pitch']            
            new_t, new_e = self.sample(ax, center_std, float(max_angle[1]), new_t, new_e)

            euler_angles_list.append(new_e)
            translation_list.append(new_t)

        return euler_angles_list, translation_list

    @staticmethod
    def sample(axis, center_std: torch.tensor, max_angle: float,  old_t: np.array, old_e: np.array, rot_only: bool =False):
        
        teta = np.random.rand()*max_angle*axis # sample random angle smaller than theta
        new_e = old_e * teta  # perturb the original pose

        old_xyz = WGS84_to_ECEF(old_t)
        perturb_c = torch.normal(0., center_std)
        new_xyz = old_xyz + np.array(perturb_c) # perturb it 
        new_t = ECEF_to_WGS84(new_xyz)

        return new_t, new_e


class RandomSamplerByAxis():
    rotate_axis = [
        [1, 0, 0], # x, pitch
        [0, 0, 1] # z, yaw
    ]
    
    def __init__(self):
        pass
    
    def sample_batch(self, n_views, center_std, max_angle, old_t, old_e):
        euler_angles_list = []
        translation_list = []

        for i in range(n_views):
            # use first axis half the time, the other for the rest
            ax_i = i // ((n_views+1) // 2)
            ax = self.rotate_axis[ax_i]
            
            new_t, new_e = self.sample(ax, center_std, float(max_angle[1]), old_t, old_e)

            euler_angles_list.append(new_e)
            translation_list.append(new_t)

        return euler_angles_list, translation_list

    @staticmethod
    def sample(axis, center_std: torch.tensor, max_angle: float,  old_t: np.array, old_e: np.array, rot_only: bool =False):
        
        teta = np.random.rand()*max_angle*axis # sample random angle smaller than theta
        new_e = old_e * teta  # perturb the original pose

        old_xyz = WGS84_to_ECEF(old_t)
        perturb_c = torch.normal(0., center_std)
        new_xyz = old_xyz + np.array(perturb_c) # perturb it 
        new_t = ECEF_to_WGS84(new_xyz)

        return new_t, new_e


class RandomSixAxisSampler():
    rotate_axis = {
        'pitch': [1, 0, 0], # x, pitch
        'roll':  [0, 1, 0], # y, roll
        'yaw':   [0, 0, 1]  # z, yaw
    }
    
    def __init__(self):
        pass
    
    def sample_batch(self, n_views, center_std, max_angle, old_t, old_e, seed_e = [1, 0, 1], seed_t = [1, 1, 1]):
                     
        euler_angles_list = []
        translation_list = []

        for _ in range(n_views):
            new_t, new_e = self.sample(seed_e, seed_t, center_std, max_angle, old_t, old_e)

            euler_angles_list.append(new_e)
            translation_list.append(new_t)
            

        return euler_angles_list, translation_list

    @staticmethod
    def sample(seed_e, seed_t, center_std: torch.tensor, max_angle: list,  old_t: np.array, old_e: np.array, rot_only: bool =False):
        
        teta = (2* np.random.rand() -1)*np.array(max_angle)*np.array(seed_e) # sample random angle smaller than theta
        new_e = old_e + teta  # perturb the original pose
        
        old_xyz = WGS84_to_ECEF(old_t)
        perturb_c = torch.normal(0., torch.tensor(np.array(center_std)*np.array(seed_t)))
        new_xyz = old_xyz + np.array(perturb_c) # perturb it 
        new_t = ECEF_to_WGS84(new_xyz)

        return new_t, new_e
class UniformSixAxisSampler():
    
    def __init__(self):
        pass
    def sample_batch(self, n_views, center_std, max_angle, old_t, old_e, seed_e = [1, 0, 1], seed_t = [1, 1, 1]):
        rotate_axis = {
        'pitch': [1, 0, 0], # x, pitch
        'roll':  [0, 1, 0], # y, roll
        'yaw':   [0, 0, 1]  # z, yaw
        } 
        trans_axis = {
        'lon': [1, 0, 0], # x, pitch
        'lat':  [0, 1, 0], # y, roll
        'alt':   [0, 0, 1]  # z, yaw
        } 
        euler_angles_list = []
        translation_list = []
        euler_seeds = []
        euler_seeds.append(old_e)
        max_angle = np.array(max_angle)
        if seed_e[0] == 1:
            e_in_pitch = old_e + np.array(max_angle)*np.array(rotate_axis['pitch'])
            euler_seeds.append(e_in_pitch)
            e_in_pitch = old_e - np.array(max_angle)*np.array(rotate_axis['pitch'])
            euler_seeds.append(e_in_pitch)  
            
        if seed_e[1] == 1:
            e_in_roll = old_e + np.array(max_angle)*np.array(rotate_axis['roll'])
            euler_seeds.append(e_in_roll)
            e_in_roll = old_e - np.array(max_angle)*np.array(rotate_axis['roll'])
            euler_seeds.append(e_in_roll)     
        if seed_e[2] == 1:
            e_in_yaw = old_e + np.array(max_angle)*np.array(rotate_axis['yaw'])
            euler_seeds.append(e_in_yaw)
            e_in_yaw = old_e - np.array(max_angle)*np.array(rotate_axis['yaw'])
            euler_seeds.append(e_in_yaw)     
        # Translation
        old_xyz = WGS84_to_ECEF(old_t)
        center_std = np.array(center_std)
        trans_seeds = []
        trans_seeds.append(old_t)
        if seed_t[0] == 1:
            new_xyz = old_xyz + np.array(center_std[0]) * np.array(trans_axis['lon'])
            new_t = ECEF_to_WGS84(new_xyz)
            trans_seeds.append(new_t)
            new_xyz = old_xyz - np.array(center_std[0]) * np.array(trans_axis['lon'])
            new_t = ECEF_to_WGS84(new_xyz)
            trans_seeds.append(new_t)
        if seed_t[1] == 1:
            new_xyz = old_xyz + np.array(center_std[1]) * np.array(trans_axis['lat'])
            new_t = ECEF_to_WGS84(new_xyz)
            trans_seeds.append(new_t)
            new_xyz = old_xyz - np.array(center_std[1]) * np.array(trans_axis['lat'])
            new_t = ECEF_to_WGS84(new_xyz)
            trans_seeds.append(new_t)
        if seed_t[2] == 1:
            new_xyz = old_xyz + np.array(center_std[2]) * np.array(trans_axis['alt'])
            new_t = ECEF_to_WGS84(new_xyz)
            trans_seeds.append(new_t)
            new_xyz = old_xyz - np.array(center_std[2]) * np.array(trans_axis['alt'])
            new_t = ECEF_to_WGS84(new_xyz)
            trans_seeds.append(new_t)
        for i in range(len(trans_seeds)):
            for j in range(len(euler_seeds)):
                euler_angles_list.append(euler_seeds[j])
                translation_list.append(trans_seeds[i])
        
        return euler_angles_list, translation_list

    @staticmethod
    def sample(seed_e, seed_t, center_std: torch.tensor, max_angle: list,  old_t: np.array, old_e: np.array, rot_only: bool =False):
        
        teta = np.array(max_angle)*np.array(seed_e) # sample random angle smaller than theta
        new_e = old_e + teta  # perturb the original pose
        
        old_xyz = WGS84_to_ECEF(old_t)
        perturb_c = torch.normal(0., torch.tensor(np.array(center_std)*np.array(seed_t)))
        new_xyz = old_xyz + np.array(perturb_c) # perturb it 
        new_t = ECEF_to_WGS84(new_xyz)

        return new_t, new_e

class ConstantScaler():
    def __init__(self, conf):
        self.max_angle = conf['max_angle']
        self.center_std = conf['center_std']

    def step(self, i):
        pass
    
    def get_noise(self):
        return self.center_std, self.max_angle
    
    def get_max_noise(self, multiplier=1):
        return self.center_std*multiplier, self.max_angle*multiplier

    
class UniformScaler():
    def __init__(self, conf):
        # gamma is the minimum multiplier that will be applied
        self.max_angle = conf.max_angle
        self.max_center_std = conf.max_center_std
        self.current_angle = conf.max_angle
        self.current_center_std = conf.max_center_std
        
        self.n_steps = conf.N_steps
        self.gamma = conf.gamma
        
    def step(self, i):
        scale_noise = max(self.gamma, (self.n_steps - i)/self.n_steps)
        
        self.current_center_std = self.max_center_std*scale_noise 
        self.current_angle = self.max_angle* scale_noise        

    def get_noise(self):
        return self.current_center_std, self.current_angle
    
    def get_max_noise(self, multiplier=1):
        return self.max_center_std*multiplier, self.max_angle*multiplier
