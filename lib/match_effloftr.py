from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter, thermal_default_cfg
from copy import deepcopy
import torch
import cv2
import os
import numpy as np
from .plotting import make_matching_figure
import matplotlib.cm as cm
from tqdm import tqdm
import math
from pathlib import Path



class ImageMatcher:
    def __init__(self):
        self.matcher = self.init()

    def show_match(self, img0_raw, img1_raw, mkpts0, mkpts1, mconf, save_path):
        color = cm.jet(mconf.cpu().numpy())
        text = [
            'LoFTR',
            'Matches: {}'.format(len(mkpts0)),
        ]
        fig = make_matching_figure(img0_raw, img1_raw, mkpts0.cpu().numpy(), mkpts1.cpu().numpy(), color, text=text, path=save_path)

    def init(self):
        # model_type = 'thermal'  # 'full' for best quality, 'opt' for best efficiency， ‘thermal'
        model_type = 'full'
        precision = 'fp32'  # 'fp16' for best efficiency

        if model_type == 'full':
            _default_cfg = deepcopy(full_default_cfg)
        elif model_type == 'opt':
            _default_cfg = deepcopy(opt_default_cfg)
        elif model_type == 'thermal':
            _default_cfg = deepcopy(thermal_default_cfg)
        
        if precision == 'mp':
            _default_cfg['mp'] = True
        elif precision == 'fp16':
            _default_cfg['half'] = True
        
        matcher = LoFTR(config=_default_cfg)
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        # matcher.load_state_dict(torch.load(cur_dir + "/../weights/thermal_ds.ckpt")['state_dict'])
        matcher.load_state_dict(torch.load(cur_dir + "/../weights/eloftr_outdoor.ckpt")['state_dict'])
        matcher = reparameter(matcher)  # no reparameterization will lead to low performance

        if precision == 'fp16':
            matcher = matcher.half()

        matcher = matcher.eval().cuda()
        return matcher
    
    def calculate_padded_size(self, dim, multiple):
            return ((dim // multiple) + (1 if dim % multiple else 0)) * multiple
    
    def match_single_pair(self, image_id, query_img, ref_img, save_loc_path= None):
      
        # padding
        # 原始尺寸
        original_height, original_width = query_img.shape[:2]
        # 补齐倍数
        multiple = 32
        # 计算补齐后的宽度和高度
        padded_width = self.calculate_padded_size(original_width, multiple)
        padded_height = self.calculate_padded_size(original_height, multiple)

        # 创建新的空白图像，这里使用黑色背景
        if len(query_img.shape) == 3 and query_img.shape[2] == 3:
            query_img = cv2.cvtColor(query_img.copy(), cv2.COLOR_BGR2GRAY)
        
        img0_raw = np.zeros((padded_height, padded_width), dtype=np.uint8)
        img0_raw[0:original_height, 0:original_width] = query_img
        
        
        img1_raw_ = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)  # render image MAT
        original_height, original_width = img1_raw_.shape[:2]
        # 计算补齐后的宽度和高度
        padded_width = self.calculate_padded_size(original_width, multiple)
        padded_height = self.calculate_padded_size(original_height, multiple)

        # 创建新的空白图像，这里使用黑色背景
        img1_raw = np.zeros((padded_height, padded_width), dtype=np.uint8)
        img1_raw[0:original_height, 0:original_width] = img1_raw_
        # scale
        maximum_size = 800
        scale0 = math.ceil(img0_raw.shape[1] / maximum_size)
        scale1 = math.ceil(img1_raw.shape[1] / maximum_size)
        
        
        img0_raw = cv2.resize(img0_raw, (int(img0_raw.shape[1]/scale0), int(img0_raw.shape[0]/scale0)))
        img1_raw = cv2.resize(img1_raw, (int(img1_raw.shape[1]/scale1), int(img1_raw.shape[0]/scale1)))    
        img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//32*32, img0_raw.shape[0]//32*32))  # input size shuold be divisible by 32
        img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//32*32, img1_raw.shape[0]//32*32))
        img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
        img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
        batch = {'image0': img0, 'image1': img1}
         # Inference with EfficientLoFTR and get prediction
        with torch.no_grad():
            self.matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()
            F, mask = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.FM_RANSAC,3, 0.99)
                
            # Filter matches based on the mask
            index = np.where(mask == 1)[0]
            mconf = torch.tensor(mconf[index]).float()
            mkpts0q = torch.tensor(mkpts0[index,:]).float()
            mkpts1r = torch.tensor(mkpts1[index,:]).float()
            # 显示匹配结果
            if len(mconf) > 0 and save_loc_path is not None:
                if not os.path.exists(save_loc_path / 'matches'):
                    os.makedirs(save_loc_path / 'matches/')
                match_vis_path = save_loc_path / 'matches' / (str(image_id) + '.png')
                # logger.info("save matching image!")
                self.show_match(img0_raw, img1_raw, mkpts0q, mkpts1r, mconf, match_vis_path)  
            mkpts0q = mkpts0q * scale0
            mkpts1r = mkpts1r * scale1   

        return [mkpts0q, mkpts1r]



