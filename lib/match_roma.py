from copy import deepcopy
import torch
import cv2
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
from .plotting import make_matching_figure
import matplotlib.cm as cm
from tqdm import tqdm
import math
from pathlib import Path
from src.romatch import tiny_roma_v1_outdoor, roma_outdoor
from PIL import Image

class ImageMatcher:
    def __init__(self):
        self.matcher = self.init()
        
    def show_match(self, img0_raw, img1_raw, mkpts0, mkpts1, save_path):
        # 将关键点转为cv2的KeyPoint对象
        keypoints1 = [cv2.KeyPoint(x[0], x[1], 1) for x in mkpts0.cpu().numpy()]
        keypoints2 = [cv2.KeyPoint(x[0], x[1], 1) for x in mkpts1.cpu().numpy()]
        
        # 构造DMatch对象
        matches = [cv2.DMatch(i, i, 0) for i in range(len(keypoints1))]
        
        # 画出匹配连线
        match_img = cv2.drawMatches(img0_raw, keypoints1, img1_raw, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # 保存图像
        cv2.imwrite(f'{save_path}', match_img)

    def init(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        roma_model = roma_outdoor(device=device)
        return roma_model
    def calculate_padded_size(self, dim, multiple):
            return ((dim // multiple) + (1 if dim % multiple else 0)) * multiple
    def match_single_pair(self, image_id, query_img, ref_img, save_loc_path= None):
        
        # ref_img = np.expand_dims(ref_img, axis=0)
        # query_img = np.expand_dims(query_img, axis=0)
        img1 = Image.fromarray(cv2.cvtColor(query_img,cv2.COLOR_BGR2RGB))
        img2 = Image.fromarray(cv2.cvtColor(ref_img,cv2.COLOR_BGR2RGB))
        W_A, H_A = img1.size
        W_B, H_B = img2.size

        warp, certainty1 = self.matcher.match(img1, img2)
        matches, _ = self.matcher.sample(warp, certainty1)
        mkpts0q, mkpts1r = self.matcher.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B) 
        F, mask = cv2.findFundamentalMat(
        mkpts0q.cpu().numpy(), mkpts1r.cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
        )
        index = np.where(mask == 1)
        mkpts0q = mkpts0q[index[0]]
        mkpts1r = mkpts1r[index[0]]
        if len(matches) > 0 and save_loc_path is not None:
            if not os.path.exists(save_loc_path / 'matches'):
                os.makedirs(save_loc_path / 'matches/')
            match_vis_path = save_loc_path / 'matches' / (str(image_id) + '.png')
            # print("save matching image!")
            self.show_match(query_img, ref_img, mkpts0q, mkpts1r, match_vis_path)   

        return [mkpts0q, mkpts1r]



