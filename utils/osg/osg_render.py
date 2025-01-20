'''
Descripttion: C++ Practice
Author: TomHeaven
Date: 2024-06-26 08:52:32
'''
from pathlib import Path
import numpy as np
import cv2
import math
from ModelRenderScene import ModelRenderScene


class RenderImageProcessor:
    def __init__(self, config, eglDpy=0):
        self.config = config
        self.osg_config = self.config["render2loc"]["osg"]
        self.renderer = self._initialize_renderer(eglDpy)
        self._delay(self.osg_config)
        print("DEBUG: RenderImageProcessor after _delay")

    def _initialize_renderer(self, eglDpy):
        # Construct paths for model
        model_path = self.osg_config["model_path"]
        render_camera = self.config['render2loc']['render_camera']
        view_width, view_height = int(render_camera[0]), int(render_camera[1])
        self.fovy, self.aspectRatio = 55, 1.33333
        print("DEBUG: before ModelRenderScene")
        return ModelRenderScene(model_path, view_width, view_height, self.fovy, self.aspectRatio)
    
    def _delay(self, config):
        initTrans = config["init_trans"]
        initRot = config["init_rot"]

        for i in range(100):
            self.update_pose(initTrans, initRot)
    def fovy_calculate(self, ref_camera):
        _,_,_,_,_, sensor_height_mm, f_mm = ref_camera
        fovy_radian = 2* math.atan(sensor_height_mm / 2 / f_mm)
        fovy_degree = math.degrees(fovy_radian)

        return fovy_degree
    def update_pose(self, Trans, Rot, ref_camera = None):
        # TODO: Fix [osgearth warning] FAILED to create a terrain engine for this map
        if ref_camera is not None:
            self.fovy = self.fovy_calculate(ref_camera)
        self.renderer.updateViewPoint(Trans, Rot)
        self.renderer.nextFrame(self.fovy)
    
    def get_color_image(self):
        colorImgMat = np.array(self.renderer.getColorImage(), copy=False)
        colorImgMat = cv2.flip(colorImgMat, 0)
        colorImgMat = cv2.cvtColor(colorImgMat, cv2.COLOR_RGB2BGR)
        
        return colorImgMat
    
    def get_depth_image(self):
        depthImgMat = np.array(self.renderer.getDepthImage(), copy=False).squeeze()
        
        return depthImgMat
    
    def save_color_image(self, outputs):
        self.renderer.saveColorImage(outputs)

    def get_EGLDisplay(self):
        return self.renderer.getEGLDisplay()
    
    
    
    
        
