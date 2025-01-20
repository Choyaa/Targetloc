import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.flow_estimator.update import BasicUpdateBlock, SmallUpdateBlock
from utils.flow_estimator.extractor import BasicEncoder, SmallEncoder
from utils.flow_estimator.corr import CorrBlock, AlternateCorrBlock
from utils.flow_estimator.utils.utils import bilinear_sampler, coords_grid, upflow8
import cv2
import argparse
from PIL import Image
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions
class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
def load_image(imfile, res_mul):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    # img = cv2.imread(imfile)
    # h, w,_ = img.shape
    # if res_mul != 1:
    #     img = cv2.resize(img, (int(w/2), int(h/2)))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].cuda()
def main(model, imfile1, imfile2, points, res_mul, flow = None, vis = False):
    image1 = load_image(imfile1,res_mul)
    image2 = load_image(imfile2,res_mul)
    if res_mul != 1:
        image1=F.interpolate(image1,scale_factor=1/res_mul)
        image2=F.interpolate(image2,scale_factor=1/res_mul)

    padder = InputPadder(image1.shape)
    image1_pad, image2_pad = padder.pad(image1, image2)

    flow_low, flow_up = model(image1_pad, image2_pad, iters=20, test_mode=True)
    flow_up = flow_up.cpu().detach().numpy().reshape(2,flow_up.shape[2], flow_up.shape[3])
    points = points.astype(int)
    x, y = points[:, 0], points[:, 1]
    if flow is None:
        flow = flow_up
    else:
        flow += flow_up
    temp = flow[:, y, x].T
    fx, fy = temp[:,0], temp[:,1]
    
    # 创建线的终点
    ex = x + fx
    ey = y + fy
    point2d_relative = np.column_stack((ex, ey))
    if vis:
        img = cv2.imread(imfile2)
        h, w, _ = img.shape
        img = cv2.resize(img, (int(w/2), int(h/2)))
        for (x1, y1) in point2d_relative:
            cv2.circle(img, (int(x1), int(y1)), 3, (0, 255, 0), -1)  # 绘制线条
        name = "/home/ubuntu/Documents/code/github/Target2loc/datasets/翡翠湾/flow_vis/" +imfile2.split('/')[-1]
        cv2.imwrite(name, img)
    
    return point2d_relative, flow
    
def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load('/home/ubuntu/Documents/code/github/Target2loc/fast_render2loc/weights/raft-things.pth'))

    model = model.module
    model.cuda()
    model.eval()
    return model

