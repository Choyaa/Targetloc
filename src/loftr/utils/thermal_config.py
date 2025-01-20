from yacs.config import CfgNode as CN


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


_CN = CN()
_CN.BACKBONE_TYPE = 'RepVGG'
_CN.ALIGN_CORNER = False
_CN.RESOLUTION = (8, 1)
_CN.FINE_WINDOW_SIZE = 8  # window_size in fine_level, must be even
_CN.MP = False
_CN.REPLACE_NAN = True
_CN.HALF = False

# 1. LoFTR-backbone (local feature CNN) config
_CN.BACKBONE = CN()
_CN.BACKBONE.BLOCK_DIMS = [64, 128, 256]  # s1, s2, s3

# 2. LoFTR-coarse module config
_CN.COARSE = CN()
_CN.COARSE.D_MODEL = 256
_CN.COARSE.D_FFN = 256
_CN.COARSE.NHEAD = 8
_CN.COARSE.LAYER_NAMES = ['self', 'cross'] * 4
_CN.COARSE.AGG_SIZE0 = 4
_CN.COARSE.AGG_SIZE1 = 4
_CN.COARSE.NO_FLASH = False
_CN.COARSE.ROPE = True
_CN.COARSE.NPE = [832, 832, 832, 832] # [832, 832, long_side, long_side] Suggest setting based on the long side of the input image, especially when the long_side > 832

# 3. Coarse-Matching config
_CN.MATCH_COARSE = CN()
_CN.MATCH_COARSE.THR = 0.2 # recommend 0.2 for full model and 25 for optimized model
_CN.MATCH_COARSE.BORDER_RM = 2
_CN.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.MATCH_COARSE.SKIP_SOFTMAX = False # False for full model and True for optimized model
_CN.MATCH_COARSE.FP16MATMUL = False # False for full model and True for optimized model
_CN.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2  # training tricks: save GPU memory
_CN.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock

# 4. Fine-Matching config
_CN.MATCH_FINE = CN()
_CN.MATCH_FINE.LOCAL_REGRESS_TEMPERATURE = 10.0 # use 10.0 as fine local regress temperature, not 1.0
_CN.MATCH_FINE.LOCAL_REGRESS_SLICEDIM = 8


_CN.LOFTR = CN()
_CN.LOFTR.BACKBONE = CN()
_CN.LOFTR.COARSE = CN()
_CN.LOFTR.MATCH_COARSE = CN()
_CN.LOFTR.MATCH_FINE = CN()
_CN.LOFTR.LOSS = CN()
_CN.DATASET = CN()
_CN.TRAINER = CN()
_CN.LOFTR.FINE = CN()
_CN.LOFTR.RESNETFPN = CN()
_CN.LOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'
_CN.LOFTR.MATCH_COARSE.SPARSE_SPVS = False

_CN.TRAINER.CANONICAL_LR = 8e-3
_CN.TRAINER.WARMUP_STEP = 1875  # 3 epochs
_CN.TRAINER.WARMUP_RATIO = 0.1
# _CN.TRAINER.MSLR_MILESTONES = [8, 12, 16, 20, 24]

_CN.TRAINER.MSLR_MILESTONES = [4, 6, 8, 10, 12, 14, 16]

# pose estimation
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5

_CN.TRAINER.OPTIMIZER = "adamw"
_CN.TRAINER.ADAMW_DECAY = 0.1

_CN.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.1

_CN.LOFTR.MATCH_COARSE.MTD_SPVS = True


_CN.LOFTR.FINE.MTD_SPVS = True

_CN.LOFTR.RESOLUTION = (8, 1)  # options: [(8, 2), (16, 4)]
_CN.LOFTR.FINE_WINDOW_SIZE = 8  # window_size in fine_level, must be odd
_CN.LOFTR.MATCH_FINE.THR = 0
# _CN.LOFTR.MATCH_FINE.TOPK = 3
_CN.LOFTR.LOSS.FINE_TYPE = 'l2'  # ['l2_with_std', 'l2']

_CN.TRAINER.EPI_ERR_THR = 5e-4 # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)

_CN.LOFTR.MATCH_COARSE.SPARSE_SPVS = True

# PAN
_CN.LOFTR.COARSE.PAN = True
_CN.LOFTR.COARSE.POOl_SIZE = 4
_CN.LOFTR.COARSE.BN = False
_CN.LOFTR.COARSE.XFORMER = True
_CN.LOFTR.COARSE.ATTENTION = 'full'  # options: ['linear', 'full']

_CN.LOFTR.FINE.PAN = False
_CN.LOFTR.FINE.POOl_SIZE = 4
_CN.LOFTR.FINE.BN = False
_CN.LOFTR.FINE.XFORMER = False

# _CN.LOFTR.COARSE.ATTENTION = 'linear'  # options: ['linear', 'full']

# noalign
_CN.LOFTR.ALIGN_CORNER = False

# fp16
_CN.DATASET.FP16 = False
_CN.LOFTR.FP16 = False

# DEBUG
_CN.LOFTR.FP16LOG = False
_CN.LOFTR.MATCH_COARSE.FP16LOG = False

# fine skip
_CN.LOFTR.FINE.SKIP = True

# clip
_CN.TRAINER.GRADIENT_CLIPPING = 0.5

# backbone
_CN.LOFTR.BACKBONE_TYPE = 'RepVGG'

# d
# A0
# _CN.LOFTR.RESNETFPN.INITIAL_DIM = 48
# _CN.LOFTR.RESNETFPN.BLOCK_DIMS = [48, 96, 192]  # s1, s2, s3
# _CN.LOFTR.COARSE.D_MODEL = 192
# _CN.LOFTR.FINE.D_MODEL = 48

# A1
_CN.LOFTR.RESNETFPN.INITIAL_DIM = 64
_CN.LOFTR.RESNETFPN.BLOCK_DIMS = [64, 128, 256]  # s1, s2, s3
_CN.LOFTR.COARSE.D_MODEL = 256
_CN.LOFTR.FINE.D_MODEL = 64

# FPN backbone_inter_feat with coarse_attn.
_CN.LOFTR.COARSE_FEAT_ONLY = True
_CN.LOFTR.INTER_FEAT = True
_CN.LOFTR.RESNETFPN.COARSE_FEAT_ONLY = True
_CN.LOFTR.RESNETFPN.INTER_FEAT = True

# loop back spv coarse match
_CN.LOFTR.FORCE_LOOP_BACK = False

# fix norm fine match
_CN.LOFTR.MATCH_FINE.NORMFINEM = True

# loss cf weight
_CN.LOFTR.LOSS.COARSE_OVERLAP_WEIGHT = True
_CN.LOFTR.LOSS.FINE_OVERLAP_WEIGHT = True

# leaky relu
_CN.LOFTR.RESNETFPN.LEAKY = False
_CN.LOFTR.COARSE.LEAKY = 0.01

# prevent FP16 OVERFLOW in dirty data
_CN.LOFTR.NORM_FPNFEAT = True
_CN.LOFTR.REPLACE_NAN = True

# force mutual nearest
_CN.LOFTR.MATCH_COARSE.FORCE_NEAREST = True
_CN.LOFTR.MATCH_COARSE.THR = 0.1

# fix fine matching
_CN.LOFTR.MATCH_FINE.FIX_FINE_MATCHING = True

# dwconv
_CN.LOFTR.COARSE.DWCONV = True

# localreg
_CN.LOFTR.MATCH_FINE.LOCAL_REGRESS = True
_CN.LOFTR.LOSS.LOCAL_WEIGHT = 0.25

# it5
_CN.LOFTR.EVAL_TIMES = 1

# rope
_CN.LOFTR.COARSE.ROPE = True

# local regress temperature
_CN.LOFTR.MATCH_FINE.LOCAL_REGRESS_TEMPERATURE = 10.0

# SLICE
_CN.LOFTR.MATCH_FINE.LOCAL_REGRESS_SLICE = True
_CN.LOFTR.MATCH_FINE.LOCAL_REGRESS_SLICEDIM = 8

# inner with no mask [64,100]
_CN.LOFTR.MATCH_FINE.LOCAL_REGRESS_INNER = True
_CN.LOFTR.MATCH_FINE.LOCAL_REGRESS_NOMASK = True

_CN.LOFTR.MATCH_FINE.TOPK = 1
_CN.LOFTR.MATCH_COARSE.FINE_TOPK = 1

_CN.LOFTR.MATCH_COARSE.FP16MATMUL = False


thermal_default_cfg = lower_config(_CN)
