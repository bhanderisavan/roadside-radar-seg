"""This file provides default configuration parameters for the project.
"""

from yacs.config import CfgNode as CN
from roadside_radar_seg.definitions import PROJECT_ROOT
from pathlib import Path
import torch

_C = CN()

######################################################################
############################### OUTPUT ###############################
######################################################################
_C.OUTPUT_DIR = str(Path(PROJECT_ROOT).joinpath("results"))

######################################################################
########################### REALTIME DEMO ############################
######################################################################
_C.ROS_DATA_READER = CN()
_C.ROS_DATA_READER.USE_BGSUB = True
_C.ROS_DATA_READER.BGSUB_GRID_PATH = ""
_C.ROS_DATA_READER.MAX_X_THRESHOLD = 100  # meters
_C.ROS_DATA_READER.MAX_ABS_V_THRESHOLD = 25  # m/S

######################################################################
############################### BG SUB ###############################
######################################################################
_C.BGSUB = CN()
_C.BGSUB.MAXIMUM_VALID_RANGE = 80  # metres
_C.BGSUB.MAXIMUM_VALID_AZIMUTH_ANGLE = 120  # degree
_C.BGSUB.MAXIMUM_VALID_ELEVATION_ANGLE = 50  # degree
# these are ARS 548 radar sensor specific parameters. Please do not change.
_C.BGSUB.RANGE_CELL_SIZE = 0.22  # metre
_C.BGSUB.AZIMUTH_CELL_SIZE = 1.2  # degree
_C.BGSUB.ELEVATION_CELL_SIZE = 2.3  # degree
# path where bg subtraction grid files are stored. only needed in training.
_C.BGSUB.GRID_FOLDER_PATH = str(
    Path(PROJECT_ROOT).joinpath("resources", "bg_sub_grids")
)

######################################################################
############################### INPUT ################################
######################################################################
_C.INPUT = CN()
_C.INPUT.MAX_ABS_VELOCITY = 25  # m/s
# points with abs velocity smaller than this threshold are considered static.
_C.INPUT.DYNAMIC_V_THRESHOLD = 0.1  # m/s
_C.INPUT.MAX_DEPTH = 120  # meters
# input fields to the network. the index field must be present in the list.
_C.INPUT.INPUT_FIELDS = ["index", "x", "y", "z", "v_x", "v_y", "rcs"]
# input feature normalization.
_C.INPUT.NORMALIZATION_METHOD = "minmax"
# per feature minimum and max values - calculated from training set.
_C.INPUT.FEATURE_WISE_MIN_MAX = CN(
    {
        "x": {"min": 0, "max": _C.INPUT.MAX_DEPTH},
        "y": {"min": -79.79488372802734, "max": 75.5746841430664},
        "z": {"min": -26.840476989746094, "max": 15.620646476745605},
        "rcs": {"min": -20.0, "max": 35.0},
        "v_x": {"min": -_C.INPUT.MAX_ABS_VELOCITY, "max": _C.INPUT.MAX_ABS_VELOCITY},
        "v_y": {"min": -_C.INPUT.MAX_ABS_VELOCITY, "max": _C.INPUT.MAX_ABS_VELOCITY},
        "range_rate": {
            "min": -_C.INPUT.MAX_ABS_VELOCITY,
            "max": _C.INPUT.MAX_ABS_VELOCITY,
        },
    }
)
# 2 means that the input tensor is of shape [batch, num_points, num_features], 1 means [batch, features, points]
_C.INPUT.FEATURE_DIM = 2
# number of points in each frame is different, so we need to padd the input in order to create a batch tensor out of it.
_C.INPUT.INPUT_PADDING_VALUE = -1

######################################################################
############################# DATASET #############################
######################################################################
# TRAIN DATASET PARAMETERS
_C.TRAIN = CN()
_C.TRAIN.DATASET_NAME = "RoadsideRadar"
_C.TRAIN.USE_BGSUB = True
# PATH TO TRAIN FOLDER. (i.e dataset_root/train)
_C.TRAIN.DATASET_PATH = ""

# VAL DATASET PARAMETERS
_C.VAL = CN()
_C.VAL.DATASET_NAME = "RoadsideRadar"
_C.VAL.USE_BGSUB = True
# PATH TO VAL FOLDER. (i.e dataset_root/val)
_C.VAL.DATASET_PATH = ""

######################################################################
############################# DATALOADER #############################
######################################################################
_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 64
_C.DATALOADER.NUM_WORKERS = 32
_C.DATALOADER.SHUFFLE = True
_C.DATALOADER.DROP_LAST = True
_C.DATALOADER.PIN_MEMORY = True

######################################################################
############################### MODEL ################################
######################################################################
# FOR ALL THE SUBMODULES , THE FOLLOWING APPLIES.
# "" (empty string) implies that no activation/norm is applied.
# ACTIVATIONS_LIST choices - ["relu", "leaky_relu", "softmax", "sigmoid", ""]
# NORMALIZATIONS_LIST choices - ["batchnorm", "layernorm", ""]
# LAYER_TYPES_LIST choices - ["linear", "conv"]

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# when running inference of resuming the training.
_C.MODEL.WEIGHTS = ""
# base model name.
_C.MODEL.META_ARCHITECTURE = "RadarDetector"

######################################################################
######################### INPUT PROCESSING ###########################
######################################################################
_C.MODEL.INPUT_PROCESSING = CN()
_C.MODEL.INPUT_PROCESSING.NAME = "MLPInputEmbeddings"
# the first entry should be len(input_fields_without_index)
_C.MODEL.INPUT_PROCESSING.NN_LAYERS_LIST = [len(_C.INPUT.INPUT_FIELDS) - 1, 32, 64]
_C.MODEL.INPUT_PROCESSING.ACTIVATIONS_LIST = ["leaky_relu", "leaky_relu"]
_C.MODEL.INPUT_PROCESSING.NORMALIZATIONS_LIST = ["layernorm", "layernorm"]
_C.MODEL.INPUT_PROCESSING.LAYER_TYPES_LIST = ["linear", "linear"]

######################################################################
############################## BACKNONE ##############################
######################################################################
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "MLPBackbone"
_C.MODEL.BACKBONE.NN_LAYERS_LIST = [64, 128, 512]
_C.MODEL.BACKBONE.ACTIVATIONS_LIST = ["leaky_relu", "leaky_relu"]
_C.MODEL.BACKBONE.NORMALIZATIONS_LIST = ["layernorm", "layernorm"]
_C.MODEL.BACKBONE.LAYER_TYPES_LIST = ["linear", "linear"]


######################################################################
############################ SEGMENTATION ############################
######################################################################
_C.MODEL.SEGM_HEAD = CN()
_C.MODEL.SEGM_HEAD.NAME = "RadarPointSegmentationHead"
# NUM_CLASSES is including background class.
_C.MODEL.SEGM_HEAD.NUM_CLASSES = 6
# the first value must be _C.MODEL.INPUT_PROCESSING.NN_LAYERS_LIST[-1] + _C.MODEL.BACKBONE.NN_LAYERS_LIST[-1]
# this implies that the segmentation is performed on local + global concatenated features.
# the last values must be number of classes.
_C.MODEL.SEGM_HEAD.NN_LAYERS_LIST = [
    _C.MODEL.INPUT_PROCESSING.NN_LAYERS_LIST[-1] + _C.MODEL.BACKBONE.NN_LAYERS_LIST[-1],
    256,
    128,
    _C.MODEL.SEGM_HEAD.NUM_CLASSES,
]
_C.MODEL.SEGM_HEAD.ACTIVATIONS_LIST = ["leaky_relu", "leaky_relu", ""]
_C.MODEL.SEGM_HEAD.NORMALIZATIONS_LIST = ["layernorm", "layernorm", ""]
_C.MODEL.SEGM_HEAD.LAYER_TYPES_LIST = ["linear", "linear", "linear"]
_C.MODEL.SEGM_HEAD.DROPOUT = 0.5
# only choice is cross entropy. Maybe will add dice loss in future.
_C.MODEL.SEGM_HEAD.LOSS = "cross_entropy"
# class wise loss weights for imbalance in the training dataset.
# generated from training data.
_C.MODEL.SEGM_HEAD.CLASS_WISE_LOSS_WEIGHTS = [
    0.1722706771,
    19.8219533546,
    20.0077313894,
    214.3424611224,
    25.2468840839,
    19.8108048994,
]

######################################################################
######################### INSTANCE FORMATION #########################
######################################################################
_C.MODEL.INSTANCE_HEAD = CN()
# we have two choices here for the input to the instance formation head.
# 1. input embeddings (output of input processing module)
# 2. local + global concatenated features.
# default is 2. _C.MODEL.INPUT_PROCESSING.NN_LAYERS_LIST[-1] + _C.MODEL.BACKBONE.NN_LAYERS_LIST[-1]
# this is performing feature reduction before passing the features to attention head.
_C.MODEL.INSTANCE_HEAD.NN_LAYERS_LIST = [
    _C.MODEL.INPUT_PROCESSING.NN_LAYERS_LIST[-1] + _C.MODEL.BACKBONE.NN_LAYERS_LIST[-1],
    256,
    128,
]
_C.MODEL.INSTANCE_HEAD.ACTIVATIONS_LIST = ["leaky_relu", "leaky_relu"]
_C.MODEL.INSTANCE_HEAD.NORMALIZATIONS_LIST = ["layernorm", "layernorm"]
_C.MODEL.INSTANCE_HEAD.LAYER_TYPES_LIST = ["linear", "linear"]
# choices - ["binary_cross_entropy", "focal_loss"]
_C.MODEL.INSTANCE_HEAD.LOSS = "binary_cross_entropy"
# it is often a nice idea to add ground truth to the network input to ensure smooth training.
# this means that we change the class of network segmention output of gt points before passing it to the instance head.
# this is done only if the network prediction has score < _C.MODEL.INSTANCE_HEAD.ADD_GT_POINTS_THRESHOLD
_C.MODEL.INSTANCE_HEAD.ADD_GT_POINTS = True
_C.MODEL.INSTANCE_HEAD.ADD_GT_POINTS_THRESHOLD = 0.7

# only the points with cofidence score > INPUT_CONFIDENCE_THRESH will be sent to instance formation step.
_C.MODEL.INSTANCE_HEAD.INPUT_CONFIDENCE_THRESH_TRAIN = 0.5
_C.MODEL.INSTANCE_HEAD.INPUT_CONFIDENCE_THRESH_TEST = 0.7

# MODEL INSTANCE HEAD ATTENTION
_C.MODEL.INSTANCE_HEAD.ATTENTION = CN()
# if True, a separate learnable linear layers is used for calculation of values, otherwise input is used as values.
_C.MODEL.INSTANCE_HEAD.ATTENTION.EMBED_VALUES = False
# choices - ["additive", "multiplicative"]
_C.MODEL.INSTANCE_HEAD.ATTENTION.SIMILARITY_TYPE = "multiplicative"
# binarization threshold.
_C.MODEL.INSTANCE_HEAD.ATTENTION.SIMILARTY_THRESHOLD = 0.5

######################################################################
############################### SOLVER ###############################
######################################################################
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCHS = 500

# lr scheduler
_C.SOLVER.LR_SCHEDULER_NAME = "MultiStepLR"
_C.SOLVER.LR_SCHEDULER_MILESTONES = [100, 300]
_C.SOLVER.LR_SCHEDULER_GAMMA = 0.1
# for ReduceLROnPlateau
_C.SOLVER.LR_SCHEDULER_PATIENCE = 5

# optimizer
_C.SOLVER.OPTIMIZER = CN()
_C.SOLVER.OPTIMIZER.NAME = "Adam"  # name must match the torch.optim class names
_C.SOLVER.BASE_LR = 3e-4
# The end lr, only used by WarmupCosineLR
_C.SOLVER.BASE_LR_END = 0.0
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.NESTEROV = False
_C.SOLVER.GAMMA = 0.1

# weight decay
_C.SOLVER.WEIGHT_DECAY = 0.001

# Save a checkpoint after every this number of iterations
_C.SOLVER.CHECKPOINT_PERIOD = 10

# gradient norm clipping for preventing exploding gradients.
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": True})
_C.SOLVER.CLIP_GRADIENTS_NORM_VALUE = 2.0

# loss weights
_C.SOLVER.SIM_LOSS_WEIGHT = 1
_C.SOLVER.SEGM_LOSS_WEIGHT = 1

######################################################################
############################## EVALUATOR #############################
######################################################################
_C.EVALUATION = CN()
_C.EVALUATION.IOU_THRESHOLDS = [0.5]
_C.EVALUATION.REC_THRESHOLDS = []
_C.EVALUATION.CLASS_METRICS = True
_C.EVALUATION.EXTENDED_SUMMARY = False
# for class agnostic map (i.e DBSCAN WITHOUT CLASS) set this to micro.
_C.EVALUATION.AVERAGE = "macro"
_C.EVALUATION.MAX_DETECTION_THRESHOLD = [100]
