import torch.nn as nn


SHIP_LENGTH_MEAN = 90  # from data anlaysis. The mean length of a ship is 90m
SHIP_WIDTH_MEAN = 30  # from data anlaysis. The mean width of a ship is 30m

BACKBONE_CPS_BLOCKS = 6
N_SCALES = 3  # number of scales in the feature pyramid.

SHIP_MAX_WIDTH = 130  # what is the widest ship allowed. Larger than this is "anomaly"
SHIP_MAX_LENGTH = 460  # what is the longest ship allowed. Larger than this is "anomaly"

SHIP_WIDTH_BIN = 2  # how big is each bin for width
SHIP_LENGTH_BIN = 2  # how big is each bin for length


TARGET_REGRESSION_INDEX = [0, 1, 2, 3]
TARGET_IOU_INDEX = [4]
TARGET_CLASS_INDEX = [5, 6]
TARGET_SIZE_INDEX = [7, 8, 9, 10]

NR_GAUSSIANS_SOG = 2
TARGET_SOG_INDEX = list(range(TARGET_SIZE_INDEX[0] + 1, TARGET_SIZE_INDEX[0] + 1 + 2 * NR_GAUSSIANS_SOG))
TARGET_COG_INDEX = list(range(TARGET_SIZE_INDEX[0] + 1 + 2 * NR_GAUSSIANS_SOG, TARGET_SIZE_INDEX[0] + 1 + 2 * NR_GAUSSIANS_SOG + NR_GAUSSIANS_SOG))


#### model
DEFAULT_ACTIVATION = nn.SiLU()  # try ReLU, LeakyReLU, SiLU, GeLU()
