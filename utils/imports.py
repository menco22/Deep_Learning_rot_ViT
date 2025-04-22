import math
import time
import random
import warnings
import io
import wandb

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
# from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    MultiStepLR,
    CosineAnnealingLR,
    LambdaLR,
    SequentialLR,
)

import torchvision
from torchvision import transforms, datasets, models
from torchvision.models import resnet18
from torchvision.transforms import ToPILImage
from datasets import load_dataset

from tqdm import tqdm
from PIL import Image
from huggingface_hub import login