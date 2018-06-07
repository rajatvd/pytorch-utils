

"""
Concenvience module for pytorch training and visualization.
"""

import torch
import torch.nn as nn
from torch import optim
from pylab import *
from torch.utils.data import *
from IPython import display
import os

style.use(['dark_background'])
rcParams['axes.grid']=True