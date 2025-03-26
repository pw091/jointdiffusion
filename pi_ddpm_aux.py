import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
import time
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR
import math
import functools
from scipy.integrate import solve_ivp
import random
import torch.autograd as autograd
import scipy as sp
import matplotlib
import copy
import os
import pickle
from matplotlib.patches import Ellipse
import itertools as it
import warnings

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(DEVICE)
torch.set_default_dtype(torch.float32)