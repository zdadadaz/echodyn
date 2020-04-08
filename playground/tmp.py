# -*- coding: utf-8 -*-

import pandas as pd
from math import exp 
import numpy as np

path = './../output/segmentation/m0_unet_m_sumloss_random/test_predictions.csv'
df = pd.read_csv(path)
aa = df.values
qq = np.array(list(aa[:,2]))
qq1 = 1/(1+np.exp(-1*qq))