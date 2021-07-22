import random, os, datetime, pickle, json, sys
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_FLD = os.path.join('..','results_200_500_50')
PRICE_FLD = '/Users/xianggao/Dropbox/distributed/code_db/price coinbase/vm-w7r-db'

def makedirs(fld):
	if not os.path.exists(fld):
		os.makedirs(fld)
