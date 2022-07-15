import pycaret
import IPython
#import the dataset from pycaret repository
from pycaret.datasets import get_data
anomaly = get_data('anomaly')
#import anomaly detection module
from pycaret.anomaly import *
#intialize the setup
exp_ano = setup(anomaly)

print(anomaly)