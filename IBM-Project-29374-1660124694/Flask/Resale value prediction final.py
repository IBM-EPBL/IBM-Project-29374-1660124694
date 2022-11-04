import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder
import pickle


df =  pd.read_csv("Data/autos.csv",header=0,sep= ',',encoding='Latin1',)
