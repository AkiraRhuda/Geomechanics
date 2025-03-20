import numpy as np
import pandas as pd
import geomec_classes
import classes
import matplotlib.pyplot as plt
from pandas import read_excel


wellDF = pd.read_excel('source\\tabela 3.xlsx', skiprows=4, index_col=None, decimal=',')
wellinfoDF = pd.read_excel('source\\tabela 3.xlsx', skipfooter=23,header=None,index_col=0, decimal=',')

geomec_classes.Bourgoyne(wellDF, wellinfoDF, name='aaaaa', water=True, sumwater=True, points=10)