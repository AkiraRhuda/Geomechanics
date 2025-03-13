import numpy as np
import pandas as pd
import estimativecorrelation
import matplotlib.pyplot as plt
from pandas import read_excel

estcorDF = pd.read_excel('source\\correlacoes_estimativa_tensao_sobrecarga.xlsx', skiprows=4, index_col=None)
wellinfoDF = pd.read_excel('source\\correlacoes_estimativa_tensao_sobrecarga.xlsx', skipfooter=22,header=None,index_col=0)

estimativecorrelation.Estimativecorrelation(estcorDF,wellinfoDF)