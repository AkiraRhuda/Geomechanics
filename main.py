import numpy as np
import pandas as pd
import classes
import matplotlib.pyplot as plt
from pandas import read_excel

# Ex 1

wellDF = pd.read_excel('source\\correlacoes_estimativa_tensao_sobrecarga.xlsx', skiprows=4, index_col=None)
wellinfoDF = pd.read_excel('source\\correlacoes_estimativa_tensao_sobrecarga.xlsx', skipfooter=22,header=None,index_col=0)

classes.Overburden(wellDF, wellinfoDF,name='1', water=0, sumwater=True)

# Ex 2

wellDF = pd.read_excel('source\\tabela 2.xlsx', skiprows=4, index_col=None)
wellinfoDF = pd.read_excel('source\\tabela 2.xlsx', skipfooter=41,header=None,index_col=0)

wellDF = classes.Gardnercorrelation(wellDF, wellinfoDF, a=0.234, b=0.25).calculate()
classes.Overburden(wellDF, wellinfoDF,name='2', water=0, unknownregion=1)

wellDF = classes.Belloti(wellDF, wellinfoDF).calculate()
classes.Overburden(wellDF, wellinfoDF,name='2.1')