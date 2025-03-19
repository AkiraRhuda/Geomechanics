import numpy as np
import pandas as pd
import classes
import matplotlib.pyplot as plt
from pandas import read_excel
"""
# Ex 1
wellDF = pd.read_excel('source\\correlacoes_estimativa_tensao_sobrecarga.xlsx', skiprows=4, index_col=None, decimal=',')
wellinfoDF = pd.read_excel('source\\correlacoes_estimativa_tensao_sobrecarga.xlsx', skipfooter=22,header=None,index_col=0, decimal=',')

classes.Overburden(wellDF, wellinfoDF,name='1', water=0, sumwater=True)

# Ex 2

wellDF = pd.read_excel('source\\tabela 2.xlsx', skiprows=4, index_col=None, decimal=',')
wellinfoDF = pd.read_excel('source\\tabela 2.xlsx', skipfooter=41,header=None,index_col=0, decimal=',')


wellDF1 = classes.Gardnercorrelation(wellDF, wellinfoDF, a=0.234, b=0.25).calculate() 

classes.Overburden(wellDF1, wellinfoDF,name='2', water=0, unknownregion=1)

wellDF2 = classes.Belloti(wellDF, wellinfoDF, force_condition='consolidated').calculate()
classes.Overburden(wellDF2, wellinfoDF,name='2.1', water=0, unknownregion=1)


# Multiplot #

wellDF1 = classes.Gardnercorrelation(wellDF, wellinfoDF, a=0.234, b=0.25).calculate() 

wellDF1, wellinfoDF, tensionDF,  gradDF, totalprofDF, water_depth = classes.Overburden(wellDF1, wellinfoDF,name='2', water=0, unknownregion=1).start()

wellDF2 = classes.Belloti(wellDF, wellinfoDF, force_condition='consolidated').calculate()
wellDF2, wellinfoDF2, tensionDF2,  gradDF2, totalprofDF2, water_depth2 = classes.Overburden(wellDF2, wellinfoDF,name='2.1', water=0, unknownregion=1).start()

#classes.multiplot(wellDF, tensionDF,  gradDF, totalprofDF, water_depth, wellDF2, tensionDF2,  gradDF2, totalprofDF2, water_depth2)
classes.multiplot(wellDF1, tensionDF,  gradDF, totalprofDF, water_depth, wellDF2, tensionDF2,  gradDF2, totalprofDF2, water_depth2)
"""

# Ex 3

wellDF = pd.read_excel('source\\tabela 3.xlsx', skiprows=4, index_col=None, decimal=',')
wellinfoDF = pd.read_excel('source\\tabela 3.xlsx', skipfooter=23,header=None,index_col=0, decimal=',')
classes.Bourgoyne(wellDF, wellinfoDF)