import numpy as np
import pandas as pd
import geomec_classes
import matplotlib.pyplot as plt
from pandas import read_excel

# Ex 1
wellDF = pd.read_excel('source\\correlacoes_estimativa_tensao_sobrecarga.xlsx', skiprows=4, index_col=None, decimal=',')
wellinfoDF = pd.read_excel('source\\correlacoes_estimativa_tensao_sobrecarga.xlsx', skipfooter=22,header=None,index_col=0, decimal=',')

geomec_classes.Overburden(wellDF, wellinfoDF,name='Ex1', water=0, sumwater=True)

# Ex 2

wellDF = pd.read_excel('source\\tabela 2.xlsx', skiprows=4, index_col=None, decimal=',')
wellinfoDF = pd.read_excel('source\\tabela 2.xlsx', skipfooter=41,header=None,index_col=0, decimal=',')


wellDF1 = geomec_classes.Gardnercorrelation(wellDF, wellinfoDF, a=0.234, b=0.25).calculate()

geomec_classes.Overburden(wellDF1, wellinfoDF,name='Ex2 - Gardner', water=0, unknownregion=1)

wellDF2 = geomec_classes.Bellotti(wellDF, wellinfoDF, force_condition='consolidated').calculate()
geomec_classes.Overburden(wellDF2, wellinfoDF,name='Ex2 - Bellotti', water=0, unknownregion=1)

# Multiplot #

wellDF1 = geomec_classes.Gardnercorrelation(wellDF, wellinfoDF, a=0.234, b=0.25).calculate()

wellDF1, wellinfoDF, tensionDF,  gradDF, totalprofDF, water_depth = geomec_classes.Overburden(wellDF1, wellinfoDF,name='Ex2 - Gardner', water=0, unknownregion=1).start()

wellDF2 = geomec_classes.Bellotti(wellDF, wellinfoDF, force_condition='consolidated').calculate()
wellDF2, wellinfoDF2, tensionDF2,  gradDF2, totalprofDF2, water_depth2 = geomec_classes.Overburden(wellDF2, wellinfoDF,name='Ex2 - Bellotti', water=0, unknownregion=1).start()

geomec_classes.multiplot(wellDF1, tensionDF,  gradDF, totalprofDF, water_depth, wellDF2, tensionDF2,  gradDF2, totalprofDF2, water_depth2)


# Ex 3

wellDF = pd.read_excel('source\\tabela 3.xlsx', skiprows=4, index_col=None, decimal=',')
wellinfoDF = pd.read_excel('source\\tabela 3.xlsx', skipfooter=23,header=None,index_col=0, decimal=',')

geomec_classes.Bourgoyne(wellDF, wellinfoDF, name='Ex3', water=True, sumwater=True)

