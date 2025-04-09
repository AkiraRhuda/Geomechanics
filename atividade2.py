import numpy as np
import pandas as pd
import geomec_classes
import matplotlib.pyplot as plt
from pandas import read_excel



wellDF = pd.read_excel('source\\atividade2\\DADOS_1.xlsx', skiprows=4, index_col=None, decimal=',')
wellinfoDF = pd.read_excel('source\\atividade2\\DADOS_1.xlsx', skipfooter=67,header=None,index_col=0, decimal=',')

wellDF1 = geomec_classes.Gardnercorrelation(wellDF, wellinfoDF, a=0.234, b=0.25).calculate()
wellDF1, _, _,  _, _, _ = geomec_classes.Overburden(wellDF1, wellinfoDF,name='Gardner', water=0).start()
wellDF1 = geomec_classes.Hidrostaticpressure(wellDF1, wellinfoDF,name='Gardner', sumwater=False).start()