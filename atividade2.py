import numpy as np
import pandas as pd
import geomec_classes
import classes
import matplotlib.pyplot as plt
from pandas import read_excel



wellDF = pd.read_excel('source\\atividade2\\DADOS_1.xlsx', skiprows=4, index_col=None, decimal=',')
wellinfoDF = pd.read_excel('source\\atividade2\\DADOS_1.xlsx', skipfooter=67,header=None,index_col=0, decimal=',')

wellDF1 = geomec_classes.Gardnercorrelation(wellDF, wellinfoDF, a=0.234, b=0.25).calculate()
wellDF1, _, _,  _, _, _ = geomec_classes.Overburden(wellDF1, wellinfoDF,name='atividade2', water=0).start()
wellDF1, _, _,  _, _, _ = geomec_classes.Hidrostaticpressure(wellDF1, wellinfoDF,name='atividade2', sumwater=False).start()
#classes.expmodellnx(wellDF1, top=36)
wellDF1 = geomec_classes.Eaton(wellDF1, wellinfoDF, name='atividade2', top=36, exponum=3, sumwater=False).start()
#geomec_classes.EatonInteractiveMatplotlib(wellDF1, wellinfoDF, name="Poco1", top=36, exponum_range=(1, 4), sumwater=False)
