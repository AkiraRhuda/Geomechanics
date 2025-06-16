import pandas as pd
import geomec_classes

wellDF1 = pd.read_csv('source\\atividade4 teste\\AV4_tensoes_insitu_poisson_MM(in).csv', skiprows=None, index_col=None, decimal='.')
wellDF2 = pd.read_excel('source\\atividade4 teste\\DADOS_1.xlsx', skiprows=4, index_col=None, decimal=',')
wellDF = pd.concat([wellDF2,wellDF1], axis=1)
wellinfoDF = pd.read_excel('source\\atividade4 teste\\DADOS_1.xlsx', skipfooter=67,header=None,index_col=0, decimal=',')

wellDF = geomec_classes.Gardnercorrelation(wellDF, wellinfoDF, a=0.234, b=0.25).calculate()
wellDF, _, _,  _, _, _ = geomec_classes.Overburden(wellDF, wellinfoDF,name='atividade4 teste', water=0).start()
wellDF, _, _,  _, _, _ = geomec_classes.NormalTensionandGrad(wellDF, wellinfoDF,name='atividade4 teste', sumwater=False).start()
wellDF, top = geomec_classes.Eaton(wellDF, wellinfoDF, name='atividade4 teste', top=36, exponum=3, sumwater=False).start()
wellDF = geomec_classes.Porepressure(wellDF).output()
wellDF = geomec_classes.Hydrostaticpressure(wellDF).output()
wellDF = geomec_classes.CollapseGradient(wellDF).output()
wellDF = geomec_classes.FractureGradient(wellDF).output()
geomec_classes.Mudweightwindow(wellDF, name='atividade4 teste', top=top)