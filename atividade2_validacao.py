import pandas as pd
import geomec_classes


wellDF = pd.read_excel('source\\atividade2_validacao\\DADOS_1.xlsx', skiprows=4, index_col=None, decimal=',')
wellinfoDF = pd.read_excel('source\\atividade2_validacao\\DADOS_1.xlsx', skipfooter=67,header=None,index_col=0, decimal=',')

wellDF1 = geomec_classes.Gardnercorrelation(wellDF, wellinfoDF, a=0.234, b=0.25).calculate()
wellDF1, _, _,  _, _, _ = geomec_classes.Overburden(wellDF1, wellinfoDF,name='atividade2_validacao', water=0).start()
wellDF1, _, _,  _, _, _ = geomec_classes.NormalTensionandGrad(wellDF1, wellinfoDF,name='atividade2_validacao', sumwater=False).start()
wellDF1 = geomec_classes.Eaton(wellDF1, wellinfoDF, name='atividade2_validacao', top=43, exponum=3, sumwater=False).start()
