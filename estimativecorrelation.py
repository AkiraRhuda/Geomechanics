import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_excel

# Importing Data #
estcorDF = pd.read_excel('source\\correlacoes_estimativa_tensao_sobrecarga.xlsx', skiprows=4, index_col=None)
estcorDF = estcorDF.fillna(0)
wellinfoDF = pd.read_excel('source\\correlacoes_estimativa_tensao_sobrecarga.xlsx', skipfooter=22,header=None,index_col=0)

# Data #
tensionDF = pd.DataFrame(np.zeros(len(estcorDF.index)))
gradDF = pd.DataFrame(np.zeros(len(estcorDF.index)))
totalprofDF = pd.DataFrame(np.zeros(len(estcorDF.index)))
totalprofDF = estcorDF['prof (m)'] + wellinfoDF[2]['LÂMINA DÁGUA (m):']

print(estcorDF)
print(wellinfoDF)

rho_seawater = wellinfoDF[2][0]
rho_region = wellinfoDF[2][1]
water_depth = wellinfoDF[2][2]
air_gap = wellinfoDF[2][3]

def tension(x,y):
    return 1.422*x*y

def grad(tensi,depth):
    return tensi/(0.1704*(depth))


for i in range(len(estcorDF.index)):
    if i==0:
        tensionDF[0][i] = tension(water_depth, rho_seawater)
        gradDF[0][i] = grad(tensionDF[0][i], totalprofDF[i])
    else:
        tensionDF[0][i] = tensionDF[0][i-1] + tension(estcorDF['ΔD (m)'][i],estcorDF['ρ (g/cm3)'][i])
        gradDF[0][i] = grad(tensionDF[0][i], totalprofDF[i])


plt.plot(tensionDF, estcorDF['prof (m)'], color='red')
plt.xlabel('Tensão [$psi$]')
plt.ylabel('Profundidade [$m$]')
plt.title('Pressão da sobrecarga $versus$ profundidade')
plt.show()

plt.plot(gradDF, totalprofDF, color='purple')
plt.xlabel('Gradiente de sobrecarga [$lb/gal$]')
plt.ylabel('Profundidade [$m$]')
plt.title('Gradiente de sobrecarga $versus$ Profundidade')
plt.show()

"""plt.plot(totalprofDF,gradDF, color='purple')
plt.xlabel('Profundidade [$m$]')
plt.ylabel('Gradiente de sobrecarga [$lb/gal$]')
plt.title('Gradiente de sobrecarga $versus$ Profundidade')
plt.show()"""