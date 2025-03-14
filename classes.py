import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_excel

class Estimativecorrelation:
    def __init__(self, estcorDF, wellinfoDF):
        self.start(estcorDF, wellinfoDF)
        self.plot(estcorDF)
        self.tensionDF
        self.gradDF
        self.totalprofDF

    @staticmethod
    def tension(x, y):
        return 1.422 * x * y

    @staticmethod
    def grad(tensi, depth):
        return tensi / (0.1704 * (depth))

    def start(self, estcorDF,wellinfoDF):
        estcorDF = estcorDF.fillna(0)
        # Data #
        self.tensionDF = pd.DataFrame(np.zeros(len(estcorDF.index)))
        self.gradDF = pd.DataFrame(np.zeros(len(estcorDF.index)))
        self.totalprofDF = pd.DataFrame(np.zeros(len(estcorDF.index)))
        self.totalprofDF = estcorDF['prof (m)'] + wellinfoDF[2]['LÂMINA DÁGUA (m):']

        rho_seawater = wellinfoDF[2][0]
        rho_region = wellinfoDF[2][1]
        water_depth = wellinfoDF[2][2]
        air_gap = wellinfoDF[2][3]

        for i in range(len(estcorDF.index)):
            if i==0:
                self.tensionDF[0][i] = self.tension(water_depth, rho_seawater)
                self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])
            else:
                self.tensionDF[0][i] = self.tensionDF[0][i-1] + self.tension(estcorDF['ΔD (m)'][i],estcorDF['ρ (g/cm3)'][i])
                self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])


    def plot(self, estcorDF):
        plt.plot(self.tensionDF, estcorDF['prof (m)'], color='red')
        plt.xlabel('Tensão [$psi$]')
        plt.ylabel('Profundidade [$m$]')
        plt.title('Pressão da sobrecarga $versus$ profundidade')
        plt.show()

        plt.plot(self.gradDF, self.totalprofDF, color='purple')
        plt.xlabel('Gradiente de sobrecarga [$lb/gal$]')
        plt.ylabel('Profundidade [$m$]')
        plt.title('Gradiente de sobrecarga $versus$ Profundidade')
        plt.show()