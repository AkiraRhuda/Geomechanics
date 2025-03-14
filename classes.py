import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_excel

class Overburden:
    def __init__(self, wellDF, wellinfoDF):
        self.start(wellDF, wellinfoDF)
        self.plot(wellDF)
        self.tensionDF
        self.gradDF
        self.totalprofDF

    @staticmethod
    def tension(x, y):
        return 1.422 * x * y

    @staticmethod
    def grad(tensi, depth):
        return tensi / (0.1704 * (depth))

    def start(self, wellDF,wellinfoDF):
        wellDF = wellDF.fillna(0)
        # Data #
        self.tensionDF = pd.DataFrame(np.zeros(len(wellDF.index)))
        self.gradDF = pd.DataFrame(np.zeros(len(wellDF.index)))
        self.totalprofDF = pd.DataFrame(np.zeros(len(wellDF.index)))
        self.totalprofDF = wellDF['prof (m)'] + wellinfoDF[2]['LÂMINA DÁGUA (m):']

        rho_seawater = wellinfoDF[2][0]
        rho_region = wellinfoDF[2][1]
        water_depth = wellinfoDF[2][2]
        #air_gap = wellinfoDF[2][3] if wellinfoDF[2][3] is not 

        for i in range(len(wellDF.index)):
            if i==0:
                self.tensionDF[0][i] = self.tension(water_depth, rho_seawater)
                self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])
            else:
                self.tensionDF[0][i] = self.tensionDF[0][i-1] + self.tension(wellDF['ΔD (m)'][i],wellDF['ρ (g/cm3)'][i])
                self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])


    def plot(self, wellDF):
        plt.plot(self.tensionDF, wellDF['prof (m)'], color='red')
        plt.xlabel('Tensão [$psi$]')
        plt.ylabel('Profundidade [$m$]')
        plt.title('Pressão da sobrecarga $versus$ profundidade')
        plt.show()

        plt.plot(self.gradDF, self.totalprofDF, color='purple')
        plt.xlabel('Gradiente de sobrecarga [$lb/gal$]')
        plt.ylabel('Profundidade [$m$]')
        plt.title('Gradiente de sobrecarga $versus$ Profundidade')
        plt.show()
        
        
        
class Gardnercorrelation:
        def __init__(self, wellDF, wellinfoDF, a, b):
            self.wellDF, self.wellinfoDF, self.a, self.b = wellDF, wellinfoDF, a, b
            self.calculate()
            
        def calculate(self):
                self.wellDF = self.wellDF.fillna(0)
                # Data #
                
                for i in range(len(self.wellDF.index)):
                    self.wellDF['ρ (g/cm3)'] = self.a * (10**6/self.wellDF['Δt (μs/ft)'])**self.b
                self.wellDF.replace([np.inf, -np.inf], np.nan, inplace=True)
                return self.wellDF
            
"""class Belloti:
        def __init__(self, wellDF, wellinfoDF):
            self.wellDF, self.wellinfoDF = wellDF, wellinfoDF
            self.calculate()
            
        def calculate(self):
                self.wellDF = self.wellDF.fillna(0)
                # Data #
                
                for i in range(len(self.wellDF.index)):
                    if wellDF['Δt (μs/ft)'] > 100
                    self.wellDF['ρ (g/cm3)'] = self.a * (10**6/self.wellDF['Δt (μs/ft)'])**self.b
                self.wellDF.replace([np.inf, -np.inf], np.nan, inplace=True)
                return self.wellDF"""