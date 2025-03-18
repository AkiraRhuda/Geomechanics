import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_excel

class Overburden:
    def __init__(self, wellDF, wellinfoDF, name, water=None, unknownregion=None, sumwater=None):
        self.start(wellDF, wellinfoDF, name, water, unknownregion, sumwater)
        self.plot(wellDF)
        self.tensionDF = None
        self.gradDF = None
        self.totalprofDF = None
        self.unknownregion = None
        self.water = None
        self.rho_seawater = None
        self.rho_region = None
        self.water_depth = None

    @staticmethod
    def tension(x, y):
        return 1.422 * x * y

    @staticmethod
    def grad(tensi, depth):
        return tensi / (0.1704 * (depth))

    def start(self, wellDF,wellinfoDF, name, water, unknownregion, sumwater):
        # Removing NaN entries #
        wellDF = wellDF.fillna(0)

        # Data and Data maniputalion #
        self.tensionDF = pd.DataFrame(np.zeros(len(wellDF.index)))
        self.gradDF = pd.DataFrame(np.zeros(len(wellDF.index)))
        self.totalprofDF = pd.DataFrame(np.zeros(len(wellDF.index)))
        if sumwater == True:
            self.totalprofDF = wellDF['prof (m)'] + wellinfoDF[2]['LÂMINA DÁGUA (m):']
        else:
            self.totalprofDF = wellDF['prof (m)']
        self.unknownregion = unknownregion if unknownregion is not None else None
        self.water = water if water is not None else None

        # Extracting constants #
        self.rho_seawater = wellinfoDF[2][0]
        self.rho_region = wellinfoDF[2][1]
        self.water_depth = wellinfoDF[2][2]

        if self.unknownregion is None and self.water is not None:
            for i in range(len(wellDF.index)):
                if i==self.water: # ALWAYS SURFACE #
                    self.tensionDF[0][i] = self.tension(self.water_depth, self.rho_seawater)
                    self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])
                else:
                    self.tensionDF[0][i] = self.tensionDF[0][i-1] + self.tension(wellDF['ΔD (m)'][i],wellDF['ρ (g/cm3)'][i])
                    self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])

        elif self.unknownregion is not None and self.water is None:
            for i in range(len(wellDF.index)):
                if i==0 and not self.unknownregion == 0:
                    self.tensionDF[0][i] = self.tension(wellDF['ΔD (m)'][i],wellDF['ρ (g/cm3)'][i])
                    self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])

                elif i == self.unknownregion and i == 0:
                    self.tensionDF[0][i] = self.tension(wellDF['ΔD (m)'][i], self.rho_region)
                    self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])

                elif i == self.unknownregion:
                    self.tensionDF[0][i] = self.tensionDF[0][i-1] + self.tension(wellDF['ΔD (m)'][i], self.rho_region)
                    self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])
                else:
                    self.tensionDF[0][i] = self.tensionDF[0][i-1] + self.tension(wellDF['ΔD (m)'][i],wellDF['ρ (g/cm3)'][i])
                    self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])

        elif self.unknownregion is None and self.water is None:
            for i in range(len(wellDF.index)):
                if i==0:
                    self.tensionDF[0][i] = self.tension(wellDF['ΔD (m)'][i],wellDF['ρ (g/cm3)'][i])
                    self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])
                else:
                    self.tensionDF[0][i] = self.tensionDF[0][i-1] + self.tension(wellDF['ΔD (m)'][i],wellDF['ρ (g/cm3)'][i])
                    self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])

        else:
            for i in range(len(wellDF.index)):
                if i==self.water: # ALWAYS SURFACE #s
                    self.tensionDF[0][i] = self.tension(self.water_depth, self.rho_seawater)
                    self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])
                elif i==self.unknownregion:
                    self.tensionDF[0][i] = self.tensionDF[0][i-1] + self.tension(wellDF['ΔD (m)'][i], self.rho_region)
                    self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])
                else:
                    self.tensionDF[0][i] = self.tensionDF[0][i-1] + self.tension(wellDF['ΔD (m)'][i],wellDF['ρ (g/cm3)'][i])
                    self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])
        os.makedirs('output', exist_ok=True)
        wellDF.to_excel(f'output\\{name}.xlsx')


    def plot(self, wellDF):
        plt.plot(self.tensionDF, wellDF['prof (m)'], color='red')
        plt.xlabel('Tensão [$psi$]')
        plt.ylabel('Profundidade [$m$]')
        plt.title('Pressão da sobrecarga $versus$ profundidade')
        plt.grid()
        plt.ylim([0, wellDF['prof (m)'].max()])
        plt.show()

        plt.plot(self.gradDF, self.totalprofDF, color='purple')
        plt.axline((0, self.totalprofDF.max()),(10, self.totalprofDF.max()), color='black', ls=':',
                   label=f'Profun máx = {self.totalprofDF.max()} $m$')
        plt.axline((0, self.water_depth), (10, self.water_depth), color='blue', ls=':',
                   label=f'Lâmi dágua = {self.water_depth} $m$')
        #plt.plot(self.totalprofDF.max(),self.totalprofDF, color='darkblue')
        plt.xlabel('Gradiente de sobrecarga [$lb/gal$]')
        plt.ylabel('Profundidade [$m$]')
        plt.ylim([0, self.totalprofDF.max()])
        plt.title('Gradiente de sobrecarga $versus$ Profundidade')
        plt.legend(loc='best')
        plt.gca().invert_yaxis()
        plt.grid()
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
            
class Belloti:
        def __init__(self, wellDF, wellinfoDF):
            self.wellDF, self.wellinfoDF = wellDF, wellinfoDF
            self.calculate()
            
        def calculate(self):
                self.wellDF = self.wellDF.fillna(0)
                # Data #
                
                for i in range(len(self.wellDF.index)):
                    if self.wellDF['Δt (μs/ft)'][i] > 100:
                        self.wellDF['ρ (g/cm3)'][i] = 0
                    elif self.wellDF['Δt (μs/ft)'][i] < 100 and self.wellDF['Δt (μs/ft)'][i] != 0:
                        self.wellDF['ρ (g/cm3)'][i] = 3.28 - self.wellDF['Δt (μs/ft)'][i]/88.95
                self.wellDF.replace([np.inf, -np.inf], np.nan, inplace=True)
                return self.wellDF

        #def plot(self):