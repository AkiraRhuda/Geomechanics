import os
from typing import Concatenate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_excel
from pandas.core.methods.selectn import SelectNSeries


class Overburden:
    def __init__(self, wellDF, wellinfoDF, name, water=None, unknownregion=None, sumwater=None):
        self.wellDF = wellDF
        self.wellinfoDF = wellinfoDF
        self.name = name
        self.water = water
        self.unknownregion = unknownregion
        self.sumwater = sumwater
        self.tensionDF = None
        self.gradDF = None
        self.totalprofDF = None
        self.unknownregion = unknownregion
        self.water = water
        self.rho_seawater = None
        self.rho_region = None
        self.water_depth = None
        self.start()
        self.plot()

    @staticmethod
    def tension(x, y):
        return 1.422 * x * y

    @staticmethod
    def grad(tensi, depth):
        return tensi / (0.1704 * (depth))

    def start(self):
        # Removing NaN entries #
        self.wellDF = self.wellDF.fillna(0)

        # Data and Data maniputalion #
        self.tensionDF = pd.DataFrame(np.zeros(len(self.wellDF.index)))
        self.gradDF = pd.DataFrame(np.zeros(len(self.wellDF.index)))
        self.totalprofDF = pd.DataFrame(np.zeros(len(self.wellDF.index)))
        if self.sumwater == True:
            self.totalprofDF = self.wellDF['prof (m)'] + self.wellinfoDF[2]['LÂMINA DÁGUA (m):']
        else:
            self.totalprofDF = self.wellDF['prof (m)']
        self.unknownregion = self.unknownregion if self.unknownregion is not None else None
        self.water = self.water if self.water is not None else None

        # Extracting constants #
        self.rho_seawater = self.wellinfoDF[2][0]
        self.rho_region = self.wellinfoDF[2][1]
        self.water_depth = self.wellinfoDF[2][2]

        if self.unknownregion is None and self.water is not None:
            for i in range(len(self.wellDF.index)):
                if i==self.water: # ALWAYS SURFACE #
                    self.tensionDF[0][i] = self.tension(self.water_depth, self.rho_seawater)
                    self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])
                else:
                    self.tensionDF[0][i] = self.tensionDF[0][i-1] + self.tension(self.wellDF['ΔD (m)'][i],self.wellDF['ρ (g/cm3)'][i])
                    self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])

        elif self.unknownregion is not None and self.water is None:
            for i in range(len(self.wellDF.index)):
                if i==0 and not self.unknownregion == 0:
                    self.tensionDF[0][i] = self.tension(self.wellDF['ΔD (m)'][i],self.wellDF['ρ (g/cm3)'][i])
                    self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])

                elif i == self.unknownregion and i == 0:
                    self.tensionDF[0][i] = self.tension(self.wellDF['ΔD (m)'][i], self.rho_region)
                    self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])

                elif i == self.unknownregion:
                    self.tensionDF[0][i] = self.tensionDF[0][i-1] + self.tension(self.wellDF['ΔD (m)'][i], self.rho_region)
                    self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])
                else:
                    self.tensionDF[0][i] = self.tensionDF[0][i-1] + self.tension(self.wellDF['ΔD (m)'][i],self.wellDF['ρ (g/cm3)'][i])
                    self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])

        elif self.unknownregion is None and self.water is None:
            for i in range(len(self.wellDF.index)):
                if i==0:
                    self.tensionDF[0][i] = self.tension(self.wellDF['ΔD (m)'][i],self.wellDF['ρ (g/cm3)'][i])
                    self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])
                else:
                    self.tensionDF[0][i] = self.tensionDF[0][i-1] + self.tension(self.wellDF['ΔD (m)'][i],self.wellDF['ρ (g/cm3)'][i])
                    self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])

        else:
            for i in range(len(self.wellDF.index)):
                if i==self.water: # ALWAYS SURFACE #s
                    self.tensionDF[0][i] = self.tension(self.water_depth, self.rho_seawater)
                    self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])
                elif i==self.unknownregion:
                    self.tensionDF[0][i] = self.tensionDF[0][i-1] + self.tension(self.wellDF['ΔD (m)'][i], self.rho_region)
                    self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])
                else:
                    self.tensionDF[0][i] = self.tensionDF[0][i-1] + self.tension(self.wellDF['ΔD (m)'][i],self.wellDF['ρ (g/cm3)'][i])
                    self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])
        self.tensionDF.columns = ['Tension']
        self.gradDF.columns = ['Overburden']
        wellDF = pd.concat([self.wellDF, self.tensionDF], axis=1)

        wellDF = pd.concat([self.wellDF, self.gradDF], axis=1)
        wellDF.to_excel(f'output\\{self.name}.xlsx')
        return self.wellDF, self.wellinfoDF, self.tensionDF,  self.gradDF, self.totalprofDF, self.water_depth

    def plot(self):

        # Cleaning zero values for Tension and overburden #
        self.tensionDF['Tension'] = self.tensionDF['Tension'].loc[(self.tensionDF['Tension'] != 0)]
        self.gradDF['Overburden'] = self.gradDF['Overburden'].loc[(self.gradDF['Overburden'] != 0)]

        plt.plot(self.tensionDF, self.wellDF['prof (m)'], color='red')
        plt.xlabel('Tensão [$psi$]')
        plt.ylabel('Profundidade [$m$]')
        plt.title('Pressão da sobrecarga $versus$ profundidade')
        plt.grid()
        plt.ylim([0, self.wellDF['prof (m)'].max()])
        plt.show()

        plt.plot(self.gradDF, self.totalprofDF, color='purple')
        plt.axline((0, self.totalprofDF.max()),(10, self.totalprofDF.max()), color='black', ls=':',
                   label=f'Profun máx = {self.totalprofDF.max()} $m$')
        plt.axline((0, self.water_depth), (10, self.water_depth), color='blue', ls=':',
                   label=f'Lâmi dágua = {self.water_depth} $m$')
        plt.xlabel('Gradiente de sobrecarga [$lb/gal$]')
        plt.ylabel('Profundidade [$m$]')
        plt.ylim([0, self.totalprofDF.max()])
        plt.title('Gradiente de sobrecarga $versus$ Profundidade')
        plt.legend(loc='best')
        plt.gca().invert_yaxis()
        plt.grid()
        plt.show()

class multiplot:
    def __init__(self, wellDF, tensionDF, gradDF, totalprofDF, water_depth, wellDF2, tensionDF2, gradDF2, totalprofDF2, water_depth2):
        self.wellDF = wellDF
        self.tensionDF = tensionDF
        self.gradDF = gradDF
        self.totalprofDF = totalprofDF
        self.water_depth = water_depth
        self.wellDF2 = wellDF2
        self.tensionDF2 = tensionDF2
        self.gradDF2 = gradDF2
        self.totalprofDF2 = totalprofDF2
        self.water_depth2 = water_depth2
        self.plot()


    def plot(self):
        
        # Cleaning zero values for Tension and overburden #
        self.tensionDF['Tension'] = self.tensionDF['Tension'].loc[(self.tensionDF['Tension'] != 0)]
        self.gradDF['Overburden'] = self.gradDF['Overburden'].loc[(self.gradDF['Overburden'] != 0)]
        self.tensionDF2['Tension'] = self.tensionDF2['Tension'].loc[(self.tensionDF2['Tension'] != 0)]
        self.gradDF2['Overburden'] = self.gradDF2['Overburden'].loc[(self.gradDF2['Overburden'] != 0)]

        
        plt.plot(self.tensionDF, self.wellDF['prof (m)'], color='red', label='Well 1')
        plt.plot(self.tensionDF2, self.wellDF2['prof (m)'], color='blue', label='Well 2')
        plt.xlabel('Tensão [$psi$]')
        plt.ylabel('Profundidade [$m$]')
        plt.title('Pressão da sobrecarga $versus$ profundidade')
        plt.grid()
        plt.legend()
        if self.wellDF['prof (m)'].max() > self.wellDF2['prof (m)'].max():
            plt.ylim([0, self.wellDF['prof (m)'].max()])
        else:
            plt.ylim([0, self.wellDF2['prof (m)'].max()])
        plt.show()

        plt.plot(self.gradDF, self.totalprofDF, color='purple', label='Well 1')
        plt.plot(self.gradDF2, self.totalprofDF2, color='pink', label='Well 2')
        plt.axline((0, self.totalprofDF.max()), (10, self.totalprofDF.max()), color='black', ls=':',
                   label=f'Profun máx well 1 = {self.totalprofDF.max()} $m$')
        plt.axline((0, self.totalprofDF2.max()), (10, self.totalprofDF2.max()), color='gray', ls=':',
                   label=f'Profun máx well 2 = {self.totalprofDF2.max()} $m$')
        plt.axline((0, self.water_depth), (10, self.water_depth), color='blue', ls=':',
                   label=f'Lâmi dágua = {self.water_depth} $m$')
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
            
class Bellotti:
        def __init__(self, wellDF, wellinfoDF, dtmatrix=None, force_condition=None):
            self.wellDF, self.wellinfoDF, self.dtmatrix = wellDF, wellinfoDF, dtmatrix
            self.force_condition = force_condition
            self.calculate()
            
        def calculate(self):
            self.wellDF = self.wellDF.fillna(0)

            # Data #
            if self.force_condition is None or False:
                for i in range(len(self.wellDF.index)):
                    if self.wellDF['Δt (μs/ft)'][i] > 100:
                        self.wellDF['ρ (g/cm3)'] = 2.75 - 2.11*(self.wellDF['Δt (μs/ft)']-self.dtmatrix)/(self.wellDF['Δt (μs/ft)']+200)
                    elif self.wellDF['Δt (μs/ft)'][i] < 100 and self.wellDF['Δt (μs/ft)'][i] != 0:
                        self.wellDF['ρ (g/cm3)'] = 3.28 - self.wellDF['Δt (μs/ft)']/88.95
            elif self.force_condition == 'consolidated':
                for i in range(len(self.wellDF.index)):
                    self.wellDF['ρ (g/cm3)'] = 3.28 - self.wellDF['Δt (μs/ft)']/88.95
            elif self.force_condition == 'unconsolidated':
                self.wellDF['ρ (g/cm3)'] = 2.75 - 2.11*(self.wellDF['Δt (μs/ft)']-self.dtmatrix)/(self.wellDF['Δt (μs/ft)']+200)
            else:
                raise Exception(f"The given condition does't exist")
                        
            self.wellDF.replace([np.inf, -np.inf], np.nan, inplace=True)
            return self.wellDF

class Bourgoyne:
    def __init__(self, wellDF, wellinfoDF):
            self.wellDF, self.wellinfoDF = wellDF, wellinfoDF
            self.rho_seawater = None
            self.rho_region = None
            self.start()
            
    def start(self):
        # Extracting constants #
        self.rho_seawater = self.wellinfoDF[1][0]
        self.rho_region = self.wellinfoDF[1][1]
        
        plt.plot(self.wellDF['Porosidade'], self.wellDF['prof (m)'], color='red')
        plt.xlabel('Porosidade')
        plt.ylabel('Profundidade [$m$]')
        plt.title('Porosidade versus profundidade')
        plt.grid()
        plt.ylim([0, self.wellDF['prof (m)'].max()])
        plt.show()