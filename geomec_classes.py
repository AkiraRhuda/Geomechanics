import os
from typing import Concatenate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import classes
from pandas import read_excel
from pandas.core.methods.selectn import SelectNSeries


class Overburden:
    """
    Calculate the tension and the overburden, then plot graphics and export the found values in a .xlsx file

    Parameters
    ----------
    wellDF : dict or Any
        Dataframe containing well information
    wellinfoDF : dict or Any
        Dataframe containing environment information
    name : str
        Parameter name (used for exporting .xlsx)
    water : int
        Insert the water layer (currently only supports first layer [surface])
    unknownregion : int
        Insert the unknownregion layer
    sumwater : bool
            Make the class sum the water depth in the total depth dataframe
    """
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
        self.wellDF = wellDF
        return self.wellDF, self.wellinfoDF, self.tensionDF,  self.gradDF, self.totalprofDF, self.water_depth

    def plot(self):

        # Cleaning zero values for Tension and overburden #
        self.tensionDF['Tension'] = self.tensionDF['Tension'].loc[(self.tensionDF['Tension'] != 0)]
        self.gradDF['Overburden'] = self.gradDF['Overburden'].loc[(self.gradDF['Overburden'] != 0)]

        plt.plot(self.tensionDF, self.wellDF['prof (m)'], color='C12', marker='o', ls='--')
        plt.xlabel('Tensão [$psi$]')
        plt.ylabel('Profundidade [$m$]')
        plt.title('Pressão da sobrecarga $versus$ profundidade')
        plt.grid()
        plt.ylim([0, self.wellDF['prof (m)'].max()])
        plt.gca().invert_yaxis()
        plt.savefig(f'output\\{self.name} - Pressão sobrecarga.jpg', format='jpg', dpi=800)
        plt.show()

        plt.plot(self.gradDF, self.totalprofDF, color='purple', marker='o', ls='--')
        plt.axline((0, self.totalprofDF.max()),(1, self.totalprofDF.max()), color='black', ls=':',
                   label=f'Profun máx = {self.totalprofDF.max()} $m$')
        plt.axline((0, self.water_depth), (1, self.water_depth), color='blue', ls=':',
                   label=f'Lâmi dágua = {self.water_depth} $m$')
        plt.xlabel('Gradiente de sobrecarga [$lb/gal$]')
        plt.ylabel('Profundidade [$m$]')
        plt.ylim([0, self.totalprofDF.max()])
        plt.title('Gradiente de sobrecarga $versus$ Profundidade')
        plt.legend(loc='best')
        plt.gca().invert_yaxis()
        plt.grid()
        plt.savefig(f'output\\{self.name} - Gradiente de sobrecarga.jpg', format='jpg', dpi=800)
        plt.show()
        
        plt.plot(self.wellDF['ρ (g/cm3)'], self.totalprofDF, color='orange', marker='o', ls='--')
        plt.axline((0, self.totalprofDF.max()),(1, self.totalprofDF.max()), color='black', ls=':',
                   label=f'Profun máx = {self.totalprofDF.max()} $m$')
        plt.axline((0, self.water_depth), (1, self.water_depth), color='blue', ls=':',
                   label=f'Lâmi dágua = {self.water_depth} $m$')
        plt.xlabel('Densidade [$(g/cm3)$]')
        plt.ylabel('Profundidade [$m$]')
        plt.ylim([0, self.totalprofDF.max()])
        plt.title('Densidade $versus$ Profundidade')
        plt.legend(loc='best')
        plt.gca().invert_yaxis()
        plt.grid()
        plt.savefig(f'output\\{self.name} - Densidade.jpg', format='jpg', dpi=800)
        plt.show()

class multiplot:
    """
    Use all the dataframes returned by Overburden class or Bourgoyne and plot two wells in the same graphics

    Parameters
    ----------
    wellDF : dict or Any
    tensionDF : dict or Any
    gradDF  : dict or Any
    totalprofDF  : dict or Any
    water_depth : dict or Any
    wellDF2 : dict or Any
    tensionDF2 : dict or Any
    gradDF2 : dict or Any
    totalprofDF2 : dict or Any
    water_depth2 : dict or Any
    """
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

        plt.plot(self.tensionDF, self.wellDF['prof (m)'], color='red', marker='o', ls='--', label='Well 1')
        plt.plot(self.tensionDF2, self.wellDF2['prof (m)'], color='C12', marker='o', ls='--', label='Well 2')
        plt.xlabel('Tensão [$psi$]')
        plt.ylabel('Profundidade [$m$]')
        plt.title('Pressão da sobrecarga $versus$ profundidade')
        plt.grid()
        plt.legend(loc='best')
        if self.wellDF['prof (m)'].max() > self.wellDF2['prof (m)'].max():
            plt.ylim([0, self.wellDF['prof (m)'].max()])
            plt.gca().invert_yaxis()
        else:
            plt.ylim([0, self.wellDF2['prof (m)'].max()])
            plt.gca().invert_yaxis()
        plt.savefig(f'output\\Multiplot - Pressão da sobrecarga.jpg', format='jpg', dpi=800)
        plt.show()

        plt.plot(self.gradDF, self.totalprofDF, color='purple', marker = 'o', ls = '--', label='Well 1')
        plt.plot(self.gradDF2, self.totalprofDF2, color='yellow', marker = 'o', ls = '--', label='Well 2')
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
        plt.savefig(f'output\\Multiplot - Gradiente de sobrecarga.jpg', format='jpg', dpi=800)
        plt.show()

class Gardnercorrelation:
        """
            Using transit time (μs/ft), calculates the density (g/cm3) using Gardner method and returns it in a new wellDF dataframe to be
            used in Overburden class
            Equation defined as 'a*(10**6/Δt)**b'
            Parameters
            ----------
            wellDF : dict or Any
                Dataframe containing well information
            wellinfoDF : dict or Any
                Dataframe containing environment information
            a : float
            b : float
        """
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
        """
            Using transit time (μs/ft), calculates the density (g/cm3) using Bellotti method and returns it in a new wellDF dataframe to be
            used in Overburden class
            Parameters
            ----------
            wellDF : dict or Any
                Dataframe containing well information
            wellinfoDF : dict or Any
                Dataframe containing environment information
            dtmatrix : float
                Transit time of matrix
            force_condition : str
                Force a condition in the class, where it can be 'consolidated' or 'unconsolidated'
            """
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
    """
    Calculate the density (g/cm3) using a linear regression of the exponential data.
    Then, plot the graphic of porosity versus depth
    and calculate the tension and the overburden values using Bourgoyne method.
    Finally, export all the found values in a .xlsx file
    DOES NOT SUPPORT UNKNOWNREGION
    Parameters
    ----------
    wellDF : dict
        Dataframe containing well information
    wellinfoDF : dict
        Dataframe containing environment information
    name : str
        Parameter name (used for exporting .xlsx)
    water : int
        Insert the water layer (currently only supports first layer [surface])
    sumwater : bool
        Make the class sum the water depth in the total depth
    points : int
        As Bourgoyne method returns a continuous function, tells how many points does the function will use to perform
        calculations
    """
    def __init__(self, wellDF, wellinfoDF, name, water=None, sumwater = None, points=None):
            self.wellDF, self.wellinfoDF = wellDF, wellinfoDF
            self.rho_seawater = None
            self.rho_region = None
            self.name = name
            self.water = water
            self.sumwater = sumwater
            self.tensionDF = None
            self.gradDF = None
            self.D = None
            self.points = points
            self.plotporosprof()
            self.constants()
            self.plotdensiprof()
            self.density()
            self.calculate()
            self.plot()
            
            
    def plotporosprof(self):        
        plt.plot(self.wellDF['prof (m)'], self.wellDF['Porosidade'], color='red', marker = 'o', ls = '--')
        plt.ylabel('Porosidade')
        plt.xlabel('Profundidade [$m$]')
        plt.title('Porosidade versus profundidade')
        plt.grid()
        plt.savefig(f'output\\{self.name} - Porosidade versus profundidade.jpg', format='jpg', dpi=800)
        plt.show()
        
    def constants(self):
        self.phi0, self.K0 = classes.exponentialmodel(self.wellDF).export()
        self.K0 = self.K0 * (-1)
        self.phi0 = np.e**self.phi0
        self.rho_seawater = self.wellinfoDF[1][0]
        self.rho_region = self.wellinfoDF[1][1]
        self.water_depth = self.wellinfoDF[1][2]
        if self.points is None:
            self.points = 100
        else:
            pass
        
        
    def density(self):
        return self.rho_region * (1 - self.phi0 * np.e ** (-self.K0*self.D)) + self.rho_seawater * self.phi0 * np.e ** (-self.K0*self.D)
    

    def tension(self):
        return 1.422 * (self.rho_seawater*self.water_depth + self.rho_region*self.D - ((self.rho_region-self.rho_seawater)/self.K0)*self.phi0*(1-np.e**(-self.K0*self.D)))

    @staticmethod
    def grad(tensi, depth):
        return tensi / (0.1704 * (depth))
        
    def plotdensiprof(self):
        self.D = np.linspace(self.wellDF['prof (m)'].min(), self.wellDF['prof (m)'].max(), self.points)
        plt.plot(self.density(), self.D, color='green')
        plt.grid()
        plt.title('Densidade versus profundidade')
        plt.xlabel('Densidade')
        plt.ylabel('Profundidade [$m$]')
        plt.legend(loc='best')
        plt.gca().invert_yaxis()
        plt.savefig(f'output\\{self.name} - Densidade versus profundidade.jpg', format='jpg', dpi=800)
        plt.show()
        
    def calculate(self):
        self.wellDF['Porosidade'] = self.wellDF['Porosidade'] .fillna(0)
        self.tensionDF = pd.DataFrame(np.zeros(len(self.D)))
        self.gradDF = pd.DataFrame(np.zeros(len(self.D)))
        self.totalprofDF = pd.DataFrame(np.zeros(len(self.D)))
        
        if self.sumwater == True:
            self.totalprofDF[0] = self.D + self.water_depth
        else:
            self.totalprofDF = self.D
        
        if self.water is not None:
            self.tensionDF[0] = self.tension()
            for i in range(len(self.tensionDF.index)):
                self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[0][i])
                    
        else:
            Exception(f"Code doesn't get up there, wait updates...")
        self.totalprofDF.columns = ['new prof (m)']
        self.wellDF = pd.concat([self.wellDF, self.totalprofDF], axis=1)
        self.density = pd.DataFrame(self.density())
        self.density.columns = ['Density']
        self.wellDF = pd.concat([self.wellDF, self.density], axis=1)
        self.tensionDF.columns = ['Tension']
        self.gradDF.columns = ['Overburden']
        self.wellDF = pd.concat([self.wellDF, self.tensionDF], axis=1)
        self.wellDF = pd.concat([self.wellDF, self.gradDF], axis=1)
        self.wellDF.to_excel(f'output\\{self.name}.xlsx')

        return self.wellDF, self.wellinfoDF, self.tensionDF, self.gradDF, self.totalprofDF, self.water_depth
        
    def plot(self):
    
        # Cleaning zero values for Tension and overburden #
        self.tensionDF['Tension'] = self.tensionDF['Tension'].loc[(self.tensionDF['Tension'] != 0)]
        self.gradDF['Overburden'] = self.gradDF['Overburden'].loc[(self.gradDF['Overburden'] != 0)]

        plt.plot(self.tensionDF, self.totalprofDF, color='C12', marker='o', ls='--')
        plt.xlabel('Tensão [$psi$]')
        plt.ylabel('Profundidade [$m$]')
        plt.title('Pressão da sobrecarga $versus$ profundidade')
        plt.grid()
        plt.ylim([0, self.totalprofDF['new prof (m)'].max()])
        plt.gca().invert_yaxis()
        plt.savefig(f'output\\{self.name} - Pressão sobrecarga.jpg', format='jpg', dpi=800)
        plt.show()

        plt.plot(self.gradDF, self.totalprofDF, color='purple', marker='o', ls='--')
        plt.axline((0, self.totalprofDF['new prof (m)'].max()),(10, self.totalprofDF['new prof (m)'].max()), color='black', ls=':',
                   label=f'Profun máx = {float(self.totalprofDF.max())} $m$')
        plt.axline((0, self.water_depth), (10, self.water_depth), color='blue', ls=':',
                   label=f'Lâmi dágua = {self.water_depth} $m$')
        plt.xlabel('Gradiente de sobrecarga [$lb/gal$]')
        plt.ylabel('Profundidade [$m$]')
        plt.ylim([0, self.totalprofDF['new prof (m)'].max()])
        plt.title('Gradiente de sobrecarga $versus$ Profundidade')
        plt.legend(loc='best')
        plt.gca().invert_yaxis()
        plt.grid()
        plt.savefig(f'output\\{self.name} - Gradiente de sobrecarga.jpg', format='jpg', dpi=800)
        plt.show()