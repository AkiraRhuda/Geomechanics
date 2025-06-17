import os
#from typing import Concatenate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import classes


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
        #self.start()
        #self.plot()

    @staticmethod
    def tension(x, y):
        return 1.422 * x * y

    @staticmethod
    def grad(tensi, depth):
        return tensi / (0.1704 * depth)

    def start(self):
        """
        The function `start` processes data related to well information, calculates tension and
        overburden values, and saves the results to an Excel file.
        :return: The `start` method returns the following variables:
        1. `self.wellDF` - DataFrame containing well data
        2. `self.wellinfoDF` - DataFrame containing well information
        3. `self.tensionDF` - DataFrame containing tension data
        4. `self.gradDF` - DataFrame containing overburden data
        5. `self.totalprofDF` - DataFrame containing total depth
        """
        # Removing NaN entries #
        self.wellDF = self.wellDF.fillna(0)

        # Data maniputalion #
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

        wellDF = pd.concat([wellDF, self.gradDF], axis=1)
        wellDF.to_excel(f'output\\{self.name}.xlsx')
        self.wellDF = wellDF
        self.plot()
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
        legend = plt.legend(loc='best')
        legend._legend_box.width = 85
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
        if self.wellDF['Δt (μs/ft)'] is not None:
            self.wellDF.loc[self.wellDF['Δt (μs/ft)'] == 0, 'Δt (μs/ft)'] = np.nan  # exclude 0 values

            plt.plot(self.wellDF['Δt (μs/ft)'], self.totalprofDF, color='orange', marker='o', ls='--')
            plt.axline((0, self.totalprofDF.max()),(1, self.totalprofDF.max()), color='black', ls=':',
                       label=f'Profun máx = {self.totalprofDF.max()} $m$')
            plt.axline((0, self.water_depth), (1, self.water_depth), color='blue', ls=':',
                       label=f'Lâmi dágua = {self.water_depth} $m$')
            plt.xlabel('Tempo de Trânsito [$μs/ft$]')
            plt.ylabel('Profundidade [$m$]')
            plt.ylim([0, self.totalprofDF.max()])
            plt.title('Tempo de Trânsito $versus$ Profundidade')
            plt.legend(loc='best')
            plt.gca().invert_yaxis()
            plt.grid()
            plt.savefig(f'output\\{self.name} - Transito.jpg', format='jpg', dpi=800)
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

    def calculate(self):
        self.wellDF = self.wellDF.fillna(0)
        # Data #

        for i in range(len(self.wellDF.index)):
            self.wellDF['ρ (g/cm3)'] = self.a * (10 ** 6 / self.wellDF['Δt (μs/ft)']) ** self.b
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

    def calculate(self):
        self.wellDF = self.wellDF.fillna(0)

        # Data #
        if self.force_condition is None or False:
            for i in range(len(self.wellDF.index)):
                if self.wellDF['Δt (μs/ft)'][i] > 100:
                    self.wellDF['ρ (g/cm3)'] = 2.75 - 2.11 * (self.wellDF['Δt (μs/ft)'] - self.dtmatrix) / (
                                self.wellDF['Δt (μs/ft)'] + 200)
                elif self.wellDF['Δt (μs/ft)'][i] < 100 and self.wellDF['Δt (μs/ft)'][i] != 0:
                    self.wellDF['ρ (g/cm3)'] = 3.28 - self.wellDF['Δt (μs/ft)'] / 88.95
        elif self.force_condition == 'consolidated':
            for i in range(len(self.wellDF.index)):
                self.wellDF['ρ (g/cm3)'] = 3.28 - self.wellDF['Δt (μs/ft)'] / 88.95
        elif self.force_condition == 'unconsolidated':
            self.wellDF['ρ (g/cm3)'] = 2.75 - 2.11 * (self.wellDF['Δt (μs/ft)'] - self.dtmatrix) / (
                        self.wellDF['Δt (μs/ft)'] + 200)
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
    wellDF : DataFrame
        Dataframe containing well information
    wellinfoDF : DataFrame
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

    def __init__(self, wellDF, wellinfoDF, name, water=None, sumwater=None, points=None):
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

    def plotporosprof(self):
        plt.plot(self.wellDF['prof (m)'], self.wellDF['Porosidade'], color='red', marker='o', ls='--')
        plt.ylabel('Porosidade')
        plt.xlabel('Profundidade [$m$]')
        plt.title('Porosidade versus profundidade')
        plt.grid()
        plt.savefig(f'output\\{self.name} - Porosidade versus profundidade.jpg', format='jpg', dpi=800)
        plt.show()

    def constants(self):
        self.phi0, self.K0 = classes.exponentialmodel(self.wellDF).export()
        self.K0 = self.K0 * (-1)
        self.phi0 = np.e ** self.phi0
        self.rho_seawater = self.wellinfoDF[1][0]
        self.rho_region = self.wellinfoDF[1][1]
        self.water_depth = self.wellinfoDF[1][2]
        if self.points is None:
            self.points = 100
        else:
            pass

    def density(self):
        return self.rho_region * (
                    1 - self.phi0 * np.e ** (-self.K0 * self.D)) + self.rho_seawater * self.phi0 * np.e ** (
                    -self.K0 * self.D)

    def tension(self):
        return 1.422 * (self.rho_seawater * self.water_depth + self.rho_region * self.D - (
                    (self.rho_region - self.rho_seawater) / self.K0) * self.phi0 * (1 - np.e ** (-self.K0 * self.D)))

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
        self.wellDF['Porosidade'] = self.wellDF['Porosidade'].fillna(0)
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
            raise NotImplementedError(f"Code doesn't get up there, wait updates...")
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
        self.plot()
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
        plt.axline((0, self.totalprofDF['new prof (m)'].max()), (10, self.totalprofDF['new prof (m)'].max()),
                   color='black', ls=':',
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


class Multiplot:
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

class NormalTensionandGrad:
    """
    Calculate the normal pore tension and the normal pore pressure gradient.
    Parameters
    ----------
    wellDF : pd.DataFrame
        Dataframe containing well information
    wellinfoDF : DataFrame
        Dataframe containing environment information
    name : str
        Parameter name (used for exporting .xlsx)
    sumwater : bool
        Make the class sum the water depth in the total depth
    """
    def __init__(self, wellDF, wellinfoDF, name, sumwater=None):
        self.wellDF = wellDF
        self.wellinfoDF = wellinfoDF
        self.name = name
        self.sumwater = sumwater
        self.tensionDF = None
        self.gradDF = None
        self.totalprofDF = None
        self.rho_seawater = None
        self.rho_region = None
        self.water_depth = None

    @staticmethod
    def tension(x, y):
        return 1.422 * x * y

    @staticmethod
    def grad(tensi, depth):
        return tensi / (0.1704 * (depth))

    def start(self):
        # Removing NaN entries #
        self.wellDF = self.wellDF.fillna(0)

        # Data maniputalion #
        self.tensionDF = pd.DataFrame(np.zeros(len(self.wellDF.index)))
        self.gradDF = pd.DataFrame(np.zeros(len(self.wellDF.index)))
        self.totalprofDF = pd.DataFrame(np.zeros(len(self.wellDF.index)))
        if self.sumwater == True:
            self.totalprofDF = self.wellDF['prof (m)'] + self.wellinfoDF[2]['LÂMINA DÁGUA (m):']
        else:
            self.totalprofDF = self.wellDF['prof (m)']
        # Extracting constants #
        self.rho_seawater = self.wellinfoDF[2][0]
        self.rho_region = self.wellinfoDF[2][1]
        self.water_depth = self.wellinfoDF[2][2]

        for i in range(len(self.wellDF.index)):
            if i == 0:
                self.tensionDF[0][i] = self.tension(self.water_depth, self.rho_seawater)
                self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])
            else:
                self.tensionDF[0][i] = self.tensionDF[0][i - 1] + self.tension(self.wellDF['ΔD (m)'][i], self.rho_seawater)
                self.gradDF[0][i] = self.grad(self.tensionDF[0][i], self.totalprofDF[i])


        self.tensionDF.columns = ['Normal Pore Tension']
        self.gradDF.columns = ['Normal Pore Pressure Gradient']
        wellDF = pd.concat([self.wellDF, self.tensionDF], axis=1)

        wellDF = pd.concat([wellDF, self.gradDF], axis=1)
        wellDF.to_excel(f'output\\{self.name}.xlsx')
        self.wellDF = wellDF
        self.plot()
        return self.wellDF, self.wellinfoDF, self.tensionDF, self.gradDF, self.totalprofDF, self.water_depth

    def plot(self):

        # Cleaning zero values for Tension and overburden #
        self.tensionDF['Normal Pore Tension'] = self.tensionDF['Normal Pore Tension'].loc[(self.tensionDF['Normal Pore Tension'] != 0)]
        self.gradDF['Normal Pore Pressure Gradient'] = self.gradDF['Normal Pore Pressure Gradient'].loc[(self.gradDF['Normal Pore Pressure Gradient'] != 0)]

        plt.plot(self.tensionDF, self.wellDF['prof (m)'], color='C12', marker='o', ls='--')
        plt.xlabel('Tensão [$psi$]')
        plt.ylabel('Profundidade [$m$]')
        plt.title('Pressão de poros normal $versus$ profundidade')
        plt.grid()
        plt.ylim([0, self.wellDF['prof (m)'].max()])
        plt.gca().invert_yaxis()
        plt.savefig(f'output\\{self.name} - Pressão de poros normal.jpg', format='jpg', dpi=800)
        plt.show()

        plt.plot(self.gradDF, self.totalprofDF, color='purple', marker='o', ls='--')
        plt.axline((0, self.totalprofDF.max()), (1, self.totalprofDF.max()), color='black', ls=':',
                   label=f'Profun máx = {self.totalprofDF.max()} $m$')
        plt.axline((0, self.water_depth), (1, self.water_depth), color='blue', ls=':',
                   label=f'Lâmi dágua = {self.water_depth} $m$')
        plt.xlabel('Gradiente de pressão de poros normal [$lb/gal$]')
        plt.ylabel('Profundidade [$m$]')
        plt.ylim([0, self.totalprofDF.max()])
        plt.title('Gradiente de pressão de poros normal $versus$ Profundidade')
        plt.legend(loc='best')
        plt.gca().invert_yaxis()
        plt.grid()
        plt.savefig(f'output\\{self.name} - Gradiente de pressão de poros normal.jpg', format='jpg', dpi=800)
        plt.show()


class Eaton:
    """
        Calculate the tension and the pore pressure gradient, then plot graphics and export the found values in a .xlsx file

        Parameters
        ----------
        wellDF: dict or Any
            Dataframe containing well information.
        name : str
            Parameter name (used for exporting .xlsx).

        top : int
            Point belonging to the top of the underpressurized zone.
        sumwater : [0,1]
            Set if it will sum the water layer.
    """
    def __init__(self, wellDF, wellinfoDF, name, top, exponum, sumwater=None):
        self.wellDF = wellDF
        self.wellinfoDF = wellinfoDF
        self.name = name
        self.top = top
        self.exponum = exponum
        self.sumwater = sumwater
        self.gradDF = None
        self.tensionDF = None


    @staticmethod
    def f(x, a, b):
        return (x-a)/b

    @staticmethod
    def funct(x, a, b):
        return (x - b) / a

    def start(self):
        # Removing NaN entries #
        self.wellDF = self.wellDF.fillna(0)

        # Data maniputalion #
        self.tensionDF = pd.DataFrame(np.zeros(len(self.wellDF.index)))
        self.gradDF = pd.DataFrame(np.zeros(len(self.wellDF.index)))
        self.normaltransit = pd.DataFrame(np.zeros(len(self.wellDF.index)))
        self.totalprofDF = pd.DataFrame(np.zeros(len(self.wellDF.index)))
        self.water_depth = self.wellinfoDF[2][2]
        if self.sumwater == True:
            self.totalprofDF = self.wellDF['prof (m)'] + self.wellinfoDF[2]['LÂMINA DÁGUA (m):']
        else:
            self.totalprofDF = self.wellDF['prof (m)']

        a, b = classes.loglinmodel(self.wellDF, top=self.top).export() # antes
        self.normaltransit['Δtn (μs/ft)'] = np.exp(self.f(self.wellDF['prof (m)'], a, b))  # antes

        #a, b = classes.autoexpmodellnx(self.wellDF, sumwater=False).export()

        # a, b = np.polyfit(self.wellDF['Δt (μs/ft)'][1:self.top], self.wellDF['prof (m)'][1:self.top], 1)
        #self.normaltransit['Δtn (μs/ft)'] = self.funct(self.wellDF['prof (m)'], a, b)

        if self.sumwater == 1:
            self.normaltransit['Δtn (μs/ft)'][0] = 0

        for i in range(len(self.wellDF.index)):
            if i==0:
                self.gradDF[0][i] = np.nan
            else:
                self.gradDF[0][i] = self.wellDF['Overburden'][i] - (self.wellDF['Overburden'][i] -
                self.wellDF['Normal Pore Pressure Gradient'][i])*(self.normaltransit['Δtn (μs/ft)'][i]/self.wellDF['Δt (μs/ft)'][i])**self.exponum

        self.gradDF.columns = ['Pore pressure Gradient']
        wellDF = pd.concat([self.wellDF, self.gradDF], axis=1)
        wellDF.to_excel(f'output\\{self.name}.xlsx')
        self.wellDF = wellDF
        self.plot()
        resp = input('Deseja testar outro expoente? ')
        if resp == 'sim' or resp == 'Sim' or resp == 's' or resp == 'S':
            self.exponum = float(input('Novo valor do expoente: '))
            self.start()
        return self.wellDF, self.top

    def plot(self):

        # Cleaning zero values for Tension and overburden #
        #self.gradDF['Overburden'] = self.gradDF['Overburden'].loc[(self.gradDF['Overburden'] != 0)]
        plt.plot(self.gradDF, self.totalprofDF, color='green', marker='o', ls='--', label='Gradiente de pressão de poros')
        plt.axline((0, self.totalprofDF.max()), (1, self.totalprofDF.max()), color='black', ls=':',
                   label=f'Profun máx = {self.totalprofDF.max()} $m$')
        plt.axline((0, self.water_depth), (1, self.water_depth), color='blue', ls=':',
                   label=f'Lâmi dágua = {self.water_depth} $m$')
        plt.axline((0, self.wellDF['prof (m)'][self.top]), (1, self.wellDF['prof (m)'][self.top]), color='orange', ls=':',
            label=f"Topo da zona superpressurizada = {self.wellDF['prof (m)'][self.top]} $m$")
        plt.plot([], [], ' ', label=f"Expoente: {self.exponum:.3f}")
        plt.xlabel('Gradientes de pressão de poros [$lb/gal$]')
        plt.ylabel('Profundidade [$m$]')
        plt.ylim([0, self.totalprofDF.max()])
        plt.title('Gradiente de pressão de poros $versus$ Profundidade')
        plt.legend(loc='best')
        plt.gca().invert_yaxis()
        plt.grid()
        plt.savefig(f'output\\{self.name} - Gradiente de pressão de poros.jpg', format='jpg', dpi=800)
        plt.show()

        # Cleaning zero values for Tension and overburden #
        #self.gradDF['Overburden'] = self.gradDF['Overburden'].loc[(self.gradDF['Overburden'] != 0)]
        plt.plot(self.gradDF, self.totalprofDF, color='green', marker='o', ls='--', label='Gradiente de pressão de poros')
        plt.plot(self.wellDF['Overburden'], self.totalprofDF, color='red', marker='o', ls='--', label='Gradiente de sobrecarga')
        plt.plot(self.wellDF['Normal Pore Pressure Gradient'], self.totalprofDF, color='blue', marker='o', ls='--',
                    label='Gradiente de pressão de poros normal')
        plt.axline((0, self.totalprofDF.max()), (1, self.totalprofDF.max()), color='black', ls=':',
                   label=f'Profun máx = {self.totalprofDF.max()} $m$')
        plt.axline((0, self.water_depth), (1, self.water_depth), color='blue', ls=':',
                   label=f'Lâmi dágua = {self.water_depth} $m$')
        plt.axline((0, self.wellDF['prof (m)'][self.top]), (1, self.wellDF['prof (m)'][self.top]), color='orange', ls=':',
            label=f"Topo da zona superpressurizada = {self.wellDF['prof (m)'][self.top]} $m$")
        plt.plot([], [], ' ', label=f"Expoente: {self.exponum:.3f}")
        plt.xlabel('Gradientes de pressões [$lb/gal$]')
        plt.ylabel('Profundidade [$m$]')
        plt.ylim([0, self.totalprofDF.max()])
        plt.title('Gradientes de pressões $versus$ Profundidade')
        plt.legend(loc='best')
        plt.gca().invert_yaxis()
        plt.grid()
        plt.savefig(f'output\\{self.name} - Gradientes de pressões.jpg', format='jpg', dpi=800)
        plt.show()

class Porepressure:
    def __init__(self, wellDF):
        self.wellDF = wellDF
        self.porepressure()

    @staticmethod
    def tension(grad, depth):
        return grad * 0.1704 * depth

    def porepressure(self):
        self.porepress = pd.DataFrame(np.zeros(len(self.wellDF.index)))
        self.porepress.columns = ['Pore Pressure']
        for i in range(len(self.wellDF.index)):
            self.porepress['Pore Pressure'][i] = self.tension(self.wellDF['Pore pressure Gradient'][i], self.wellDF['prof (m)'][i])

        self.wellDF = pd.concat([self.wellDF, self.porepress], axis=1)
    def output(self):
        return self.wellDF

class Unconfinedcompressforce:
    def __init__(self, wellDF):
        self.wellDF = wellDF
        self.rockcohesion()

    def rockcohesion(self):
        self.Co = pd.DataFrame(np.zeros(len(self.wellDF.index)))
        self.Co.columns = ['Unconfined compress force (Co)']

        for i in range(len(self.wellDF.index)):
            self.Co['Unconfined compress force (Co)'][i] = (2*self.wellDF['Coesao(psi)'][i]*(np.cos(self.wellDF['angulo_atrito_interno'][i]*np.pi/180))
                                /(1-np.sin(self.wellDF['angulo_atrito_interno'][i]*np.pi/180)))

        self.wellDF = pd.concat([self.wellDF, self.Co], axis=1)

    def output(self):
        return self.wellDF


class Hydrostaticpressure:
    def __init__(self, wellDF):
        self.wellDF = wellDF
        self.hidrostaticpress = pd.DataFrame(np.zeros(len(self.wellDF.index)))
        self.hidrostaticpress.columns = ['Hydrostatic Pressure']
        self.hydrostaticpressure()

    def hydrostaticpressure(self):
        for i in range(len(self.wellDF.index)):
            angle = np.pi/4+self.wellDF['angulo_atrito_interno'][i]/2*(np.pi/180)
            A = 3*self.wellDF['TH'][i]-self.wellDF['Th'][i]-self.wellDF['Unconfined compress force (Co)'][i]+self.wellDF['Pore Pressure'][i]*(np.tan(angle)**2-1)
            B = np.tan(angle)**2+1
            self.hidrostaticpress['Hydrostatic Pressure'][i] = A/B

        self.wellDF = pd.concat([self.wellDF, self.hidrostaticpress], axis=1)

    def output(self):
        return self.wellDF

class CollapseGradient:
    """
    Calculate the collapse gradient of hydrostatic pressure
    Parameters
    ------
    wellDF: dict or Any
        Dataframe containing well information.
    """
    def __init__(self, wellDF):
        self.wellDF = wellDF
        self.collapsegradientdf = pd.DataFrame(np.zeros(len(self.wellDF.index)))
        self.collapsegradientdf.columns = ['Collapse Gradient']
        self.collapsegradient()

    @staticmethod
    def grad(tensi, depth):
        return tensi / (0.1704 * (depth))

    def collapsegradient(self):
        for i in range(len(self.wellDF.index)):
            self.collapsegradientdf['Collapse Gradient'][i] = self.grad(self.wellDF['Hydrostatic Pressure'][i], self.wellDF['prof (m)'][i])
        self.wellDF = pd.concat([self.wellDF, self.collapsegradientdf], axis=1)

    def output(self):
        return self.wellDF

class FractureGradient:
    """
    Calculate the fracture gradient

    Parameters
        ----------
        wellDF: dict or Any
            Dataframe containing well information.
    """
    def __init__(self, wellDF):
        self.wellDF = wellDF
        self.fractgrad = pd.DataFrame(np.zeros(len(self.wellDF.index)))
        self.fractgrad.columns = ['Fracture Gradient']
        self.fracturegradient()

    def fracturegradient(self):
        for i in range(len(self.wellDF.index)):
            K = self.wellDF['Poisson'][i]/(1-self.wellDF['Poisson'][i])
            self.fractgrad['Fracture Gradient'][i] = K*(self.wellDF['Overburden'][i]-self.wellDF['Pore pressure Gradient'][i])+self.wellDF['Pore pressure Gradient'][i]

        self.wellDF = pd.concat([self.wellDF, self.fractgrad], axis=1)

    def output(self):
        return self.wellDF

class Mudweightwindow:
    """
    Print and save the figure containing the mug weight window

    Parameters
        ----------
        wellDF: dict or Any
            Dataframe containing well information.
        name : str
            Parameter name (used for exporting .png).
        top : float
            Top of super pressurized zone
        probezone : float
            Probe zone where Mohr circle will be made from top zone
    """
    def __init__(self, wellDF, name, top=None, probezone=None):
        self.wellDF = wellDF
        self.name = name
        self.top = top
        if probezone is not None:
            self.probezone = probezone
        else:
            self.probezone = 2
        self.print()
        self.mohrcircle()

    def print(self):
        plt.plot(self.wellDF['Pore pressure Gradient'], self.wellDF['prof (m)'], color='green', marker='o', ls='--',
                 label='Gradiente de pressão de poros', lw=0.5)
        plt.plot(self.wellDF['Overburden'], self.wellDF['prof (m)'], color='red', marker='o', ls='--',
                 label='Gradiente de sobrecarga', lw=0.5)
        plt.plot(self.wellDF['Collapse Gradient'], self.wellDF['prof (m)'], color='orange', marker='o', ls='--',
                 label='Gradiente de colapso', lw=0.5)
        plt.plot(self.wellDF['Fracture Gradient'], self.wellDF['prof (m)'], color='purple', marker='o', ls='--',
                 label='Gradiente de fratura', lw=0.5)
        plt.fill_betweenx(self.wellDF['prof (m)'], self.wellDF['Collapse Gradient'], self.wellDF['Fracture Gradient'],
                          where=(self.wellDF['Fracture Gradient'] > self.wellDF['Collapse Gradient']), color='yellow', alpha=0.8, label='Janela Operacional')
        plt.fill_betweenx(self.wellDF['prof (m)'], self.wellDF['Collapse Gradient'], self.wellDF['Pore pressure Gradient'], color='white')
        plt.axline((0, self.wellDF['prof (m)'].max()), (1, self.wellDF['prof (m)'].max()), color='black', ls=':',
                   label=f'Profun máx = {self.wellDF["prof (m)"].max()} $m$')
        plt.axline((0, self.wellDF['prof (m)'].min()), (1, self.wellDF['prof (m)'].min()), color='blue', ls=':',
                   label=f'Lâmi dágua = {self.wellDF["prof (m)"].min()} $m$')
        if self.top is not None:
            plt.axline((0, self.wellDF['prof (m)'][self.top]), (1, self.wellDF['prof (m)'][self.top]), color='orange',
                       ls=':',label=f"Topo da zona superpressurizada = {self.wellDF['prof (m)'][self.top]} $m$")
        plt.xlabel('Gradientes de pressões [$lb/gal$]')
        plt.ylabel('Profundidade [$m$]')
        plt.ylim([0, self.wellDF['prof (m)'].max()])
        plt.title('Janela Operacional')
        plt.legend(loc='best')
        plt.gca().invert_yaxis()
        plt.grid()
        plt.savefig(f'output\\{self.name} - Mugweightwindow.jpg', format='jpg', dpi=800)
        plt.show()
        self.wellDF.to_excel(f'output\\{self.name}.xlsx')

    def mohrcircle(self):
        supzone = self.top-self.probezone
        infzone = self.top+self.probezone
        zones = [supzone, infzone]
        plt.plot(self.wellDF['Pore pressure Gradient'], self.wellDF['prof (m)'], color='green', marker='o', ls='--',
                 label='Gradiente de pressão de poros', lw=0.5)
        plt.plot(self.wellDF['Overburden'], self.wellDF['prof (m)'], color='red', marker='o', ls='--',
                 label='Gradiente de sobrecarga', lw=0.5)
        plt.plot(self.wellDF['Collapse Gradient'], self.wellDF['prof (m)'], color='orange', marker='o', ls='--',
                 label='Gradiente de colapso', lw=0.5)
        plt.plot(self.wellDF['Fracture Gradient'], self.wellDF['prof (m)'], color='purple', marker='o', ls='--',
                 label='Gradiente de fratura', lw=0.5)
        plt.fill_betweenx(self.wellDF['prof (m)'], self.wellDF['Collapse Gradient'], self.wellDF['Fracture Gradient'],
                          where=(self.wellDF['Fracture Gradient'] > self.wellDF['Collapse Gradient']), color='yellow', alpha=0.8, label='Janela Operacional')
        plt.fill_betweenx(self.wellDF['prof (m)'], self.wellDF['Collapse Gradient'], self.wellDF['Pore pressure Gradient'], color='white')
        plt.axline((0, self.wellDF['prof (m)'].max()), (1, self.wellDF['prof (m)'].max()), color='black', ls=':',
                   label=f'Profun máx = {self.wellDF["prof (m)"].max()} $m$')
        plt.axline((0, self.wellDF['prof (m)'].min()), (1, self.wellDF['prof (m)'].min()), color='blue', ls=':',
                   label=f'Lâmi dágua = {self.wellDF["prof (m)"].min()} $m$')
        if self.top is not None:
            plt.axline((0, self.wellDF['prof (m)'][self.top]), (1, self.wellDF['prof (m)'][self.top]), color='orange',
                       ls=':',label=f"Topo da zona superpressurizada = {self.wellDF['prof (m)'][self.top]} $m$")
        for zone in zones:
            plt.scatter(y=self.wellDF['prof (m)'][zone], x=self.wellDF['Collapse Gradient'][zone], marker='x', s=125, color='red')
            plt.scatter(y=self.wellDF['prof (m)'][zone], x=self.wellDF['Collapse Gradient'][zone] * 1.1, marker='x', s=125, color='blue')
            plt.scatter(y=self.wellDF['prof (m)'][zone], x=self.wellDF['Fracture Gradient'][zone] * 0.9, marker='x', s=125, color='green')
            plt.scatter(y=self.wellDF['prof (m)'][zone], x=self.wellDF['Fracture Gradient'][zone], marker='x', s=125, color='purple')
        plt.xlabel('Gradientes de pressões [$lb/gal$]')
        plt.ylabel('Profundidade [$m$]')
        plt.ylim([0, self.wellDF['prof (m)'].max()])
        plt.title('Janela Operacional')
        plt.legend(loc='best')
        plt.gca().invert_yaxis()
        plt.grid()
        plt.savefig(f'output\\{self.name} - Mugweightwindow Mohr positions.jpg', format='jpg', dpi=800)
        plt.show()

        for zone in zones:
            Gw = self.wellDF['Collapse Gradient'][zone]
            Pw = Gw * 0.1704 * self.wellDF['prof (m)'][zone]
            sigmar = Pw - self.wellDF['Pore Pressure'][zone]
            sigmathetamax = 3*self.wellDF['TH'][zone]-self.wellDF['Th'][zone] - Pw - self.wellDF['Pore Pressure'][zone]
            sigmathetamin = 3*self.wellDF['Th'][zone]-self.wellDF['TH'][zone] - Pw - self.wellDF['Pore Pressure'][zone]
            tension = [sigmar, sigmathetamax, sigmathetamin]
            Gw2 = self.wellDF['Collapse Gradient'][zone] * 1.10
            Pw2 = Gw2 * 0.1704 * self.wellDF['prof (m)'][zone]
            sigmar2 = Pw2 - self.wellDF['Pore Pressure'][zone]
            sigmathetamax2 = 3*self.wellDF['TH'][zone]-self.wellDF['Th'][zone] - Pw2 - self.wellDF['Pore Pressure'][zone]
            sigmathetamin2 = 3*self.wellDF['Th'][zone]-self.wellDF['TH'][zone] - Pw2 - self.wellDF['Pore Pressure'][zone]
            tension2 = [sigmar2, sigmathetamax2, sigmathetamin2]
            Gw3 = self.wellDF['Fracture Gradient'][zone] * 0.9
            Pw3 = Gw3 * 0.1704 * self.wellDF['prof (m)'][zone]
            sigmar3 = Pw3 - self.wellDF['Pore Pressure'][zone]
            sigmathetamax3 = 3 * self.wellDF['TH'][zone] - self.wellDF['Th'][zone] - Pw3 - self.wellDF['Pore Pressure'][zone]
            sigmathetamin3 = 3 * self.wellDF['Th'][zone] - self.wellDF['TH'][zone] - Pw3 - self.wellDF['Pore Pressure'][zone]
            tension3 = [sigmar3, sigmathetamax3, sigmathetamin3]
            Gw4 = self.wellDF['Fracture Gradient'][zone]
            Pw4 = Gw4 * 0.1704 * self.wellDF['prof (m)'][zone]
            sigmar4 = Pw4 - self.wellDF['Pore Pressure'][zone]
            sigmathetamax4 = 3 * self.wellDF['TH'][zone] - self.wellDF['Th'][zone] - Pw4 - self.wellDF['Pore Pressure'][zone]
            sigmathetamin4 = 3 * self.wellDF['Th'][zone] - self.wellDF['TH'][zone] - Pw4 - self.wellDF['Pore Pressure'][zone]
            tension4 = [sigmar4, sigmathetamax4, sigmathetamin4]
            index = ['Sigmar', 'sigmathetamax', 'sigmathetamin']
            data = {'Tensions' : index, 'Circle 1' : tension, 'Circle 2' : tension2, 'Circle 3' : tension3, 'Circle 4' : tension4}
            DataFrame = pd.DataFrame(data=data)
            DataFrame.to_excel(f'output\\{self.name} - Zone {zone}.xlsx')
            sigma1 = [max(tension) , max(tension2), max(tension3), max(tension4)]
            sigma1 = pd.Series(sigma1)
            sigma3 = [min(tension), min(tension2), min(tension3), min(tension4)]
            sigma3 = pd.Series(sigma3)
            MohrCircle(sigma1, sigma3, name=f'Círculo de Mohr - zona {zone} ({self.wellDF["prof (m)"][zone]}m)', S0=self.wellDF['Coesao(psi)'][zone], phi=self.wellDF['angulo_atrito_interno'][zone]*np.pi/180)

        """for zone in zones:
            Gw = self.wellDF['Collapse Gradient'][zone]
            Pw = Gw * 0.1704 * self.wellDF['prof (m)'][zone]
            sigmar = Pw - self.wellDF['Pore Pressure'][zone]
            sigmathetamax = 3*self.wellDF['TH'][zone]-self.wellDF['Th'][zone] - Pw - self.wellDF['Pore Pressure'][zone]
            sigmathetamin = 3*self.wellDF['Th'][zone]-self.wellDF['TH'][zone] - Pw - self.wellDF['Pore Pressure'][zone]
            if sigmathetamax-sigmar > sigmathetamin - sigmar:
                sigmatheta = sigmathetamax
            else:
                sigmatheta = sigmathetamin
            tension = [sigmar, sigmatheta]

            Gw2 = self.wellDF['Collapse Gradient'][zone] * 1.10
            Pw2 = Gw2 * 0.1704 * self.wellDF['prof (m)'][zone]
            sigmar2 = Pw2 - self.wellDF['Pore Pressure'][zone]
            sigmathetamax2 = 3*self.wellDF['TH'][zone]-self.wellDF['Th'][zone] - Pw2 - self.wellDF['Pore Pressure'][zone]
            sigmathetamin2 = 3*self.wellDF['Th'][zone]-self.wellDF['TH'][zone] - Pw2 - self.wellDF['Pore Pressure'][zone]

            if sigmathetamax2-sigmar2 > sigmathetamin2 - sigmar2:
                sigmatheta2 = sigmathetamax2
            else:
                sigmatheta2 = sigmathetamin2
            tension2 = [sigmar2, sigmatheta2]

            Gw3 = self.wellDF['Fracture Gradient'][zone] * 0.9
            Pw3 = Gw3 * 0.1704 * self.wellDF['prof (m)'][zone]
            sigmar3 = Pw3 - self.wellDF['Pore Pressure'][zone]
            sigmathetamax3 = 3 * self.wellDF['TH'][zone] - self.wellDF['Th'][zone] - Pw3 - self.wellDF['Pore Pressure'][zone]
            sigmathetamin3 = 3 * self.wellDF['Th'][zone] - self.wellDF['TH'][zone] - Pw3 - self.wellDF['Pore Pressure'][zone]

            if sigmathetamax3-sigmar3 > sigmathetamin3 - sigmar3:
                sigmatheta3 = sigmathetamax3
            else:
                sigmatheta3 = sigmathetamin3
            tension3 = [sigmar3, sigmatheta3]

            Gw4 = self.wellDF['Fracture Gradient'][zone]
            Pw4 = Gw4 * 0.1704 * self.wellDF['prof (m)'][zone]
            sigmar4 = Pw4 - self.wellDF['Pore Pressure'][zone]
            sigmathetamax4 = 3 * self.wellDF['TH'][zone] - self.wellDF['Th'][zone] - Pw4 - self.wellDF['Pore Pressure'][zone]
            sigmathetamin4 = 3 * self.wellDF['Th'][zone] - self.wellDF['TH'][zone] - Pw4 - self.wellDF['Pore Pressure'][zone]

            if sigmathetamax4-sigmar4 > sigmathetamin4 - sigmar4:
                sigmatheta4 = sigmathetamax4
            else:
                sigmatheta4 = sigmathetamin4
            tension4 = [sigmar4, sigmatheta4]

            index = ['Sigmar', 'sigmatheta']
            data = {'Tensions' : index, 'Circle 1' : tension, 'Circle 2' : tension2, 'Circle 3' : tension3, 'Circle 4' : tension4}
            DataFrame = pd.DataFrame(data=data)
            DataFrame.to_excel(f'output\\{self.name} - Zone {zone}.xlsx')
            sigma1 = [max(tension) , max(tension2), max(tension3), max(tension4)]
            sigma1 = pd.Series(sigma1)
            sigma3 = [min(tension), min(tension2), min(tension3), min(tension4)]
            sigma3 = pd.Series(sigma3)
            MohrCircle(sigma1, sigma3, name=f'Circulo de Mohr zona {zone}', S0=self.wellDF['Coesao(psi)'][zone], phi=self.wellDF['angulo_atrito_interno'][zone]*np.pi/180)"""


class MohrCircle:
    """
        Calculate and print a Mohr Circle given the sigma 1 and sigma 3 values

        Parameters
            ----------
            sigma1: DataFrame
                Dataframe containing sigma 1 values.
            sigma3 : DataFrame
                Dataframe containing sigma 3 values.
            name : str
                Name of the image that will be printed and saved.
            linemodel: bool
                 Defines whether or not the line model will be made.
        """
    def __init__(self, sigma1, sigma3, name=None, linemodel=None, S0=None, phi=None):
        #self.wellDF = wellDF
        #self.name = name
        if len(sigma1) != len(sigma3):
            raise Exception('Sigma1 and Sigma3 must have same length!')

        self.sigma1 = sigma1
        self.sigma3 = sigma3
        self.S0, self.phi = S0, phi
        self.failplaneangle, self.failplaneforces = None, None
        self.linemodel = linemodel
        self.name = name
        if self.linemodel is True:
            self.model()
            self.calculate()
        elif self.S0 is not None and self.phi is not None:
            self.calculate()
        else:
            pass
        self.plot()

    @staticmethod
    def mohrcoulombcriteria(x, a, b):
        return  np.tan(a)*x + b

    @staticmethod
    def sheartension(sigma1, sigma3, failplaneangle):
        return (sigma1-sigma3)/2*np.sin(2*failplaneangle)


    @staticmethod
    def normaltension(sigma1, sigma3, failplaneangle):
        return (sigma1+sigma3)/2 + (sigma1-sigma3)/2*np.cos(2*failplaneangle)

    def model(self):
        linear, angular = classes.linearmodel(self.sigma3, self.sigma1).export()
        self.phi = np.arcsin((angular-1)/(angular+1))
        self.S0 = linear/2*(1-np.sin(self.phi))/(np.cos(self.phi))

    def calculate(self):
        self.failplaneangle = 45 + self.phi/2 * 180/np.pi
        index = [f"Circle {i+1}" for i in range(len(self.sigma1))]
        columns = ["Normal tension", "Shear tension"]
        self.failplaneforces = pd.DataFrame(index=index, columns=columns)
        for i in range(len(self.sigma1)):
            self.failplaneforces["Normal tension"][f'Circle {i+1}'] = self.normaltension(self.sigma1[i], self.sigma3[i], self.failplaneangle*np.pi/180)
            self.failplaneforces["Shear tension"][f'Circle {i+1}'] = self.sheartension(self.sigma1[i], self.sigma3[i], self.failplaneangle*np.pi/180)


    def plot(self):
        color = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
        rd = []
        fig, ax = plt.subplots()
        for i in range(len(self.sigma1)):
            radius = (self.sigma1[i] - self.sigma3[i])/2
            x0 = radius + self.sigma3[i]
            y0 = 0
            circle = plt.Circle((x0, y0), radius, color=color[i], fill=False,
                                label=f'Corpo de prova {i}: σ₁={self.sigma1[i]:.3f}, σ₃ = {self.sigma3[i]:.3f}')
            ax.add_patch(circle)
            rd.append(abs(radius))
        ax.grid()
        ax.set_xlabel('Tensão [psi]')
        ax.set_ylabel('Cisalhamento [psi]')

        if self.sigma1.max() > self.sigma3.max() and self.sigma3.min() > 0:
            ax.set_xlim(0, self.sigma1.max() * 1.1)
            x = np.linspace(0, self.sigma1.max())
        elif self.sigma3.max() > self.sigma1.max() and self.sigma3.min() > 0:
            ax.set_xlim(0, self.sigma3.max() * 1.1)
            x = np.linspace(0, self.sigma3.max())
        elif self.sigma1.max() > self.sigma3.max() and self.sigma3.min() < 0:
            ax.set_xlim(self.sigma3.min() *1.1, self.sigma1.max() * 1.1)
            x = np.linspace(self.sigma3.min(), self.sigma1.max())
        else:
            ax.set_xlim(self.sigma3.min() *1.1, self.sigma3.max() * 1.1)
            x = np.linspace(self.sigma3.min(), self.sigma3.max())
        ax.set_ylim(-max(rd) * 1.1, max(rd) * 1.1)
        ax.axis('equal')
        if self.S0 is not None and self.phi is not None:
            ax.plot(x, self.mohrcoulombcriteria(x, self.phi, self.S0), color='black',
                    label=f"Modelo: y = {float(np.tan(self.phi)):.4f}x + {float(self.S0):.4f}")
        ax.legend()
        if self.name is None:
            ax.set_title('Círculo de Mohr')
            fig.savefig(f'output\\Círculo de Mohr.jpg', format='jpg', dpi=800)
        else:
            ax.set_title(self.name)
            fig.savefig(f'output\\{self.name}.jpg', format='jpg', dpi=800)

        plt.show()

    def export(self):
        return self.phi, self.S0, self.failplaneforces