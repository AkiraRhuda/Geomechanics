import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

def GaussIngenua(A, b):
    # Verifica se A é uma matriz quadrada
    m, n = A.shape
    if m != n:
        raise ValueError('A matriz A deve ser quadrada')

    nb = n + 1
    Aum = np.hstack((A, b.reshape(-1, 1)))

    # Eliminação progressiva
    for k in range(n - 1):
        for i in range(k + 1, n):
            fator = Aum[i, k] / Aum[k, k]
            Aum[i, k:nb] = Aum[i, k:nb] - fator * Aum[k, k:nb]

    # Substituição regressiva
    x = np.zeros(n)
    x[n - 1] = Aum[n - 1, nb - 1] / Aum[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (Aum[i, nb - 1] - np.dot(Aum[i, i + 1:n], x[i + 1:])) / Aum[i, i]

    return x

class exponentialmodel:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.fit()
            self.statistics()
            self.plot()
            self.export()

        def fit(self):
            #self.x = self.x.loc[(self.x!= 0)]
            self.lny = np.log(self.y)
            sum_xi = sum(self.x)
            sum_xi2 = sum(self.x ** 2)
            sum_xilnyi = sum(self.x * self.lny)
            sum_lnyi = sum(self.lny)

            n = len(self.x)
            matriz_coef = np.array([[n, sum_xi], [sum_xi, sum_xi2]])

            A = matriz_coef

            matriz_resp = np.array([[sum_lnyi], [sum_xilnyi]])

            b = matriz_resp

            a = GaussIngenua(A, b)
            self.Data = pd.DataFrame()
            self.Data['Coeficiente angular'] = [a[1]]
            self.Data['Coeficiente linear'] = [a[0]]

        @staticmethod
        def f(x, a):
            return float(a['Coeficiente linear']) + float(a['Coeficiente angular']) * x

        def statistics(self):

            # Estatisticas
            ymean = sum(self.lny)/len(self.lny)
            St = sum((self.lny-ymean)**2)
            Sr = sum((float(self.Data['Coeficiente linear'])-float(self.Data['Coeficiente angular'])*self.x)**2)
            self.R2 = (St - Sr)/St # Coeficiente de correlacao
            self.Syx = (Sr/(len(self.lny)-2))**(1/2) # Erro padrao de estimativa

        def plot(self):

            xt = np.linspace(min(self.x), max(self.x), 30)
            plt.scatter(self.x, np.log(self.y))
            plt.title('Grafico do modelo exponencial')
            plt.plot(xt, self.f(xt, self.Data), color='green', label=f'Reta: ln(y) = {float(self.Data['Coeficiente angular']):.4f}x + {float(self.Data['Coeficiente linear']):.4f}')
            plt.plot([], [], ' ', label=f"Coeficiente de correlacao: {self.R2:.4f}")
            plt.plot([], [], ' ', label=f"Erro padrao de estimativa: {self.Syx:.4f}")
            plt.ylabel('$ln(y)$')
            plt.xlabel('$x$')
            plt.legend(loc='best')
            plt.grid()
            plt.savefig(f'output\\Exponential Model.jpg', format='jpg', dpi=800)
            plt.show()

        def export(self):
            return np.array([float(self.Data['Coeficiente linear']), float(self.Data['Coeficiente angular'])])



