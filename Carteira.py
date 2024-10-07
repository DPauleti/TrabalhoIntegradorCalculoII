# Importação de módulos e bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sp
import yfinance as yfin
import datetime as dt
import math
import cvxopt as opt
import cvxpy as cp
from bcb import sgs
from datetime import date

opt.solvers.options['show_progress'] = False
np.set_printoptions(suppress=True) # Regra para o numpy deixar de imprimir o número em notação científica

class CARTEIRA:
    def __init__(self, stocks: list, start: str, end: str) -> None:
        self.__chosenStocks = stocks
        self.__startDate = start
        self.__endDate = end

    def prepare(self) -> pd.DataFrame:
        stocksClose = yfin.download(self.__chosenStocks, self.__startDate, self.__endDate)['Adj Close']

        stocksClose = stocksClose.dropna(axis=0, how='any')

        return stocksClose

    def returnYearly(self) -> pd.Series:
        stocksClose = self.prepare()
        logReturnDaily = (pd.DataFrame(stocksClose.shift(periods=-1) / stocksClose)).dropna()
        logReturnDaily.apply(lambda x: np.log(x))

        # Para visualização:
        returnDaily = logReturnDaily.mean()
        returnYearly = returnDaily * 252
        return logReturnDaily
    
    def print1(self):
        stocksClose = self.prepare()
        logReturnDaily = (pd.DataFrame(stocksClose.shift(periods=-1) / stocksClose)).dropna()
        logReturnDaily.apply(lambda x: np.log(x))

        # Para visualização:
        returnDaily = logReturnDaily.mean()
        returnYearly = returnDaily * 252
        print("Retorno Diário:")
        print(logReturnDaily)
        print("Média diária:")
        print(returnDaily)
        print("Média anual:")
        print(returnYearly)

    def print2(self):
        logReturnDaily = self.returnYearly()
        volatilityDaily = logReturnDaily.std()
        volatilityYearly = volatilityDaily * math.sqrt(252)

        print('Percentil 95')
        for c in logReturnDaily.columns:
            columnPerc = np.percentile(logReturnDaily[c], 95)
            print(c, columnPerc)
        print('Percentil 99')
        for c in logReturnDaily.columns:
            columnPerc = np.percentile(logReturnDaily[c], 99)
            print(c, columnPerc)
        
        print("Volatilidade diária:")
        print(volatilityDaily)
        print("Volatilidade anual:")
        print(volatilityYearly)



    def createMatrix(self, gold, printMatrix) -> np.array:
        logReturnDaily = self.returnYearly()

        #Para visualização
        volatilityDaily = logReturnDaily.std()
        volatilityYearly = volatilityDaily * math.sqrt(252)

        #print('Percentil 95')
        for c in logReturnDaily.columns:
            columnPerc = np.percentile(logReturnDaily[c], 95)
            # print(c, columnPerc)
        #print('Percentil 99')
        for c in logReturnDaily.columns:
            columnPerc = np.percentile(logReturnDaily[c], 99)
            # print(c, columnPerc)

        if gold == True:
            covarVariance = np.cov([logReturnDaily['BBAS3.SA'], logReturnDaily['GOLD11.SA'], logReturnDaily['PETR4.SA'], logReturnDaily['SUZB3.SA'],  logReturnDaily['VALE3.SA']]) # Numpy cria matriz de covariância
            onesMatrix = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]) # Criar matriz com 1 nas diagonais
        else:
            covarVariance = np.cov([logReturnDaily['BBAS3.SA'], logReturnDaily['PETR4.SA'], logReturnDaily['SUZB3.SA'], logReturnDaily['VALE3.SA']])            
            onesMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) # Criar matriz com 1 nas diagonais

        np.copyto(covarVariance, onesMatrix, where=(onesMatrix == 1)) # Copiar 1 para a matriz de covariâncias, pois essa usa as varianças nas diagonais

        if printMatrix:
            maxColSize = max(len(col) for col in logReturnDaily) # Definir máximo de caracteres no nome das colunas

            print(' ' * (maxColSize + 4), end='') # Imprime espaços antes dos labels das colunas
                                        # "end=" define como deve terminar o print
                                        # O caractér vazio serve para impedir o Python de pular uma linha automaticamente

            for col in logReturnDaily: # Iterar pelas colunas da matriz e imprimir o nome delas mais espaços o suficiente para alinhar com a matriz de covariâncias
                colSize = len(col)
                print(col + ' ' * (14 - colSize), end='') 

            print() # Print vazio para pular uma linha após o "for" acima
            loopCount = 0 

            for col in logReturnDaily: # Iterar pelas colunas da matriz
                colSize = len(col)
                print(col + ' ' * (maxColSize - colSize + 1), end='')
                for value in covarVariance[loopCount]:
                    print(f'{value:13.8f}', end=' ') # Imprimir valor com tamanho e casas decimais fixas
                print()
                loopCount += 1

        return covarVariance
    
    def randWeights(self, n):
        k = np.random.rand(n) # Gerar pesos aleatórios na quantidade especificada
        return k / sum(k)

    def riskReturn(self, covar, returns, weights=[]) -> None:
        r = np.asmatrix(returns) # Converter retornos em matriz
        w = np.asmatrix(weights)
        if len(weights) == 0:
            w = np.asmatrix(self.randWeights(len(returns))) # Gerar pesos aleatórios se pesos não forem fornecidos
        
        mu = w * r.T # Calcular retorno
        sigma = np.sqrt(w * covar * w.T) # Calcular risco

        if sigma > 2: # Remover outliers
            return self.randWeights(5)

        return mu, sigma
    
    def plotReturns(self, n, returns):
        covar = self.createMatrix(True, False) # Gerar matriz de covariância
        means, stds = np.column_stack([
            self.riskReturn(covar, returns) # Rodar a função acima o número de vezes especificado, guardando outputs como pontos
            for _ in range(n)
        ])

        fig = plt.figure()
        plt.plot(stds, means, 'o', markersize=5) # Colocar pontos num gráfico
        plt.xlabel('Risco')
        plt.ylabel('Retorno')
        plt.show()

    def minRisk(self, covar, retornos, targetReturn):
        n = len(retornos)
        retornos = np.array(retornos)
        covar = np.array(covar)
        
        w = cp.Variable(n) # Definir variavel a minimizar

        portfolioVariance = cp.quad_form(w, covar)
        objective = cp.Minimize(portfolioVariance)

        constraints = [
            cp.sum(w) == 1,
            w @ retornos == targetReturn,
            w >= 0,
        ]            

        problem = cp.Problem(objective, constraints)
        
        problem.solve()

        if problem.status != cp.OPTIMAL:
            raise ValueError("No optimal solution found")
        
        return w.value

    def weightsTable(self, covar, retornos):
        returns, risks = np.column_stack([self.riskReturn(covar, retornos) for _ in range(1000)]) # Rodar a função acima o número de vezes especificado, guardando outputs como pontos

        weights = []

        for i in returns:
            weight = self.minRisk(covar, retornos, i)
            weights.append(weight)
        
        returns = np.asarray(returns).flatten()
        risks = np.asarray(risks).flatten()

        dfWeights = pd.DataFrame({'Risks' : np.asarray(risks), 'Returns' : np.asarray(returns), 'Weights' : weights})
        print(dfWeights)
        
    def efficientFrontier(self, covar, retornos):
        n = len(retornos)
        returns = np.asmatrix(retornos)

        N = 100
        mus = [10**(5.0 * t/N - 1.0) for t in range(N)]


        S = opt.matrix(covar)
        pbar = opt.matrix(returns)


        G = -opt.matrix(np.eye(n))
        h = opt.matrix(0.0, (n ,1))
        A = opt.matrix(1.0, (1, n))
        b = opt.matrix(1.0)

        portfolios = []

        

        for mu in mus:
            P = mu * S
            P = opt.matrix(P, (n, n), 'd')  # Converter as matrizes aos tipos e tamanhos certos
            q = -pbar  
            q = opt.matrix(q, (5, 1), 'd')
            solution = opt.solvers.qp(P, q, G, h, A, b)
            portfolios.append(solution['x'])

        returns = [opt.blas.dot(pbar, x) for x in portfolios]
        risks = [np.sqrt(opt.blas.dot(x, S*x)) for x in portfolios]

        m1 = np.polyfit(returns, risks, 2)
        x1 = np.sqrt(m1[2] / m1[0])

        P = opt.matrix(P, (n, n), 'd')
        q = -pbar
        q = opt.matrix(q, (5, 1), 'd')

        wt = opt.solvers.qp(opt.matrix(x1 * S), q, G, h, A, b)['x']
        return np.asarray(wt), returns, risks
    
    def plotFrontier(self, n, retornos, sharpeLine):
        covar = self.createMatrix(True, False) # Gerar matriz de covariância
        means, stds = np.column_stack([
            self.riskReturn(covar, retornos) # Rodar a função acima o número de vezes especificado, guardando outputs como pontos
            for _ in range(n)
        ])

        fig = plt.figure()
        plt.plot(stds, means, 'o', markersize=5) # Colocar pontos num gráfico
        plt.xlabel('Risco')
        plt.ylabel('Retorno')
        weights, returns, risks = self.efficientFrontier(covar, retornos)
        
        plt.plot(risks, returns, color="red")

        if sharpeLine:
            selic = sgs.get(('selic', 432), start = date.today()) # Obter taxa Selic atual como dataframe
            selic = selic.iloc[0]['selic'] # Extrair o valor da taxa da dataframe
            selic = selic / 100

            
            sharpe = self.sharpeCalc(covar, retornos)['Sharpe']
            sharpeReturn = []
            xRF = np.linspace(0, 0.7, 400)
            for i in xRF:
                sharpeReturn.append(selic + sharpe * i)
        
            plt.plot(xRF, sharpeReturn, color='green')


        plt.show()
    
    def sharpeCalc(self, covar, retornos):
        selic = sgs.get(('selic', 432), start = date.today()) # Obter taxa Selic atual como dataframe
        selic = selic.iloc[0]['selic'] # Extrair o valor da taxa da dataframe
        selic = selic / 100

        weights, returns, risks = self.efficientFrontier(covar, retornos)
        dfRiskReturn = pd.DataFrame({'Risks' : risks, 'Returns' : returns})
        sharpeList = []
        for index, row in dfRiskReturn.iterrows(): # Calcular Sharpe para cada ponto na fronteira
            risk = row['Risks']
            ret = row['Returns']
            sharpe = (ret - selic) / risk
            sharpeList.append(sharpe)
        
        dfRiskReturn.insert(1, 'Sharpe', sharpeList)
        

        return dfRiskReturn.loc[dfRiskReturn['Sharpe'].idxmax()]
    
    def rfPortfolio(self, covar, retornos):
        risk = self.sharpeCalc(covar, retornos)['Risks']
        targetRisk = 0.2

        portfolioWeight = targetRisk / risk
        rfWeight = 1 - portfolioWeight

        print('Carteira: ', portfolioWeight)
        print('RF: ', rfWeight)



    
    def run(self):
        print('Rodando...')
        returns = [0.666, 0.374, 0.467, 0.411, 0.254]
        covar = self.createMatrix(True, False)
        self.print1()
        self.print2()
        self.createMatrix(True, True)
        self.plotReturns(1000, returns)
        print(self.sharpeCalc(covar, returns))
        print(self.minRisk(covar, returns, self.sharpeCalc(covar, returns)['Returns']))
        self.plotFrontier(1000, returns, True)
        self.weightsTable(covar, returns)
        self.rfPortfolio(covar, returns)

        # Oi mundo