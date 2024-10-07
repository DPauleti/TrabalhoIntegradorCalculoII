from Carteira import CARTEIRA

if __name__ == '__main__':
    Cart = CARTEIRA(stocks=['PETR4.SA', 'VALE3.SA', 'SUZB3.SA', 'BBAS3.SA', 'GOLD11.SA'], 
                      start='2021-03-26',
                      end='2024-03-26')
    Cart.run()