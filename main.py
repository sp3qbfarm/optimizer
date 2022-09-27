# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Portfolio takes in a number of stocks and optimizes on sharpe
import sys
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string
from datetime import datetime
from dateutil.relativedelta import relativedelta
from datetime import timedelta
import statistics
from time import perf_counter
from scipy.optimize import minimize


class Asset:
    def __init__(self, underlying, quantity):
        self.underlying = underlying
        self.quantity = quantity
        self.data = []

    def get_underlying(self):
        return self.underlying

    def get_quantity(self):
        return self.quantity

    def get_data(self):
        return self.data


class Stock(Asset):
    def __init__(self, underlying, quantity):
        super().__init__(underlying, quantity)
        self.data = []

    # finding the price from the loaddata function
    def loadstockdata(self):
        # getting the data from Yahoo finance
        self.data = yf.Ticker(self.underlying)

    def find_bidask(self):
        try:
            bid = self.data.info['bid']
            ask = self.data.info['ask']
            return bid, ask
        except:
            pass

    def get_type(self):
        return "Stock"


def returns(data):
    rets = pd.Series()
    prev = data.shift(1)
    rets = (data / prev) - 1
    return rets


class Portfolio:
    def __init__(self):
        self.stock = []
        self.prices = {}
        self.data = {}
        self.mktval = 0
        self.pxhist = pd.DataFrame()

    def add_position(self, position, check=False):
        # check looks in the dataholder class
        if check == False:
            self.stock.append(position)
            position.loadstockdata()
            self.prices[position.get_underlying()] = position.find_bidask()[1]
            self.data[position] = position.get_data()
            self.mktval += self.prices[position.get_underlying()] * position.get_quantity()
            self.pxhist = pd.DataFrame(columns=[position.get_underlying()],
                                       index=position.get_data().history('max').index)
            self.pxhist[position.get_underlying()] = position.history('max')['Close']
        else:
            self.stock.append(position)
            self.data[position] = check.alldata[position.get_underlying()]
            self.prices[position.get_underlying()] = check.allasks[position.get_underlying()]
            self.mktval += self.prices[position.get_underlying()] * position.get_quantity()
            self.pxhist = check.pxhist

    def returndata(self, yrs):
        # collect the stock data, here close is actually the adjusted close. yrsel is actual yrs elapsed
        # (in case yrs goes out of bounds)
        stockdata = pd.DataFrame(
            columns=['yrsel', 'price', 'Ticker', str(yrs) + 'yrVol', str(yrs) + 'yrRet', 'quantity'])
        streams = pd.DataFrame()
        for i in self.stock:
            streams[i.get_underlying()] = self.data[i].history(str(yrs) + 'y')['Close']
            yrsel = len(self.data[i].history(str(yrs) + 'y')['Close']) / 252
            temp = {'yrsel': yrsel,
                    'price': self.prices[i.get_underlying()],
                    'Ticker': i.get_underlying(),
                    str(yrs) + 'yrVol': np.std(returns(self.data[i].history(str(yrs) + 'y')['Close'])) * np.sqrt(252),
                    str(yrs) + 'yrRet': streams[i.get_underlying()][-1] / streams[i.get_underlying()][0] ** (
                            1 / yrsel) - 1,
                    'quantity': i.get_quantity()}
            stockdata = stockdata.append(pd.DataFrame(data=temp, index=[0]))
            stockdata = stockdata.reset_index(drop=True)
            stockdata['yrsel'] = min(stockdata['yrsel'])
        for i in streams.columns:
            streams[i] = returns(streams[i])
        return stockdata, streams

    def histport(self, yrs):
        a = self.pxhist.dropna()
        a = a[-252 * yrs:]
        a['mktval'] = 0
        for j in self.data:
            a['mktval'] = a['mktval'] + a[j.get_underlying()] * j.get_quantity()
        a = a.dropna(subset=['mktval'])
        a['return'] = returns(a['mktval'])
        return a

    # portfolio metrics, get portfolio return, risk, sharpe
    def retriskann(self, yrs):
        data = self.histport(yrs)
        yrsel = len(data) / 252
        retann = (data['mktval'][-1] / data['mktval'][0]) ** (1 / yrsel) - 1
        riskann = np.std(data['return']) * np.sqrt(252)
        return retann, riskann

    def retriskann1(self, yrs):
        a = self.pxhist.dropna()
        a = a[-252 * yrs:]
        yrsel = len(a) / 252
        temp = 0
        for i in self.stock:
            ret = (a[i.get_underlying()][-1] / a[i.get_underlying()][0])
            temp += ((self.prices[i.get_underlying()] * i.get_quantity()) / self.mktval) * (ret)
        retann = temp ** (1 / yrsel) - 1

        rets = pd.DataFrame()
        for q in a.columns:
            a[q + 'previous_close'] = a[q].shift(1)
            rets[q + 'return'] = (a[q] / a[q + 'previous_close']) - 1

        covmatrix = rets.cov()
        riskann = np.sqrt(covmatrix.sum().sum()) * np.sqrt(252)

        return retann, riskann


class DataHolder:
    def __init__(self):
        self.alldata = {}
        self.allbids = {}
        self.allasks = {}
        self.pxhist = pd.DataFrame()

    def add_data(self, ticker):
        stock = Stock(ticker, 1)
        stock.loadstockdata()
        self.alldata[ticker] = stock.get_data()
        self.allbids[ticker] = stock.find_bidask()[0]
        self.allasks[ticker] = stock.find_bidask()[1]
        if len(self.alldata) < 1:
            self.pxhist = self.alldata[ticker].history('max')[['Close']]
            self.pxhist.rename(columns={'Close': ticker}, inplace=True)
        else:
            self.pxhist[ticker] = self.alldata[ticker].history('max')['Close']


def optimizer(stocks, allocations, yrs=10, check=False):
    # check looks in the dataholder class
    # stocks = list of tickers, allocations = list of allocations
    p = Portfolio()
    sharpes = pd.DataFrame(columns=['return', 'risk', 'sharpe'])

    if check == False:
        library = DataHolder()
        for i in stocks:
            library.add_data(i)
    else:
        library = check

    for j in range(len(stocks)):
        num = (100 * allocations[j]) / library.allasks[stocks[j]]
        p.add_position(Stock(stocks[j], num), library)
    add = p.retriskann1(yrs)

    temp = {'return': add[0],
            'risk': add[1],
            'sharpe': add[0] / add[1]}

    sharpes = pd.DataFrame(data=temp, index=[0])

    for i in range(len(stocks)):
        sharpes[stocks[i] + ' %'] = allocations[i]
    return sharpes


def sharpeonly(stocks, allocations, yrs=10, check=False):
    # check looks in the dataholder class
    # stocks = list of tickers, allocations = list of allocations
    p = Portfolio()

    if check == False:
        library = DataHolder()
        for i in stocks:
            library.add_data(i)
    else:
        library = check

    for j in range(len(stocks)):
        num = (100 * allocations[j]) / library.allasks[stocks[j]]
        p.add_position(Stock(stocks[j], num), library)
    add = p.retriskann1(yrs)
    sharpes = add[0] / add[1]
    return sharpes


if __name__ == '__main__':
    global library
    global stocks
    library = DataHolder()
    library.add_data('SPY')
    library.add_data('SPAB')
    library.add_data('QQQ')
    stocks = ['SPY', 'QQQ', 'SPAB']

    zzz = perf_counter()
    print('using scipy optimize')

    def obj_func(x):
        # x is the allocations
        return -1 * sharpeonly(stocks, x, yrs=37, check=library)

    def equality_cons(x):
        ## sum of allocations should be 1
        return sum(x) - 1

    bounds = tuple((0, 1) for i in stocks)
    cons = [{'type': 'eq', 'fun': equality_cons}]
    start_pos = [(1 / len(stocks)) for i in stocks]

    res = minimize(obj_func, start_pos, bounds=bounds, constraints=cons)

    zzzend = perf_counter()
    print(zzzend - zzz)
    print(res)
