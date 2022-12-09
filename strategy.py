#import sys
#sys.path.insert(0,"/workspace/bt")

import numpy as np
import pandas as pd
import bt

print('bt version : ', bt.__version__)

#상대모멘텀 종목 선정
class SelectRelativeMomentum(bt.Algo):
    def __init__(self, rank, lookback=pd.DateOffset(months=1), lag=pd.DateOffset(days=1)):
        super(SelectRelativeMomentum, self).__init__()
        self.rank = rank
        self.lookback = lookback
        self.lag = lag
        
    def __call__(self, target):
        assets = target.universe.columns
        end = target.now - self.lag
        start = end - self.lookback
        prc = target.universe.loc[start:end, assets]
        momentum = prc.calc_total_return()
        rank = momentum.rank(ascending=False)
        
        selected = pd.Series(dtype=object)
        for i in range(0, len(momentum)):
            if (rank[assets[i]] <= self.rank):
                selected = pd.concat([selected, pd.Series([assets[i]],index=[assets[i]],dtype=object)])
                
        target.temp['selected'] = selected.drop_duplicates()         
        return True
    
    
#절대모멘텀 종목 선정
class SelectAbsoluteMomentum(bt.Algo):
    def __init__(self, rank, lookback=pd.DateOffset(months=1), lag=pd.DateOffset(days=1)):
        super(SelectAbsoluteMomentum, self).__init__()
        self.rank = rank
        self.lookback = lookback
        self.lag = lag
        
    def __call__(self, target):
        assets = target.universe.columns
        end = target.now - self.lag
        start = end - self.lookback
        prc = target.universe.loc[start:end, assets]
        momentum = prc.calc_total_return()
        rank = momentum.rank(ascending=False)
        
        selected = pd.Series(dtype=object)
        for i in range(0, len(momentum)):
            if (rank[assets[i]] <= self.rank) & (momentum[i] > 0):
                selected = pd.concat([selected, pd.Series([assets[i]],index=[assets[i]],dtype=object)])
                
        target.temp['selected'] = selected.drop_duplicates()         
        return True
    
# 듀얼모멘텀 종목 선정
class SelectDualMomentum(bt.Algo):
    """
    듀얼모멘텀 종목을 선정한다.
        Args:
            rank : 상대모멘텀으로 선택할 종목의 개수
            lookback : 반추기간(기본값: 12개월)
            lag : 리밸런스 지연일(기본값: 1일)
        Returns:
            target.temp['selected']에 듀얼모멘텀 종목을 반환
    """
    def __init__(self, rank=1, lookback=pd.DateOffset(months=1), lag=pd.DateOffset(days=1)):
        super(SelectDualMomentum, self).__init__()
        self.rank = rank
        self.lookback = lookback
        self.lag = lag
    
    def __call__(self, target):
        assets = target.universe.columns
        end = target.now - self.lag
        start = end - self.lookback
        prc = target.universe.loc[start:end][assets[0:len(assets)-1]]
        momentum = prc.calc_total_return()
        rank = momentum[assets[0:len(assets)-2]].rank(ascending=False)
        
        selected = pd.Series(dtype=object)
        for i in range(0, len(assets)-2):
            if rank[assets[i]] <= self.rank:
                if momentum[assets[i]] > momentum[-1]:
                    selected = pd.concat([selected, pd.Series([assets[i]],index=[assets[i]],dtype=object)])
                else:
                    selected = pd.concat([selected, pd.Series([assets[-1]],index=[assets[-1]],dtype=object)])

        target.temp['selected'] = selected.drop_duplicates()
        return True
    
class WeighAMSwithCash(bt.Algo):
    """
    자산의 평균모멘텀점수를 계산하여 비중을 할당한다.
        Args:
            lookback : 반추기간(기본값: 12개월)
            lag : 리밸런스 지연일(기본값: 1일)
            cash_weight : 현금 비중(기본값: 0)
        Returns:
            target.temp['weights']에 평균모멘텀점수에 따른 비중을 반환
    """
    def __init__(self, lookback=12, lag=1, cash_weight=0.0):
        super(WeighAMSwithCash, self).__init__()
        self.lookback = lookback
        self.lag = lag
        self.cash_weight = cash_weight
    
    @staticmethod
    def avg_momentum_score(prc_monthly, lookback):
        sumOfmomentum = 0
        sumOfscore = 0
        for i in range(1, lookback+1):
            sumOfmomentum = prc_monthly / prc_monthly.shift(i) + sumOfmomentum
            sumOfscore = np.where(prc_monthly / prc_monthly.shift(i) > 1, 1,0) + sumOfscore
    
        sumOfmomentum[sumOfmomentum > 0] = sumOfscore/lookback
        return sumOfmomentum        
    
    def __call__(self, target):
        selected = target.temp['selected']
        
        # 투자자산 반추기간 데이터 추출
        end = target.now - pd.DateOffset(days=self.lag)
        while not (end in target.universe.index):
            end = end - pd.DateOffset(days=self.lag)
        start = end - pd.DateOffset(months=self.lookback)
        while not (start in target.universe.index):
            start = start - pd.DateOffset(days=self.lag)
        prc = target.universe.loc[start:end, selected]

        # 투자자산 모멘텀 산출
        prc_monthly = prc.copy().drop(['cash'], axis=1)
        prc_monthly = prc_monthly.resample('BM').last().dropna()
        weights = WeighAMSwithCash.avg_momentum_score(prc_monthly, self.lookback)
        
        # 투자자산 및 현금 비중 결정
        weights = weights.loc[weights.index.max()]
        weights = weights * (1-self.cash_weight) / len(weights.index)
        weights['cash'] = 1 - weights.sum()
        
        target.temp['weights'] = weights
        return True

class WeighAMSwithYieldCurve(bt.Algo):
    """
    
    """
    def __init__(self, lookback=12, lag=1, ylookback=6):
        super(WeighAMSwithYieldCurve, self).__init__()
        self.lookback = lookback
        self.lag = lag        
        self.ylookback = ylookback
        
    def __call__(self, target):
        selected = target.temp['selected']
        # 전략수익률 데이터 추출 & 모멘텀 가공
        end = target.now - pd.DateOffset(days=self.lag)
        while not (end in target.universe.index):
            end = end - pd.DateOffset(days=self.lag)
        start = end - pd.DateOffset(months=self.ylookback)
        while not (start in target.universe.index):
            start = start - pd.DateOffset(days=self.lag)
        prc = target.universe.loc[start:end]['yieldcurve']
        prc_monthly = prc.copy()
        prc_monthly = prc_monthly.resample('BM').last().dropna()
        ymomscore = WeighAMSwithCash.avg_momentum_score(prc_monthly, self.ylookback)
        yweight = ymomscore.loc[ymomscore.index.max()]
        
        # 투자자산 데이터 추출 & 모멘텀 가공
        start = end - pd.DateOffset(months=self.lookback)
        while not (start in target.universe.index):
            start = start - pd.DateOffset(days=self.lag)
        prc = target.universe.loc[start:end, selected]
        prc_monthly = prc.copy().drop(['yieldcurve','cash'], axis=1)
        prc_monthly = prc_monthly.resample('BM').last().dropna()
        weights = WeighAMSwithCash.avg_momentum_score(prc_monthly, self.lookback)

        # 전략수익률 곡선 모멘텀에 따른 현금 및 투자자산 비중 결정
        weights = weights.loc[weights.index.max()]
        weights = weights * yweight / len(weights.index)
        weights['cash'] = 1 - weights.sum()
        weights['yieldcurve'] = 0

        target.temp['weights'] = weights
        return True

class WeighInvVol_monthly(bt.Algo):
    """
    자산의 변동성을 산출하여 역가중 방식으로 비중을 할당한다.
        Args:
            lookback : 반추기간(기본값: 12개월)
            lag : 리밸런스 지연일(기본값: 1일)
        Returns:
            target.temp['weights']에 변동성 역가중 방식에 따른 비중을 반환
    """
    def __init__(self, lookback=12, lag=1):
        super(WeighInvVol_monthly, self).__init__()
        self.lookback = lookback;
        self.lag = lag
        
    def __call__(self, target):
        selected = target.temp['selected']
        
        end = target.now - pd.DateOffset(days=self.lag)
        start = end - pd.DateOffset(months=self.lookback+1)
        #print('start day={}, end_day={}'.format(start, end))
        prc = target.universe.loc[start:end, selected].resample('BM').last()
        #print('monthly prc \n', prc)
        target.temp['weights'] = (bt.ffn.calc_inv_vol_weights(prc.to_returns().dropna())).dropna()
        return True
        
        