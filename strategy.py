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
            모멘텀점수에 따른 비중을 반환
    """
    def __init__(self, lookback=12, lag=1, cash_weight=0.0):
        super(WeighAMSwithCash, self).__init__()
        self.lookback = lookback
        self.lag = lag
        self.cash_weight = cash_weight
    
    # 12개월 평균모멘텀 스코어 산출    
    def __avg_momentum_score(self, prc_monthly):
        sumOfmomentum = 0
        sumOfscore = 0
        for i in range(1, self.lookback+1):
            sumOfmomentum = prc_monthly / prc_monthly.shift(i) + sumOfmomentum
            sumOfscore = np.where(prc_monthly / prc_monthly.shift(i) > 1, 1,0) + sumOfscore
    
        sumOfmomentum[sumOfmomentum > 0] = sumOfscore/self.lookback
        return sumOfmomentum        
    
    def __call__(self, target):
        selected = target.temp['selected']
        end = target.now - pd.DateOffset(days=self.lag)
        start = end - pd.DateOffset(months=self.lookback+1)
        #print('start day={}, end_day={}'.format(start, end))
        prc = target.universe.loc[start:end, selected]
        #print('price \n', prc)
        prc_monthly = prc.copy()
        prc_monthly = prc_monthly.resample('BM').last().dropna()
        #print('monthly prc \n', prc_monthly)
        
        momentum_score = self.__avg_momentum_score(prc_monthly)
        #print('momentum score \n', momentum_score)
        #weights = pd.Series(momentum_score.loc[momentum_score.index.max()] * 1/(len(selected)-1), index=momentum_score.columns)
        
        assets_count = len(selected) - 1
        weights = pd.Series(momentum_score.loc[momentum_score.index.max()], index=momentum_score.columns)
        weights = weights * (1-self.cash_weight) / assets_count
        
        weights['cash'] = self.cash_weight
        weights['cash'] = 1 - weights.sum()
        
        #print('before weight \n', weights)
        #weights[-1] = 1 - (weights.sum() - weights[-1])
        target.temp['weights'] = weights
        #print('after weight \n', weights)
        #print('weights \n', target.temp['weights'])
        return True
    
class WeighInvVol_monthly(bt.Algo):
    """
    
    
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
        
        