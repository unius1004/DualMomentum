#import sys
#sys.path.insert(0,"/workspace/bt")

import bt
import pandas as pd

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
    
class WeighAvgMomentumScore(bt.Alog):
    def __init__(self, run_on_end_of_period=True, lookback=pd.DateOffset(months=1), lag=pd.DateOffset(days=1)):
        super(WeighAvgMomentumScore, self).__init__()
        self.run_on_end_of_period = run_on_end_of_period
        self.lookback = lookback
        self.lag = lag
    
    def __avg_momentum_score(self, prc_monthly):
        
        
        return momentum_score
    
    def __call__(self, target):
        selected = target.temp['selected']
        end = target.now - self.lag
        start = end - self.lookback
        prc = target.universe.loc[start:end, selected]
        prc_monthly = prc.copy()
        if (self.run_on_end_of_period):
            prc_monthly = prc_monthly.resample('M').last().dropna()
        else:
            prc_monthly = prc_monthly.resample('MS').last().dropna()
            
        momentum_score = self.__avg_momentum_score(prc_monthly)
        
        target.temp['weights'] = 
        return True