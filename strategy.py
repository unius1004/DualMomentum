import sys
sys.path.insert(0,"/workspace/bt")

import numpy as np
import pandas as pd
import bt
#import ffn

print('bt version : ', bt.__version__)
#print('ffn version : ', ffn.__version__)

#상대모멘텀 종목 선정
class SelectRelativeMomentum(bt.Algo):
    def __init__(self, rank, lookback=pd.DateOffset(months=1), lag=pd.DateOffset(days=1)):
        super(SelectRelativeMomentum, self).__init__()
        self.rank = rank
        self.lookback = lookback
        self.lag = lag
        
    def __call__(self, target):
        assets = target.universe.columns
        t0 = target.now - self.lag
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
    
class WeighAMS(bt.Algo):
    """
    전략수익률의 평균모멘텀에 따른 비중을 자산의 평균모멘텀점수와 곱하여 비중을 할당한다.
        Args:
            lag : 리밸런스 지연일(기본값: 1일)
            cash_weight : 현금 비중(기본값: 0)
            returns : 전략 수익률 데이터
        Returns:
            target.temp['weights']에 평균모멘텀점수에 따른 비중을 반환
    """
    def __init__(self, lag=1, cash_weight=0.0, returns=pd.DataFrame(), ylookback=12):
        if (not returns.empty) & (ylookback < 1):
            raise ValueError("lookback of Yield can't be less than 0")
        
        super(WeighAMS, self).__init__()
        self.lag = lag
        self.cash_weight = cash_weight
        self.returns = returns
        self.ylookback = ylookback
    
    @staticmethod
    def avg_momentum_score(prc_monthly, lookback):
        sumOfmomentum = 0
        sumOfscore = 0
        for i in range(1, lookback+1):
            sumOfmomentum = prc_monthly / prc_monthly.shift(i) + sumOfmomentum
            sumOfscore = np.where(prc_monthly / prc_monthly.shift(i) > 1, 1,0) + sumOfscore
    
        sumOfmomentum[sumOfmomentum > 0] = sumOfscore/lookback
        return sumOfmomentum
    
    @staticmethod
    def amsofYield(returns, lag, lookback, end, target):
        if (not returns.empty):
            start = end - pd.DateOffset(months=lookback)
            while not (start in target.universe.index):
                start = start - pd.DateOffset(days=lag)
            prc = returns.loc[start:end]
            prc_monthly = prc.copy()
            prc_monthly = prc_monthly.resample('BM').last().dropna()
            ymomscore = WeighAMS.avg_momentum_score(prc_monthly, lookback)
            yweight = ymomscore.loc[ymomscore.index.max()]
        return yweight
    
    def __call__(self, target):
        selected = target.temp['selected']

        end = target.now - pd.DateOffset(days=self.lag)
        while not (end in target.universe.index):
            end = end - pd.DateOffset(days=self.lag)

        s0 = target.now
        if (not self.returns.empty):
            s0 = self.returns.index.min() + pd.DateOffset(months=self.ylookback)
        # 전략수익률 데이터 추출 & 모멘텀 가공
        if (not self.returns.empty) & (target.now >= s0):
            start = end - pd.DateOffset(months=self.ylookback)
            while not (start in target.universe.index):
                start = start - pd.DateOffset(days=self.lag)
            prc = self.returns.loc[start:end]
            prc_monthly = prc.copy()
            prc_monthly = prc_monthly.resample('BM').last().dropna()
            ymomscore = WeighAMS.avg_momentum_score(prc_monthly, self.ylookback)
            yweight = ymomscore.loc[ymomscore.index.max()]
        
        # 투자자산 반추기간 데이터 추출
        start = end - pd.DateOffset(months=12)
        while not (start in target.universe.index):
            start = start - pd.DateOffset(days=self.lag)
        prc = target.universe.loc[start:end, selected]

        # 투자자산 모멘텀 산출
        prc_monthly = prc.copy().drop(['cash'], axis=1)
        prc_monthly = prc_monthly.resample('BM').last().dropna()
        weights = WeighAMS.avg_momentum_score(prc_monthly, lookback=12)
        
        # 투자자산 및 현금 비중 결정
        weights = weights.loc[weights.index.max()]
        if (not self.returns.empty) & (target.now >= s0):
            weights = weights * yweight.iloc[0] / len(weights.index)
        else:
            weights = weights * (1-self.cash_weight) / len(weights.index)
        weights['cash'] = 1 - weights.sum()
        target.temp['weights'] = weights
        return True

class WeighFixed(bt.Algo):
    """
    전략수익률의 평균모멘텀에 따른 비중을 자산의 고정 비중에 곱하여 투자 비중을 반환한다 
    """
    def __init__(self, returns=pd.DataFrame(), lag=1, ylookback=12, **weights):
        super(WeighFixed, self).__init__()
        self.weights = pd.Series(weights)
        self.returns = returns
        self.lag = lag
        self.ylookback = ylookback
        
    def __call__(self, target):
        selected = target.temp['selected']

        t0 = target.now - pd.DateOffset(days=self.lag)
        while not (t0 in target.universe.index):
            t0 = t0 - pd.DateOffset(days=self.lag)
        
        # 수익율의 12개월 평균 모멘텀은 1년의 기간이 필요하다.
        # 투자자산 비중 가공
        weights = self.weights.copy()
        start = self.returns.index.min()
        if (not self.returns.empty):
            start = start + pd.DateOffset(months=self.ylookback)
            if (t0 >= start):
                yweight = WeighAMS.amsofYield(self.returns, self.lag, self.ylookback, t0, target)
                weights = self.weights * yweight.iloc[0]
                weights['SAFE'] = 1 - weights.sum()
            
        target.temp['weights'] = weights.copy()
        return True;
    
class WeighFixedWithCorr(bt.Algo):
    """
    
    """
    def __init__(self, lag, rolling_corr, cval, **weights):
        super(WeighFixedWithCorr, self).__init__()
        self.lag = lag
        self.rolling_corr = rolling_corr
        self.cval = cval
        self.weights = pd.Series(weights)
    
    def __call__(self, target):
        selected = target.temp['selected']

        t0 = target.now - pd.DateOffset(days=self.lag)
        while not (t0 in target.universe.index):
            t0 = t0 - pd.DateOffset(days=self.lag)

        weights = self.weights.copy()
        s0 = target.universe.index.min() + pd.DateOffset(months=12)
        if (t0 >= s0) & (self.rolling_corr.loc[t0].values > self.cval):
            # 투자자산 반추기간 데이터 추출        
            start = t0 - pd.DateOffset(months=12)
            while not (start in target.universe.index):
                start = start - pd.DateOffset(days=self.lag)
            prc = target.universe.loc[start:t0, selected]

            # 투자자산 모멘텀 산출
            prc_monthly = prc.copy()
            prc_monthly = prc_monthly.resample('BM').last().dropna()
            weights = WeighAMS.avg_momentum_score(prc_monthly, lookback=12)
            weights = weights.loc[weights.index.max()]

            # 투자자산 및 현금 비중 결정
            weights = weights * self.weights
            weights['SAFE'] = 1 - weights.sum()
         
        target.temp['weights'] = weights.copy()
        return True
    
class WeighInvVol_12(bt.Algo):
    """
    자산의 변동성을 산출하여 역가중 방식으로 비중을 할당한다.
        Args:
            lookback : 반추기간(기본값: 12개월)
            lag : 리밸런스 지연일(기본값: 1일)
            returns : 전략 수익률 데이터
        Returns:
            target.temp['weights']에 변동성 역가중 방식에 따른 비중을 반환
    """
    def __init__(self,lag=1, returns=pd.DataFrame(), ylookback=12):
        super(WeighInvVol_12, self).__init__()
        self.lag = lag
        self.returns = returns
        self.ylookback=ylookback
        
    def __call__(self, target):
        selected = target.temp['selected']
        
        end = target.now - pd.DateOffset(days=self.lag)
        while not (end in target.universe.index):
            end = end - pd.DateOffset(days=self.lag)
        
        s0 = target.now
        if (not self.returns.empty):
            s0 = self.returns.index.min() + pd.DateOffset(months=self.ylookback)
        # 전략수익률 데이터 추출 & 모멘텀 가공
        if (not self.returns.empty) & (target.now >= s0):
            start = end - pd.DateOffset(months=self.ylookback)
            while not (start in target.universe.index):
                start = start - pd.DateOffset(days=self.lag)
            prc = self.returns.loc[start:end]
            prc_monthly = prc.copy()
            prc_monthly = prc_monthly.resample('BM').last().dropna()
            ymomscore = WeighAMS.avg_momentum_score(prc_monthly, self.ylookback)
            yweight = ymomscore.loc[ymomscore.index.max()]
                
        # 투자자산 반추기간 데이터 추출            
        start = end - pd.DateOffset(months=12)
        #print('start day={}, end_day={}'.format(start, end))
        if (not self.returns.empty) & (target.now >= s0):
            prc = target.universe.loc[start:end].drop(['cash'],axis=1).resample('BM').last()
        else:
            prc = target.universe.loc[start:end, selected].resample('BM').last()

        # 투자자산 및 현금 비중 결정
        #print('monthly prc \n', prc)
        weights = (bt.ffn.calc_inv_vol_weights(prc.to_returns().dropna())).dropna()
        if (not self.returns.empty) & (target.now >= s0):
            weights = weights * yweight.iloc[0] / len(weights.index)
            weights['cash'] = 1- weights.sum()
        
        target.temp['weights'] = weights
        return True

class WeighADM(bt.Algo):
    """
    n개의 공격자산과 1개의 안전자산으로 구성된 투자자산(target.temp['selected'])에 가속듀얼모멘텀 방식으로 비중을 할당한다.
    n개의 공격자산중에서 (1,3,6개월) 평균 수익률 순위(self.rank)내에 있는 공격자산에 동일 비중으로 투자한다.
    단, 해당 공격자산의 평균 수익률이 0보다 작을 경우 안전자산으로 변경함.
        Args:
            rank : 투자할 공격자산의 개수
            lag : 리밸런스 지연일
        Returns:
            target.temp['weights']에 가속듀얼모멘텀 방식에 따른 비중을 반환
    """
    def __init__(self, rank=1, lag=1):
        super(WeighADM, self).__init__()
        self.rank = rank
        self.lag = lag
        
    def __call__(self, target):
        selected = target.temp['selected'].copy()
        t0 = target.now - self.lag
        momentum1 = target.universe[selected].loc[t0 - pd.DateOffset(months=1):t0]
        momentum3 = target.universe[selected].loc[t0 - pd.DateOffset(months=1):t0]
        momentum6 = target.universe[selected].loc[t0 - pd.DateOffset(months=1):t0]
        
        ret1 = momentum1.calc_total_return()
        ret3 = momentum3.calc_total_return()
        ret6 = momentum6.calc_total_return()
        
        assetcount = len(selected) - 1
        
        avg = (ret1 + ret3 + ret6) / 3
        #print('avg \n', avg)
        rank = avg[selected[0:assetcount]].rank(ascending=False)
        
        weights = pd.Series([0]*len(selected), index=selected)
        for i in range(0, assetcount):
            if rank[selected[i]] <= self.rank:
                if avg[selected[i]] > 0:
                    weights[selected[i]] = 1
                else:
                    #print('selected[{}]={}'.format(assetcount, selected[assetcount]))
                    weights[selected[assetcount]] = 1
        
        #print('weights \n', weights)
        weights = weights / weights.sum()
        #print('final weights \n', weights)
        target.temp['weights'] = weights
        return True
        