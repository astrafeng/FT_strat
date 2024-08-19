import logging
from functools import reduce
import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib
from freqtrade.strategy import IStrategy, RealParameter

logger = logging.getLogger(__name__)

class AlexStrategyFinalV4Hyper(IStrategy):
    """
    This is an example strategy designed for Hyperopt. 
    The freqtradeai components are disabled for the optimization process.
    """
    
    # Hyperspace parameters:
    buy_params = {
      "threshold_buy": 0.7489597985120735,
      "w0": 0.10525888874953596,
      "w1": 0.6435030916677758,
      "w2": 0.507638726796731,
      "w3": 0.7592548870241507,
      "w4": 0.039608683210666136,
      "w5": 0.22325150455398607,
      "w6": 0.48455635995280055,
      "w7": 0.20137758903438124,
      "w8": 0.48582647909942456
    }

    sell_params = {
       "threshold_sell": -0.24274929030474213

    }

    # ROI table:
    minimal_roi = {
        "0": 0.04,
        "186": 0.135,
        "663": 0.069,
        "948": 0
    }

    # Stoploss:
    stoploss = -0.237

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    timeframe = "15m"
    can_short = True
    can_long = True
    use_exit_signal = True
    process_only_new_candles = True

    startup_candle_count = 20
    leverage_value = 7.0

    threshold_buy = RealParameter(-1, 1, default=0, space='buy')
    threshold_sell = RealParameter(-1, 1, default=0, space='sell')

    # Weights for calculating the aggregate score - the sum of all weighted normalized indicators has to be 1!
    w0 = RealParameter(0, 1, default=0.10, space='buy')
    w1 = RealParameter(0, 1, default=0.15, space='buy')
    w2 = RealParameter(0, 1, default=0.10, space='buy')
    w3 = RealParameter(0, 1, default=0.15, space='buy')
    w4 = RealParameter(0, 1, default=0.10, space='buy')
    w5 = RealParameter(0, 1, default=0.10, space='buy')
    w6 = RealParameter(0, 1, default=0.10, space='buy')
    w7 = RealParameter(0, 1, default=0.05, space='buy')
    w8 = RealParameter(0, 1, default=0.15, space='buy')

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: dict, **kwargs) -> DataFrame:
        dataframe["%-cci-period"] = ta.CCI(dataframe, timeperiod=20)
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=10)
        dataframe["%-momentum-period"] = ta.MOM(dataframe, timeperiod=4)
        dataframe['%-ma-period'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['%-macd-period'], dataframe['%-macdsignal-period'], dataframe['%-macdhist-period'] = ta.MACD(dataframe['close'], slowperiod=12, fastperiod=26)
        dataframe['%-roc-period'] = ta.ROC(dataframe, timeperiod=2)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=period, stds=2.2)
        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]
        dataframe["%-bb_width-period"] = (dataframe["bb_upperband-period"] - dataframe["bb_lowerband-period"]) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = dataframe["close"] / dataframe["bb_lowerband-period"]

        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe['date'] = pd.to_datetime(dataframe['date'])
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ma'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=2)
        dataframe['macd'], dataframe['macdsignal'], dataframe['macdhist'] = ta.MACD(dataframe['close'], slowperiod=12, fastperiod=26)
        dataframe['momentum'] = ta.MOM(dataframe, timeperiod=4)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=10)
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_upperband'] = bollinger['upperband']
        dataframe['bb_middleband'] = bollinger['middleband']
        dataframe['bb_lowerband'] = bollinger['lowerband']
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        dataframe['stoch'] = ta.STOCH(dataframe)['slowk']
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['obv'] = ta.OBV(dataframe)

        # Normalize Indicators:
        dataframe['normalized_stoch'] = (dataframe['stoch'] - dataframe['stoch'].rolling(window=14).mean()) / dataframe['stoch'].rolling(window=14).std()
        dataframe['normalized_atr'] = (dataframe['atr'] - dataframe['atr'].rolling(window=14).mean()) / dataframe['atr'].rolling(window=14).std()
        dataframe['normalized_obv'] = (dataframe['obv'] - dataframe['obv'].rolling(window=14).mean()) / dataframe['obv'].rolling(window=14).std()
        dataframe['normalized_ma'] = (dataframe['close'] - dataframe['close'].rolling(window=10).mean()) / dataframe['close'].rolling(window=10).std()
        dataframe['normalized_macd'] = (dataframe['macd'] - dataframe['macd'].rolling(window=26).mean()) / dataframe['macd'].rolling(window=26).std()
        dataframe['normalized_roc'] = (dataframe['roc'] - dataframe['roc'].rolling(window=2).mean()) / dataframe['roc'].rolling(window=2).std()
        dataframe['normalized_momentum'] = (dataframe['momentum'] - dataframe['momentum'].rolling(window=4).mean()) / dataframe['momentum'].rolling(window=4).std()
        dataframe['normalized_rsi'] = (dataframe['rsi'] - dataframe['rsi'].rolling(window=10).mean()) / dataframe['rsi'].rolling(window=10).std()
        dataframe['normalized_bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']).rolling(window=20).mean() / (dataframe['bb_upperband'] - dataframe['bb_lowerband']).rolling(window=20).std()
        dataframe['normalized_cci'] = (dataframe['cci'] - dataframe['cci'].rolling(window=20).mean()) / dataframe['cci'].rolling(window=20).std()

        # Calculate aggregate score S
        w = [self.w0.value, self.w1.value, self.w2.value, self.w3.value, self.w4.value, self.w5.value, self.w6.value, self.w7.value, self.w8.value]
        dataframe['S'] = w[0] * dataframe['normalized_ma'] + w[1] * dataframe['normalized_macd'] + w[2] * dataframe['normalized_roc'] + w[3] * dataframe['normalized_rsi'] + w[4] * dataframe['normalized_bb_width'] + w[5] * dataframe['normalized_cci'] + w[6] * dataframe['normalized_momentum'] + w[7] * dataframe['normalized_stoch'] + w[8] * dataframe['normalized_obv']

        # Market Regime Filter R
        dataframe['R'] = 0
        dataframe.loc[(dataframe['close'] > dataframe['bb_middleband']) & (dataframe['close'] > dataframe['bb_upperband']), 'R'] = 1
        dataframe.loc[(dataframe['close'] < dataframe['bb_middleband']) & (dataframe['close'] < dataframe['bb_lowerband']), 'R'] = -1

        # Additional Market Regime Filter based on long-term MA
        dataframe['ma_100'] = ta.SMA(dataframe, timeperiod=100)
        dataframe['R2'] = np.where(dataframe['close'] > dataframe['ma_100'], 1, -1)

        # Volatility Adjustment V
        bb_width = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']
        dataframe['V'] = 1 / bb_width

        # Another Volatility Adjustment using ATR
        dataframe['V2'] = 1 / dataframe['atr']

        # Get Final Target Score to incorporate new calculations
        dataframe['T'] = dataframe['S'] * dataframe['R'] * dataframe['V'] * dataframe['R2'] * dataframe['V2']
        dataframe['&-target'] = dataframe['T']

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_long_conditions = [
            df['&-target'] > self.threshold_buy.value,
            df['volume'] > 0
        ]

        enter_short_conditions = [
            df['&-target'] < self.threshold_sell.value,
            df['volume'] > 0
        ]

        df.loc[reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]] = (1, "long")
        df.loc[reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [
            df['&-target'] < self.threshold_sell.value
        ]

        exit_short_conditions = [
            df['&-target'] > self.threshold_buy.value
        ]

        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), ["exit_long", "exit_tag"]] = (1, "exit_long")

        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), ["exit_short", "exit_tag"]] = (1, "exit_short")

        return df

    def leverage(self, pair: str, current_time: 'datetime', current_rate: float, proposed_leverage: float, **kwargs) -> float:
        return self.leverage_value
