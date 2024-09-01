import datetime
import numpy as np
import pandas_ta as pta
import talib.abstract as ta
from pandas import DataFrame
from datetime import datetime
from technical import qtpylib
from scipy.stats import linregress
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter, IStrategy)
from freqtrade.persistence import Trade
from typing import Optional, Tuple, Union


class SlopeV4(IStrategy):
    INTERFACE_VERSION = 3
    
    can_short = True
    timeframe = '5m'
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.1
    
    
    # Buy hyperspace params:
    buy_params = {
        "aroonosc_buy" : -60,
        "minus_di": 25,
        "plus_di": 57,
        "volume_long": 2.312,
        "window": 24,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "volume_short": 5.143,
    }

    # ROI table:  # value loaded from strategy
    minimal_roi = {
        "0": 0.398,
        "99": 0.11,
        "209": 0.03,
        "284": 0
    }

    # Stoploss:
    stoploss = -0.2
    
    # Max Open Trades:
    max_open_trades = -1

    # Hyperparameters
    aroonosc_buy =     IntParameter(   -100,  0, space='buy', default=-80,  optimize=True)
    timeperiod =     IntParameter(   1,  120, space='buy', default=34,  optimize=True)
    volume_pct = DecimalParameter(0.0, 100.0, space='buy', default=3.5, optimize=True)
    window = IntParameter(1, 120, space='buy',  default=buy_params['window'], optimize=False)
    minus_di = IntParameter(1, 100, space='buy',  default=buy_params['minus_di'], optimize=True)
    plus_di = IntParameter(1, 100, space='buy',  default=buy_params['plus_di'], optimize=True)
    volume_long = DecimalParameter(0.0, 100.0, space='buy',  default=buy_params['volume_long'], optimize=True)
    volume_short = DecimalParameter(0.0, 100.0, space='sell', default=sell_params['volume_short'], optimize=True)
    rsi_entry_long  = IntParameter(0, 100, default=buy_params.get('rsi_entry_long'),  space='buy',  optimize=True)
    rsi_exit_long   = IntParameter(0, 100, default=buy_params.get('rsi_exit_long'),   space='sell', optimize=True)
    rsi_entry_short = IntParameter(0, 100, default=buy_params.get('rsi_entry_short'), space='buy',  optimize=True)
    rsi_exit_short  = IntParameter(0, 100, default=buy_params.get('rsi_exit_short'),  space='sell', optimize=True)
    

    @property
    def plot_config(self):
        plot_config = {
            'main_plot' : {
            'sar'   : { 'color' : 'white' },
            },
            'subplots' : {
                'Directional AROON' : {
                    'aroonup'   : { 'color' : 'red' },
                    'aroondown' : { 'color' : 'blue' },
                    'aroonosc' : { 'color' : 'white' },
                },
                'Directional MACD' : {
                    'macdsignal'   : { 'color' : 'green' },
                    'macdhist'   : { 'color' : 'bleu' },
                    'cci'  : { 'color' : 'yellow' },
                    'macd'  : { 'color' : 'red' },
                },
                #'Directional RSI' : {
                #    'rsi'   : { 'color' : 'green' },
                #    'rsi_ema'  : { 'color' : 'red' },
                #    'rsi_gra' : { 'color' : 'blue' },
                #},
                'Indicator' : {
                    'adx'   : { 'color' : 'green' },
                    'tema' : { 'color' : 'blue' },
                    'mfi' : { 'color' : 'yellow' },
                    'rsi' : { 'color' : 'red' },
                },
                #'Directional DI' : {
                #    'mid_di'   : { 'color' : 'white' },
                #    'plus_di'  : { 'color' : 'red' },
                #    'minus_di' : { 'color' : 'blue' },
                #    'volume_pct' : { 'color' : 'black' },
                #},                      
            }
        }

        return plot_config
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag:str, side: str, **kwargs) -> float:
        return 10.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Parabolic SAR
        dataframe['sar'] = ta.SAR(dataframe)

        # TEMA - Triple Exponential Moving Average
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)        
        
        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)
        
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        )
        
        ###MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['cci'] = ta.CCI(dataframe)

        # AROON
        aroon = ta.AROON(dataframe)
        dataframe['aroonup'] = aroon['aroonup']
        dataframe['aroondown'] = aroon['aroondown']
        dataframe['aroonosc'] = ta.AROONOSC(dataframe)
    
        ###DI
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=self.timeperiod.value)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=self.timeperiod.value)
        dataframe['mid_di'] = dataframe['plus_di'] - dataframe['minus_di']
        dataframe['volume_pct'] = dataframe['volume'].pct_change(self.timeperiod.value)
        dataframe['maximum'] = np.where( dataframe['minus_di'].shift(1) < dataframe['minus_di'], dataframe['close'], np.nan )
        dataframe['minimum'] = np.where( dataframe['minus_di'].shift(1) > dataframe['minus_di'], dataframe['close'], np.nan )
        
        ###RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_ema'] = dataframe['rsi'].ewm(span=self.window.value).mean()
        dataframe['rsi_gra'] = np.gradient(dataframe['rsi_ema'])

        ###ADX
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['short'] = ta.SMA(dataframe, timeperiod=3)
        dataframe['long'] = ta.SMA(dataframe, timeperiod=6)       


        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # Uptrend detected, entry long
        dataframe.loc[
            (
                #(dataframe['volume_pct'] > self.volume_pct.value) &
                (dataframe['mfi'] > dataframe['mfi'].shift(1)) &
                (dataframe['adx'] > dataframe['adx'].shift(1)) &
                (dataframe['mfi'] < dataframe['rsi']) &
                (dataframe['volume']     > 0)
            ),
        'enter_long'] = 1

        dataframe.loc[
            (       
                (dataframe['plus_di']   < dataframe['minus_di']) &
                (dataframe['rsi_gra']   < dataframe['rsi_gra'].shift(1)) &
                (dataframe['rsi']   > dataframe['rsi_ema'].shift(1)) &
                (dataframe['rsi']   > dataframe['rsi'].shift(1))
            ),
        'enter_short'] = 1
import datetime
import numpy as np
import pandas_ta as pta
import talib.abstract as ta
from pandas import DataFrame
from datetime import datetime
from technical import qtpylib
from scipy.stats import linregress
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter, IStrategy)
from freqtrade.persistence import Trade
from typing import Optional, Tuple, Union


class SlopeV4(IStrategy):
    INTERFACE_VERSION = 3
    
    can_short = True
    timeframe = '15m'
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.1
    
    # ROI table:
    minimal_roi = { '0': 1 }

    # Stoploss:
    stoploss = -0.2

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.065
    trailing_only_offset_is_reached = True
    
    # Max Open Trades:
    max_open_trades = -1

    # Hyperparameters
    timeperiod =     IntParameter(   1,  120, space='buy', default=34,  optimize=True)
    volume_pct = DecimalParameter(0.0, 100.0, space='buy', default=3.5, optimize=True)

    # Hyperparameters
    window = IntParameter(1, 120, space='buy',  default=14, optimize=False)
    #minus_di = IntParameter(1, 100, space='buy',  default=buy_params['minus_di'], optimize=True)
    #plus_di = IntParameter(1, 100, space='buy',  default=buy_params['plus_di'], optimize=True)
    #volume_long = DecimalParameter(0.0, 100.0, space='buy',  default=buy_params['volume_long'], optimize=True)
    #volume_short = DecimalParameter(0.0, 100.0, space='sell', default=sell_params['volume_short'], optimize=True)
    #rsi_entry_long  = IntParameter(0, 100, default=buy_params.get('rsi_entry_long'),  space='buy',  optimize=True)
    #rsi_exit_long   = IntParameter(0, 100, default=buy_params.get('rsi_exit_long'),   space='sell', optimize=True)
    #rsi_entry_short = IntParameter(0, 100, default=buy_params.get('rsi_entry_short'), space='buy',  optimize=True)
    #rsi_exit_short  = IntParameter(0, 100, default=buy_params.get('rsi_exit_short'),  space='sell', optimize=True)
    

    @property
    def plot_config(self):
        plot_config = {
            'main_plot' : {
            'sar'   : { 'color' : 'white' },
            'bb_middleband'   : { 'color' : 'green' },
            },
            'subplots' : {
                'Directional AROON' : {
                    'aroonup'   : { 'color' : 'red' },
                    'aroondown' : { 'color' : 'blue' },
                    'aroonosc' : { 'color' : 'white' },
                },
                'Directional MACD' : {
                    'macdsignal'   : { 'color' : 'green' },
                    'macdhist'   : { 'color' : 'bleu' },
                    'cci'  : { 'color' : 'yellow' },
                    'macd'  : { 'color' : 'red' },
                },
                #'Directional RSI' : {
                #    'rsi'   : { 'color' : 'green' },
                #    'rsi_ema'  : { 'color' : 'red' },
                #    'rsi_gra' : { 'color' : 'blue' },
                #},
                'Indicator' : {
                    'adx'   : { 'color' : 'green' },
                    'tema' : { 'color' : 'blue' },
                    'mfi' : { 'color' : 'yellow' },
                    'rsi' : { 'color' : 'red' },
                },
                #'Directional DI' : {
                #    'mid_di'   : { 'color' : 'white' },
                #    'plus_di'  : { 'color' : 'red' },
                #    'minus_di' : { 'color' : 'blue' },
                #    'volume_pct' : { 'color' : 'black' },
                #},                      
            }
        }

        return plot_config
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag:str, side: str, **kwargs) -> float:
        return 10.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        ## Parabolic SAR
        #dataframe['sar'] = ta.SAR(dataframe)
        #
        ## TEMA - Triple Exponential Moving Average
        #dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)        
        #
        ## MFI
        #dataframe['mfi'] = ta.MFI(dataframe)
        #
        ## Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=self.window.value, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        )
        #
        ####MACD
        #macd = ta.MACD(dataframe)
        #dataframe['macd'] = macd['macd']
        #dataframe['macdsignal'] = macd['macdsignal']
        #dataframe['macdhist'] = macd['macdhist']
        #dataframe['cci'] = ta.CCI(dataframe)
        #
        ## AROON
        #aroon = ta.AROON(dataframe)
        #dataframe['aroonup'] = aroon['aroonup']
        #dataframe['aroondown'] = aroon['aroondown']
        #dataframe['aroonosc'] = ta.AROONOSC(dataframe)
    
        ###DI
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=self.timeperiod.value)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=self.timeperiod.value)
        dataframe['mid_di'] = dataframe['plus_di'] - dataframe['minus_di']
        dataframe['volume_pct'] = dataframe['volume'].pct_change(self.timeperiod.value)
        dataframe['maximum'] = np.where( dataframe['minus_di'].shift(1) < dataframe['minus_di'], dataframe['close'], np.nan )
        dataframe['minimum'] = np.where( dataframe['minus_di'].shift(1) > dataframe['minus_di'], dataframe['close'], np.nan )
        
        ###RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_ema'] = dataframe['rsi'].ewm(span=self.window.value).mean()
        dataframe['rsi_gra'] = np.gradient(dataframe['rsi_ema'])
        
        ####ADX
        #dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        #dataframe['short'] = ta.SMA(dataframe, timeperiod=3)
        #dataframe['long'] = ta.SMA(dataframe, timeperiod=6)       

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Uptrend detected, entry long
        dataframe.loc[
            (
                (dataframe['volume_pct'] > self.volume_pct.value) &
                (dataframe['minus_di']   < dataframe['plus_di']) &
                (dataframe['bb_middleband']   > dataframe['bb_middleband'].shift(1)) &
                (dataframe['volume']     > 0)
            ),
        'enter_long'] = 1

        # Downtrend detected, entry short
        dataframe.loc[
            (
                (dataframe['volume_pct'] > self.volume_pct.value) &
                (dataframe['minus_di']   > dataframe['plus_di']) &
                (dataframe['bb_middleband']   < dataframe['bb_middleband'].shift(1)) &
                (dataframe['volume']     > 0)
            ),
        'enter_short'] = 1
        
        dataframe.loc[
            (    
                (dataframe['plus_di']   < dataframe['minus_di']) &
                (dataframe['rsi_gra']   < dataframe['rsi_gra'].shift(1)) &
                (dataframe['rsi']   > dataframe['rsi_ema'].shift(1)) &
                (dataframe['rsi']   > dataframe['rsi'].shift(1))
            ),
        'enter_short'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Downtrend detected, exit long
        dataframe.loc[
            (
                (dataframe['maximum'].notnull()) &
                (dataframe['volume'] > 0)
            ),
        'exit_long'] = 1

        # Uptrend detected, exit short
        dataframe.loc[
            (
                (dataframe['minimum'].notnull()) &
                (dataframe['volume'] > 0)
            ),
        'exit_short'] = 1

        return dataframe