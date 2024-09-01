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

class eZ3(IStrategy):
    INTERFACE_VERSION = 3
    
    can_short = True
    process_only_new_candles  = True
    timeframe = '5m'
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.1
    
    # ROI table:
    minimal_roi = { '0': 1 }

    # Stoploss:
    stoploss = -0.99

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
                    'aroondiv' : { 'color' : 'black' },
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
        dataframe['aroondiv'] = (dataframe['aroonup'] - dataframe['aroondown'])
        dataframe['aroonosc'] = ta.AROONOSC(dataframe)
    
        ###DI
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=self.timeperiod.value)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=self.timeperiod.value)
        dataframe['mid_di'] = dataframe['plus_di'] - dataframe['minus_di']
        dataframe['mid_di2'] = dataframe['minus_di'] - dataframe['plus_di']
        dataframe['volume_pct'] = dataframe['volume'].pct_change(self.timeperiod.value)
        dataframe['maximum'] = np.where( dataframe['minus_di'].shift(1) < dataframe['minus_di'], dataframe['close'], np.nan )
        dataframe['minimum'] = np.where( dataframe['minus_di'].shift(1) > dataframe['minus_di'], dataframe['close'], np.nan )
        
        ###RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_ema'] = dataframe['rsi'].ewm(span=self.window.value).mean()
        dataframe['rsi_gra'] = np.gradient(dataframe['rsi_ema'])

       #StochRSI 
        period = 14
        smoothD = 3
        SmoothK = 3
        stochrsi  = (dataframe['rsi'] - dataframe['rsi'].rolling(period).min()) / (dataframe['rsi'].rolling(period).max() - dataframe['rsi'].rolling(period).min())
        dataframe['srsi_k'] = stochrsi.rolling(SmoothK).mean() * 100
        dataframe['srsi_d'] = dataframe['srsi_k'].rolling(smoothD).mean()
        dataframe['srsi_diff'] = (dataframe['srsi_k'] - dataframe['srsi_d'])
        dataframe['srsi_macd'] = (dataframe['srsi_k'] - dataframe['srsi_d'])

        ###ADX
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['short'] = ta.SMA(dataframe, timeperiod=3)
        dataframe['long'] = ta.SMA(dataframe, timeperiod=6)


        ###CUSTOM
        dataframe['ghost'] = (dataframe['cci'] - dataframe['mfi'])
        dataframe['witch'] = (dataframe['macd'] + dataframe['macdhist'])

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Uptrend detected, entry long
        dataframe.loc[
            (
                (dataframe['rsi_gra']   > dataframe['rsi_gra'].shift(1)) &
                (dataframe['ghost']   < dataframe['ghost'].shift(1)) &
                (dataframe['volume']     > 0)
            ),
        'enter_long'] = 1
        

        dataframe.loc[
            (    
                (dataframe['rsi_gra']   < dataframe['rsi_gra'].shift(1)) &
                (dataframe['ghost']   > dataframe['ghost'].shift(1)) &
                (dataframe['volume']     > 0)
            ),
        'enter_short'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Downtrend detected, exit long
        dataframe.loc[
            (
                (dataframe['rsi_gra']   < dataframe['rsi_gra'].shift(1)) &
                (dataframe['ghost']   < dataframe['ghost'].shift(1)) &
                (dataframe['volume'] > 0)
            ),
        'exit_long'] = 1

        # Uptrend detected, exit short
        dataframe.loc[
            (
                (dataframe['rsi_gra']   > dataframe['rsi_gra'].shift(1)) &
                (dataframe['ghost']   > dataframe['ghost'].shift(1)) &
                (dataframe['volume'] > 0)
            ),
        'exit_short'] = 1
        return dataframe