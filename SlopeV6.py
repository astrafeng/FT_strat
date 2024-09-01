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
from scipy.signal import argrelextrema



class SlopeV6(IStrategy):
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

    @property
    def plot_config(self):
        plot_config = {
            'main_plot' : { },
            'subplots' : {
                'Directional Indicator' : {
                    'plus_di'    : { 'color' : 'red'   },
                    'minus_di'   : { 'color' : 'blue'  },
                    'volume_pct' : { 'color' : 'black' },
                }
            }
        }

        return plot_config
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag:str, side: str, **kwargs) -> float:
        return 10.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['volume_pct'] = dataframe['volume'].pct_change(self.timeperiod.value)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=self.timeperiod.value)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=self.timeperiod.value)
        dataframe['maximum'] = np.where( dataframe['minus_di'].shift(1) < dataframe['minus_di'], dataframe['close'], np.nan )
        dataframe['minimum'] = np.where( dataframe['minus_di'].shift(1) > dataframe['minus_di'], dataframe['close'], np.nan )
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Uptrend detected, entry long
        dataframe.loc[
            (
                (dataframe['volume_pct'] > self.volume_pct.value) &
                (dataframe['minus_di']   < dataframe['plus_di']) &
                (dataframe['volume']     > 0)
            ),
        'enter_long'] = 1

        # Downtrend detected, entry short
        dataframe.loc[
            (
                (dataframe['volume_pct'] > self.volume_pct.value) &
                (dataframe['minus_di']   > dataframe['plus_di']) &
                (dataframe['volume']     > 0)
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