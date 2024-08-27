
import warnings
warnings.filterwarnings('ignore')

import warnings
warnings.filterwarnings('ignore')
import numpy
import warnings
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from pandas import DataFrame, errors
from datetime import datetime
import numpy
from scipy.signal import argrelextrema
from freqtrade.strategy import (IStrategy, DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter)

warnings.simplefilter(action="ignore", category=errors.PerformanceWarning)



class el_rsiqui(IStrategy):
    INTERFACE_VERSION = 3

    can_short = True

    entry_params = {
        'base_nb_candles_entry': 12,
        'ewo_high': 4.428,
        'ewo_low': -12.383,
        'low_offset': 0.915,
        'rsi_entry': 44,
    }

    exit_params = {
        'base_nb_candles_exit': 72,
        'high_offset': 1.008,
    }

    minimal_roi = {
        '0': 0.5,
        '60': 0.45,
        '120': 0.4,
        '240': 0.3,
        '360': 0.25,
        '720': 0.2,
        '1440': 0.15,
        '2880': 0.1,
        '3600': 0.05,
        '7200': 0.02,
    }

    stoploss = -0.05
    can_short = True
    max_open_trades = 9
    timeframe = '5m'
    informative_timeframe = '1h'
    trailing_stop = False
    
    rsi_entry_long = IntParameter(0, 50, default=30, space='buy', optimize=True)
    rsi_entry_short = IntParameter(50, 100, default=70, space='buy', optimize=True)
    rsi_exit_long = IntParameter(50, 100, default=60, space='sell', optimize=True)
    rsi_exit_short = IntParameter(0, 50, default=40, space='sell', optimize=True)

    cooldown_lookback = IntParameter(2, 48, default=1, space='protection', optimize=True)
    stop_duration = IntParameter(12, 200, default=4, space='protection', optimize=True)
    use_stop_protection = BooleanParameter(default=True, space='protection', optimize=True)

    @property
    def protections(self):
        prot = []
        prot.append(
            {
                'method': 'CooldownPeriod', 
                'stop_duration_candles': self.cooldown_lookback.value
            }
        )
        if self.use_stop_protection.value:
            prot.append(
                {
                    'method': 'StoplossGuard',
                    'lookback_period_candles': 24 * 3,
                    'trade_limit': 2,
                    'stop_duration_candles': self.stop_duration.value,
                    'only_per_pair': False,
                }
            )
        return prot
    
    @property
    def plot_config(self):
        plot_config = {}

        plot_config['main_plot'] = {
            'rsi': {},
        }
        plot_config['subplots'] = {
            'RSI': {
                'rsi_gra' : {},
            },
        }

        return plot_config
    
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def get_informative_indicators(self, metadata: dict):
        dataframe = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_gra'] = numpy.gradient(dataframe['rsi'], 60)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < 30) &
                qtpylib.crossed_above(dataframe['rsi_gra'], 0)

            ),
        'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['rsi'] > 70) &
                qtpylib.crossed_below(dataframe['rsi_gra'], 0)
            ),
        'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > 60) &
                qtpylib.crossed_below(dataframe['rsi_gra'], 0)
            ),
        'exit_long'] = 1
        
        dataframe.loc[
            (
                (dataframe['rsi'] < 40) &
                qtpylib.crossed_above(dataframe['rsi_gra'], 0)
            ),
        'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag:str, side: str, **kwargs) -> float:
        return 10.0