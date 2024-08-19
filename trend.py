from functools import reduce
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy

class trend(IStrategy):
    INTERFACE_VERSION = 3
    INTERFACE_VERSION: int = 3
    # ROI table:
    minimal_roi = {'0': 0.166, '10': 0.024, '36': 0.011, '93': 0}
    # minimal_roi = {"0": 1}
    # Stoploss:
    stoploss = -0.299
    can_short = True
    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.052
    trailing_stop_positive_offset = 0.147
    trailing_only_offset_is_reached = False
    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate OBV
        dataframe['obv'] = ta.OBV(dataframe['close'], dataframe['volume'])
        # Add your trend following indicators here
        dataframe['trend'] = dataframe['close'].ewm(span=20, adjust=False).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Add your trend following buy signals here
        dataframe.loc[(dataframe['close'] > dataframe['trend']) & (dataframe['close'].shift(1) <= dataframe['trend'].shift(1)) & (dataframe['obv'] > dataframe['obv'].shift(1)), 'enter_long'] = 1
        # Add your trend following sell signals here
        dataframe.loc[(dataframe['close'] < dataframe['trend']) & (dataframe['close'].shift(1) >= dataframe['trend'].shift(1)) & (dataframe['obv'] < dataframe['obv'].shift(1)), 'enter_short'] = -1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Add your trend following exit signals for long positions here
        dataframe.loc[(dataframe['close'] < dataframe['trend']) & (dataframe['close'].shift(1) >= dataframe['trend'].shift(1)) & (dataframe['obv'] > dataframe['obv'].shift(1)), 'exit_long'] = 1
        # Add your trend following exit signals for short positions here
        dataframe.loc[(dataframe['close'] > dataframe['trend']) & (dataframe['close'].shift(1) <= dataframe['trend'].shift(1)) & (dataframe['obv'] < dataframe['obv'].shift(1)), 'exit_short'] = 1
        return dataframe