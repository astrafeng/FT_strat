import numpy
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Tuple, Union
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (IStrategy, DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter)


class RsiquiV4(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = '5m'

    buy_params = {
        "rsi_entry_long": 27,
        "rsi_entry_short": 59,
    }

    minimal_roi = {
        "0": 0.239,
        "10": 0.057,
        "25": 0.016,
        "106": 0
    }

    stoploss = -0.287
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = True
    max_open_trades = -1

    rsi_entry_long  = IntParameter(0,  50,  default=buy_params.get('rsi_entry_long'), space='buy',  optimize=True)
    rsi_entry_short = IntParameter(50, 100, default=buy_params.get('rsi_entry_short'), space='buy',  optimize=True)

    @property
    def plot_config(self):
        plot_config = {
            'main_plot' : {},
            'subplots' : {
                'RSI': {
                    'rsi': {},
                },
                'Gradients' : {
                    'rsi_gra' : {},
                },
            }
        }

        return plot_config

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_gra'] = numpy.gradient(dataframe['rsi'], 60)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < self.rsi_entry_long.value) &
                qtpylib.crossed_above(dataframe['rsi_gra'], 0)
            ),
        'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['rsi'] > self.rsi_entry_short.value) &
                qtpylib.crossed_below(dataframe['rsi_gra'], 0)
            ),
        'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
            ),
        'exit_long'] = 1

        dataframe.loc[
            (
            ),
        'exit_short'] = 1

        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, min_stake: Optional[float], max_stake: float, current_entry_rate: float, current_exit_rate: float, current_entry_profit: float, current_exit_profit: float, **kwargs) -> Optional[float]:
        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        
        if current_profit > 0.25 and trade.nr_of_successful_exits == 0:
            return -(trade.stake_amount / 4)
        if current_profit > 0.40 and trade.nr_of_successful_exits == 1:
            return -(trade.stake_amount / 3)
        
        if (current_profit > -0.15 and count_of_entries == 1) or \
        (current_profit > -0.3 and count_of_entries == 2) or \
        (current_profit > -0.6 and count_of_entries == 3):
            return None
        
        try:
            stake_amount = filled_entries[0].cost if filled_entries else None
            return stake_amount
        except Exception as exception:
            return None


    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        window_size = 50
        base_leverage = 10

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        historical_close_prices = dataframe["close"].tail(window_size)
        historical_high_prices = dataframe["high"].tail(window_size)
        historical_low_prices = dataframe["low"].tail(window_size)

        rsi_values = ta.RSI(historical_close_prices, timeperiod=14)
        atr_values = ta.ATR(historical_high_prices, historical_low_prices, historical_close_prices, timeperiod=14)
        macd_line, signal_line, _ = ta.MACD(historical_close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        sma_values = ta.SMA(historical_close_prices, timeperiod=20)

        current_rsi = rsi_values[-1] if len(rsi_values) > 0 else 50.0
        current_atr = atr_values[-1] if len(atr_values) > 0 else 0.0
        current_macd = (macd_line[-1] - signal_line[-1]) if len(macd_line) > 0 and len(signal_line) > 0 else 0.0
        current_sma = sma_values[-1] if len(sma_values) > 0 else 0.0

        dynamic_rsi_low = numpy.nanmin(rsi_values) if len(rsi_values) > 0 and not numpy.isnan(numpy.nanmin(rsi_values)) else 30.0
        dynamic_rsi_high = numpy.nanmax(rsi_values) if len(rsi_values) > 0 and not numpy.isnan(numpy.nanmax(rsi_values)) else 70.0

        leverage_factors = {
            'long': {'increase': 1.5, 'decrease': 0.5},
            'short': {'increase': 1.5, 'decrease': 0.5},
        }

        if side == "long":
            base_leverage = base_leverage * leverage_factors['long']['increase'] if current_rsi < dynamic_rsi_low else base_leverage
            base_leverage = base_leverage * leverage_factors['long']['decrease'] if current_rsi > dynamic_rsi_high else base_leverage
            base_leverage = base_leverage * leverage_factors['long']['increase'] if current_macd > 0 else base_leverage
            base_leverage = base_leverage * leverage_factors['long']['decrease'] if current_rate < current_sma else base_leverage
        elif side == "short":
            base_leverage = base_leverage * leverage_factors['short']['increase'] if current_rsi > dynamic_rsi_high else base_leverage 
            base_leverage = base_leverage * leverage_factors['short']['decrease'] if current_rsi < dynamic_rsi_low else base_leverage
            base_leverage = base_leverage * leverage_factors['short']['increase'] if current_macd < 0 else base_leverage
            base_leverage = base_leverage * leverage_factors['short']['decrease'] if current_rate > current_sma else base_leverage

        adjusted_leverage = max(min(base_leverage, max_leverage), 1.0)

        return adjusted_leverage
