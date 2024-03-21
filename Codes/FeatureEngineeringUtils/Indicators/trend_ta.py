import numpy as np
import pandas as pd

from Codes.FeatureEngineeringUtils.Indicators.ta_utils import IndicatorMixin, _ema, _get_min_max, _sma


class AroonIndicator(IndicatorMixin):
    """Aroon Indicator

    Identify when trends are likely to change direction.

    Aroon Up = ((N - Days Since N-day High) / N) x 100
    Aroon Down = ((N - Days Since N-day Low) / N) x 100
    Aroon Indicator = Aroon Up - Aroon Down

    https://www.investopedia.com/terms/a/aroon.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, window: int = 25, fillna: bool = False):
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods = 0 if self._fillna else self._window
        rolling_close = self._close.rolling(self._window, min_periods=min_periods)
        self._aroon_up = rolling_close.apply(
            lambda x: float(np.argmax(x) + 1) / self._window * 100, raw=True
        )
        self._aroon_down = rolling_close.apply(
            lambda x: float(np.argmin(x) + 1) / self._window * 100, raw=True
        )

    def aroon_up(self) -> pd.Series:
        """Aroon Up Channel

        Returns:
            pandas.Series: New feature generated.
        """
        aroon_up_series = self._check_fillna(self._aroon_up, value=0)
        return pd.Series(aroon_up_series, name=f"aroon_up_{self._window}")

    def aroon_down(self) -> pd.Series:
        """Aroon Down Channel

        Returns:
            pandas.Series: New feature generated.
        """
        aroon_down_series = self._check_fillna(self._aroon_down, value=0)
        return pd.Series(aroon_down_series, name=f"aroon_down_{self._window}")


class MACD(IndicatorMixin):
    """Moving Average Convergence Divergence (MACD)

    Is a trend-following momentum indicator that shows the relationship between
    two moving averages of prices.

    https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd

    Args:
        close(pandas.Series): dataset 'Close' column.
        window_fast(int): n period short-term.
        window_slow(int): n period long-term.
        window_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
            self,
            close: pd.Series,
            opn: pd.Series,
            window_slow: int = 26,
            window_fast: int = 12,
            window_sign: int = 9,
            fillna: bool = False,
    ):
        self._close = close
        self._open = opn
        self._window_slow = window_slow
        self._window_fast = window_fast
        self._window_sign = window_sign
        self._fillna = fillna
        self._run()

    def _run(self):
        self._emafast = _ema(self._close, self._window_fast, self._fillna)
        self._emaslow = _ema(self._close, self._window_slow, self._fillna)

        self._macd = self._emafast - self._emaslow
        self._macd_signal = _ema(self._macd, self._window_sign, self._fillna)
        self._macd_diff = (self._macd - self._macd_signal) / self._macd_signal
        self._s_macd = self._macd / self._emaslow
        self._s_macd_signal = _ema(self._s_macd, self._window_sign, self._fillna)

    def s_macd(self) -> pd.Series:
        macd_series = self._check_fillna(self._s_macd, value=0)
        return pd.Series(macd_series, name=f"MACD_{self._window_fast}_{self._window_slow}")

    def ns_macd_signal(self):
        ns_signal = self._check_fillna(self._macd_signal, value=0)
        return pd.Series(ns_signal, name=f"ns_macd_signal")

    def s_macd_signal(self) -> pd.Series:
        macd_signal_series = self._check_fillna(self._s_macd_signal, value=0)
        return pd.Series(macd_signal_series, name=f"MACD_sign_{self._window_fast}_{self._window_slow}")

    def s_macd_diff(self) -> pd.Series:
        macd_diff_series = self._check_fillna(self._macd_diff, value=0)
        return pd.Series(macd_diff_series, name=f"MACD_diff_{self._window_fast}_{self._window_slow}")


class EMAIndicator(IndicatorMixin):
    """EMA - Exponential Moving Average

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, opn: pd.Series, window: int = 14, fillna: bool = False):
        self._close = close
        self._open = opn
        self._window = window
        self._fillna = fillna
        self.ema = _ema(self._close, self._window, self._fillna)

    def ema_indicator(self) -> pd.Series:
        """Exponential Moving Average (EMA)

        Returns:
            pandas.Series: New feature generated.
        """
        scaled_ema = (self.ema - self._open) / self._open
        return pd.Series(scaled_ema, name=f"ema_{self._window}")

    def ns_ema_indicator(self) -> pd.Series:
        return pd.Series(self.ema, name=f"ema_{self._window}")


class SMAIndicator(IndicatorMixin):
    """SMA - Simple Moving Average

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, opn: pd.Series, window: int, fillna: bool = False):
        self._close = close
        self._open = opn
        self._window = window
        self._fillna = fillna

    def sma_indicator(self) -> pd.Series:
        """Simple Moving Average (SMA)

        Returns:
            pandas.Series: New feature generated.
        """
        sma_ = _sma(self._close, self._window, self._fillna)
        scaled_sma = (sma_ - self._open) / self._open
        return pd.Series(scaled_sma, name=f"sma_{self._window}")


class WMAIndicator(IndicatorMixin):
    """WMA - Weighted Moving Average

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, window: int = 9, fillna: bool = False):
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        _weight = pd.Series(
            [
                i * 2 / (self._window * (self._window + 1))
                for i in range(1, self._window + 1)
            ]
        )

        def weighted_average(weight):
            def _weighted_average(x):
                return (weight * x).sum()

            return _weighted_average

        self._wma = self._close.rolling(self._window).apply(
            weighted_average(_weight), raw=True
        )

    def wma(self) -> pd.Series:
        """Weighted Moving Average (WMA)

        Returns:
            pandas.Series: New feature generated.
        """
        wma = self._check_fillna(self._wma, value=0)
        return pd.Series(wma, name=f"wma_{self._window}")


class TRIXIndicator(IndicatorMixin):
    """Trix (TRIX)

    Shows the percent rate of change of a triple exponentially smoothed moving
    average.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:trix

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, opn: pd.Series, window: int = 15, fillna: bool = False):
        self._opn = opn
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        ema1 = _ema(self._close, self._window, self._fillna)
        self._dema = _ema(ema1, self._window, self._fillna)
        self._trix = _ema(self._dema, self._window, self._fillna)
        self.s_trix = (self._trix - self._opn) / self._opn

    def dema(self):
        dema_series = self._check_fillna(self._dema, value=0)
        return pd.Series(dema_series, name=f"trix_{self._window}")

    def trix(self) -> pd.Series:
        trix_series = self._check_fillna(self._trix, value=0)
        return pd.Series(trix_series, name=f"trix_{self._window}")

    def strix(self) -> pd.Series:
        strix_series = self._check_fillna(self.s_trix, value=0)
        return pd.Series(strix_series, name=f"trix_{self._window}")


class MassIndex(IndicatorMixin):
    """Mass Index (MI)

    It uses the high-low range to identify trend reversals based on range
    expansions. It identifies range bulges that can foreshadow a reversal of
    the current trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mass_index

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        window_fast(int): fast period value.
        window_slow(int): slow period value.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
            self,
            high: pd.Series,
            low: pd.Series,
            window_fast: int = 9,
            window_slow: int = 25,
            fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._window_fast = window_fast
        self._window_slow = window_slow
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods = 0 if self._fillna else self._window_slow
        amplitude = self._high - self._low
        self._tr_ema = _ema(amplitude, self._window_fast, self._fillna)
        self._tr_dema = _ema(self._tr_ema, self._window_fast, self._fillna)
        self._mass = self._tr_ema / self._tr_dema
        self._mass_index = self._mass.rolling(self._window_slow, min_periods=min_periods).sum() / self._window_slow

    def tr_ema(self):
        tr_ema = self._check_fillna(self._tr_ema)
        return pd.Series(tr_ema, name='trend_tr_ema')

    def tr_dema(self):
        tr_dema = self._check_fillna(self._tr_dema)
        return pd.Series(tr_dema, name='trend_tr_dema')

    def mass(self):
        mass = self._check_fillna(self._mass)
        return pd.Series(mass, name='trend_tr_dema')

    def mass_index(self) -> pd.Series:
        mass_index = self._check_fillna(self._mass_index, value=0)
        return pd.Series(mass_index, name=f"mass_index_{self._window_fast}_{self._window_slow}")

"Span a and b require paranthesis"
class IchimokuIndicator(IndicatorMixin):
    """Ichimoku Kinkō Hyō (Ichimoku)

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ichimoku_cloud

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        window1(int): n1 low period.
        window2(int): n2 medium period.
        window3(int): n3 high period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
            self,
            high: pd.Series,
            low: pd.Series,
            opn: pd.Series,
            window1: int = 9,
            window2: int = 26,
            window3: int = 52,
            fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._open = opn
        self._window1 = window1
        self._window2 = window2
        self._window3 = window3
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods_n1 = 0 if self._fillna else self._window1
        min_periods_n2 = 0 if self._fillna else self._window2
        self._conv = 0.5 * (
                self._high.rolling(self._window1, min_periods=min_periods_n1).max()
                + self._low.rolling(self._window1, min_periods=min_periods_n1).min()
        )
        self._base = 0.5 * (
                self._high.rolling(self._window2, min_periods=min_periods_n2).max()
                + self._low.rolling(self._window2, min_periods=min_periods_n2).min()
        )

        self._spana = 0.5 * (self._conv + self._base)

        self._spanb = 0.5 * (
                self._high.rolling(self._window3, min_periods=0).max()
                + self._low.rolling(self._window3, min_periods=0).min()
        )

    def ichimoku_conversion_line(self) -> pd.Series:
        conversion = self._check_fillna(self._conv, value=-1)
        scaled_conversion = (conversion - self._open) / self._open
        return pd.Series(
            scaled_conversion, name=f"ichimoku_conv_{self._window1}_{self._window2}"
        )

    def ichimoku_base_line(self) -> pd.Series:
        base = self._check_fillna(self._base, value=-1)
        scaled_base = (base - self._open) / self._open
        return pd.Series(scaled_base, name=f"ichimoku_base_{self._window1}_{self._window2}")

    def lead_a(self):
        lead_a = self._check_fillna(self._spana, value=-1)
        return pd.Series(lead_a, name='lead_a')

    def lead_b(self):
        lead_b = self._check_fillna(self._spanb, value=-1)
        return pd.Series(lead_b, name='lead_b')

    def visual_ichimoku_a(self) -> pd.Series:
        spana = self._spana.shift(self._window2)
        spana = self._check_fillna(spana, value=-1)
        scaled_spana = (spana - self._open) / self._open
        return pd.Series(scaled_spana, name=f"ichimoku_a_{self._window1}_{self._window2}")

    def visual_ichimoku_b(self) -> pd.Series:
        """Senkou Span B (Leading Span B)

        Returns:
            pandas.Series: New feature generated.
        """
        spanb = self._spanb.shift(self._window2)
        spanb = self._check_fillna(spanb, value=-1)
        scaled_spanb = (spanb - self._open) / self._open
        return pd.Series(scaled_spanb, name=f"ichimoku_b_{self._window1}_{self._window2}")

    @staticmethod
    def ichimoku_a(_conv, _base, _open, window2, steps):
        """Senkou Span A (Leading Span A)

        Returns:
            pandas.Series: New feature generated.
        """
        spana = 0.5 * (_conv + _base)
        for i in range(steps):
            spana.loc[max(spana.index) + 1] = None
        spana = spana.shift(window2 - steps, fill_value=spana.mean())
        spana.fillna(method='ffill', inplace=True)
        scaled_spana = pd.DataFrame(
            [spana[i: i + steps] - _open[i + steps - 1] / _open[i + steps - 1] for i in range(len(_open))],
            columns=[f'minus_{i}' for i in range(steps - 1, -1, -1)])
        return scaled_spana

    @staticmethod
    def ichimoku_b(_high, _low, _open, window2, window3, steps):
        """Senkou Span B (Leading Span B)

        Returns:
            pandas.Series: New feature generated.
        """
        spanb = 0.5 * (_high.rolling(window3, min_periods=0).max() + _low.rolling(window3, min_periods=0).min())
        for i in range(steps):
            spanb.loc[max(spanb.index) + 1] = None
        spanb = spanb.shift(window2 - steps, fill_value=spanb.mean())
        spanb.fillna(method='ffill', inplace=True)
        scaled_spanb = pd.DataFrame(
            [spanb[i: i + steps] - _open[i + steps - 1] / _open[i + steps - 1] for i in range(len(_open))],
            columns=[f'minus_{i}' for i in range(steps - 1, -1, -1)])
        return scaled_spanb


class KSTIndicator(IndicatorMixin):
    """KST Oscillator (KST Signal)

    It is useful to identify major stock market cycle junctures because its
    formula is weighed to be more greatly influenced by the longer and more
    dominant time spans, in order to better reflect the primary swings of stock
    market cycle.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:know_sure_thing_kst

    Args:
        close(pandas.Series): dataset 'Close' column.
        roc1(int): roc1 period.
        roc2(int): roc2 period.
        roc3(int): roc3 period.
        roc4(int): roc4 period.
        window1(int): n1 smoothed period.
        window2(int): n2 smoothed period.
        window3(int): n3 smoothed period.
        window4(int): n4 smoothed period.
        nsig(int): n period to signal.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
            self,
            close: pd.Series,
            roc1: int = 10,
            roc2: int = 15,
            roc3: int = 20,
            roc4: int = 30,
            window1: int = 10,
            window2: int = 10,
            window3: int = 10,
            window4: int = 15,
            nsig: int = 9,
            fillna: bool = False,
    ):
        self._close = close
        self._r1 = roc1
        self._r2 = roc2
        self._r3 = roc3
        self._r4 = roc4
        self._window1 = window1
        self._window2 = window2
        self._window3 = window3
        self._window4 = window4
        self._nsig = nsig
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods_n1 = 0 if self._fillna else self._window1
        min_periods_n2 = 0 if self._fillna else self._window2
        min_periods_n3 = 0 if self._fillna else self._window3
        min_periods_n4 = 0 if self._fillna else self._window4

        self._roc10 = (self._close - self._close.shift(self._r1)) / self._close.shift(self._r1)
        self._roc10 = self._roc10.replace([np.inf], 1)
        self._roc10 = self._roc10.replace([-np.inf], -1)
        rocma1 = self._roc10.rolling(self._window1, min_periods=min_periods_n1).mean()

        self._roc15 = (self._close - self._close.shift(self._r2)) / self._close.shift(self._r2)
        self._roc15 = self._roc15.replace([np.inf], 1)
        self._roc15 = self._roc15.replace([-np.inf], -1)
        rocma2 = self._roc15.rolling(self._window2, min_periods=min_periods_n2).mean()

        self._roc20 = (self._close - self._close.shift(self._r3)) / self._close.shift(self._r3)
        self._roc20 = self._roc20.replace([np.inf], 1)
        self._roc20 = self._roc20.replace([-np.inf], -1)
        rocma3 = self._roc20.rolling(self._window3, min_periods=min_periods_n3).mean()

        self._roc30 = (self._close - self._close.shift(self._r4)) / self._close.shift(self._r4)
        self._roc30 = self._roc30.replace([np.inf], 1)
        self._roc30 = self._roc30.replace([-np.inf], -1)
        rocma4 = self._roc30.rolling(self._window4, min_periods=min_periods_n4).mean()

        self._kst = 10 * (rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4)
        self._kst_sig = self._kst.rolling(self._nsig, min_periods=0).mean()

    def roc10(self):
        roc10 = self._check_fillna(self._roc10, value=0)
        return pd.Series(roc10, name='roc10')

    def roc15(self):
        roc15 = self._check_fillna(self._roc15, value=0)
        return pd.Series(roc15, name='roc15')

    def roc20(self):
        roc20 = self._check_fillna(self._roc20, value=0)
        return pd.Series(roc20, name='roc20')

    def roc30(self):
        roc30 = self._check_fillna(self._roc30, value=0)
        return pd.Series(roc30, name='roc30')

    def kst(self) -> pd.Series:
        """Know Sure Thing (KST)

        Returns:
            pandas.Series: New feature generated.
        """
        kst_series = self._check_fillna(self._kst, value=0)
        return pd.Series(kst_series, name="kst")

    def kst_sig(self) -> pd.Series:
        """Signal Line Know Sure Thing (KST)

        nsig-period SMA of KST

        Returns:
            pandas.Series: New feature generated.
        """
        kst_sig_series = self._check_fillna(self._kst_sig, value=0)
        return pd.Series(kst_sig_series, name="kst_sig")


class DPOIndicator(IndicatorMixin):
    """Detrended Price Oscillator (DPO)

    Is an indicator designed to remove trend from price and make it easier to
    identify cycles.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:detrended_price_osci

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, window: int = 20, fillna: bool = False):
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods = 0 if self._fillna else self._window
        n_period_sma = self._close.rolling(self._window, min_periods=min_periods).mean()
        n_period_sma = n_period_sma.replace([0], 0.0001)
        self._dpo = (self._close.shift(int((0.5 * self._window) + 1), fill_value=0) - n_period_sma) / n_period_sma

    def dpo(self) -> pd.Series:
        """Detrended Price Oscillator (DPO)

        Returns:
            pandas.Series: New feature generated.
        """
        dpo_ = self._check_fillna(self._dpo, value=0)
        scaled_dpo = dpo_
        return pd.Series(scaled_dpo, name="dpo_" + str(self._window))


class CCIIndicator(IndicatorMixin):
    """Commodity Channel Index (CCI)

    CCI measures the difference between a security's price change and its
    average price change. High positive readings indicate that prices are well
    above their average, which is a show of strength. Low negative readings
    indicate that prices are well below their average, which is a show of
    weakness.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        constant(int): constant.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
            self,
            high: pd.Series,
            low: pd.Series,
            close: pd.Series,
            window: int = 20,
            constant: float = 0.015,
            fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._close = close
        self._window = window
        self._constant = constant
        self._fillna = fillna
        self._run()

    def _run(self):
        def _mad(x):
            return np.mean(np.abs(x - np.mean(x)))

        min_periods = 0 if self._fillna else self._window
        typical_price = (self._high + self._low + self._close) / 3.0
        self._tp = typical_price
        self._cci = (typical_price - typical_price.rolling(self._window, min_periods=min_periods).mean()) / (
                self._constant * typical_price.rolling(self._window, min_periods=min_periods).apply(_mad,
                                                                                                    True)
        )

    def tp(self):
        tp = self._check_fillna(self._tp, value=0)
        return pd.Series(tp, name="tp")

    def cci(self) -> pd.Series:
        """Commodity Channel Index (CCI)

        Returns:
            pandas.Series: New feature generated.
        """
        cci_series = self._check_fillna(self._cci, value=0)
        return pd.Series(cci_series, name="cci")


class ADXIndicator(IndicatorMixin):
    """Average Directional Movement Index (ADX)

    The Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    are derived from smoothed averages of these differences, and measure trend
    direction over time. These two indicators are often referred to
    collectively as the Directional Movement Indicator (DMI).

    The Average Directional Index (ADX) is in turn derived from the smoothed
    averages of the difference between +DI and -DI, and measures the strength
    of the trend (regardless of direction) over time.

    Using these three indicators together, chartists can determine both the
    direction and strength of the trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
            self,
            high: pd.Series,
            low: pd.Series,
            close: pd.Series,
            window: int = 14,
            fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        if self._window == 0:
            raise ValueError("window may not be 0")

        close_shift = self._close.shift(1)
        pdm = _get_min_max(self._high, close_shift, "max")
        pdn = _get_min_max(self._low, close_shift, "min")
        diff_directional_movement = pdm - pdn
        ddm = diff_directional_movement.reset_index(drop=True)

        self._trs = np.ones(len(self._close))
        self._trs[self._window] = ddm[1: self._window+1].mean()


        diff_up = self._high - self._high.shift(1)
        diff_down = self._low.shift(1) - self._low
        pos = ((diff_up > diff_down) & (diff_up > 0)) * diff_up
        pos = pos.reset_index(drop=True)
        neg = ((diff_down > diff_up) & (diff_down > 0)) * diff_down
        neg = neg.reset_index(drop=True)

        self._pdi = np.zeros(len(self._close))
        self._pdi[self._window] = pos[1: self._window+1].mean()


        self._ndi = np.zeros(len(self._close))
        self._ndi[self._window] = neg[1: self._window+1].mean()

        w = self._window - 1
        for i in range(self._window + 1, len(self._close), 1):
            self._trs[i] = (w * self._trs[i - 1] + ddm[i]) / self._window

            self._pdi[i] = (w * self._pdi[i - 1] + pos[i]) / self._window

            self._ndi[i] = (w * self._ndi[i - 1] + neg[i]) / self._window
        d1 = np.where(self._trs == 0, self._pdi/10 + 0.00001, self._trs)
        d2 = np.where(self._trs == 0, self._ndi/10 + 0.00001, self._trs)
        self._spdi = 100 * self._pdi / d1
        self._sndi = 100 * self._ndi / d2


    def adx(self) -> pd.Series:
        """Average Directional Index (ADX)

        Returns:
            pandas.Series: New feature generated.tr
        """
        _pdi = self._pdi

        _ndi = self._ndi

        denum = _pdi + _ndi
        for i in range(100):
            denum[i] = denum[i] if denum[i] != 0 else 0.1

        directional_index = 100 * np.abs((_pdi - _ndi) / denum)

        adx_series = np.zeros(len(self._close))
        adx_series[2 * self._window - 1] = directional_index[self._window: 2 * self._window].mean()

        w = self._window - 1
        for i in range(2 * self._window, len(adx_series)):
            adx_series[i] = (w * adx_series[i - 1] + directional_index[i]) / self._window

        adx_series = self._check_fillna(pd.Series(adx_series, index=self._close.index), value=20)
        return pd.Series(adx_series, name="adx")

    def pdi(self) -> pd.Series:
        adx_pos_series = self._check_fillna(pd.Series(self._pdi, index=self._close.index), value=-1)
        return pd.Series(adx_pos_series, name="adx_pos")

    def spdi(self) -> pd.Series:
        adx_pos_series = self._check_fillna(pd.Series(self._spdi, index=self._close.index), value=-1)
        return pd.Series(adx_pos_series, name="adx_pos")

    def ndi(self) -> pd.Series:
        adx_neg_series = self._check_fillna(pd.Series(self._ndi, index=self._close.index), value=-1)
        return pd.Series(adx_neg_series, name="adx_neg")

    def sndi(self) -> pd.Series:
        adx_neg_series = self._check_fillna(pd.Series(self._sndi, index=self._close.index), value=-1)
        return pd.Series(adx_neg_series, name="adx_neg")

    def atr(self):
        atr = self._check_fillna(pd.Series(self._trs, index=self._close.index), value=-1)
        return pd.Series(atr, name="atr")

"replace inf with proper value"
class VortexIndicator(IndicatorMixin):
    """Vortex Indicator (VI)

    It consists of two oscillators that capture positive and negative trend
    movement. A bullish signal triggers when the positive trend indicator
    crosses above the negative trend indicator or a key level.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vortex_indicator

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
            self,
            high: pd.Series,
            low: pd.Series,
            close: pd.Series,
            window: int = 14,
            fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        close_shift = self._close.shift(1, fill_value=self._close.mean())
        true_range = self._true_range(self._high, self._low, close_shift)
        min_periods = 0 if self._fillna else self._window
        trn = true_range.rolling(self._window, min_periods=min_periods).sum()
        trn = trn.fillna(method='ffill')
        self._vmp = np.abs(self._high - self._low.shift(1))
        self._vmn = np.abs(self._low - self._high.shift(1))
        self._vip = self._vmp.rolling(self._window, min_periods=min_periods).sum() / trn
        self._vip = self._vip.replace([np.inf, -np.inf], 1)
        self._vin = self._vmn.rolling(self._window, min_periods=min_periods).sum() / trn
        self._vin = self._vin.replace([np.inf, -np.inf], 1)

    def vortex_indicator_pos(self):
        """+VI

        Returns:
            pandas.Series: New feature generated.
        """
        vip = self._check_fillna(self._vip, value=1)
        return pd.Series(vip, name="vip")

    def vortex_indicator_neg(self):
        """-VI

        Returns:
            pandas.Series: New feature generated.
        """
        vin = self._check_fillna(self._vin, value=1)
        return pd.Series(vin, name="vin")

    def p_vm(self):
        p_vm = self._check_fillna(self._vmp)
        return pd.Series(p_vm, name="p_vm")

    def n_vm(self):
        n_vm = self._check_fillna(self._vmn)
        return pd.Series(n_vm, name="n_vm")


class PSARIndicator(IndicatorMixin):
    """Parabolic Stop and Reverse (Parabolic SAR)

    The Parabolic Stop and Reverse, more commonly known as the
    Parabolic SAR,is a trend-following indicator developed by
    J. Welles Wilder. The Parabolic SAR is displayed as a single
    parabolic line (or dots) underneath the price bars in an uptrend,
    and above the price bars in a downtrend.

    https://school.stockcharts.com/doku.php?id=technical_indicators:parabolic_sar

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        step(float): the Acceleration Factor used to compute the SAR.
        max_step(float): the maximum value allowed for the Acceleration Factor.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
            self,
            high: pd.Series,
            low: pd.Series,
            close: pd.Series,
            opn: pd.Series,
            step: float = 0.02,
            max_step: float = 0.20,
            fillna: bool = False,
    ):
        self._high = high.copy()
        self._low = low.copy()
        self._close = close
        self._open = opn
        self._step = step
        self._max_step = max_step
        self._fillna = fillna
        self._run()

    def _run(self):  # noqa
        up_trend = True
        acceleration_factor = self._step
        ep = self._high.iloc[0]

        self._ep = pd.Series(index=self._close.index, dtype='float64')
        self._af = pd.Series(index=self._close.index, dtype='float64')
        self._psar = self._close.copy()
        self._up_trend = pd.Series(index=self._close.index, dtype='float64')
        self._indicator = pd.Series(0, index=self._close.index)

        for i in range(2, len(self._close)):
            reversal = False
            max_high = self._high.iloc[i]
            min_low = self._low.iloc[i]

            if up_trend:
                self._psar.iloc[i] = self._psar.iloc[i - 1] + (acceleration_factor * (ep - self._psar.iloc[i - 1]))
                if min_low < self._psar.iloc[i]:
                    reversal = True
                    self._indicator[i] = -1
                    self._psar.iloc[i] = ep
                    ep = min_low
                    acceleration_factor = self._step
                else:
                    if max_high > ep:
                        ep = max_high
                        acceleration_factor = min(acceleration_factor + self._step, self._max_step)
                    top = min(self._low.iloc[i - 1], self._low.iloc[i - 2])
                    if top < self._psar.iloc[i]:
                        self._psar.iloc[i] = top
            else:
                self._psar.iloc[i] = self._psar.iloc[i - 1] - (
                        acceleration_factor * (self._psar.iloc[i - 1] - ep)
                )

                if max_high > self._psar.iloc[i]:
                    reversal = True
                    self._indicator[i] = 1
                    self._psar.iloc[i] = ep
                    ep = max_high
                    acceleration_factor = self._step
                else:
                    if min_low < ep:
                        ep = min_low
                        acceleration_factor = min(acceleration_factor + self._step, self._max_step)

                    bottom = max(self._high.iloc[i - 1], self._high.iloc[i - 2])
                    if bottom > self._psar.iloc[i]:
                        self._psar[i] = bottom

            self._ep[i] = ep
            self._af[i] = acceleration_factor
            up_trend = up_trend != reversal  # XOR
            self._up_trend[i] = up_trend

    def ep(self):
        ep = self._check_fillna(self._ep, value=-1)
        return pd.Series(ep, name='ep')

    def af(self):
        af = self._check_fillna(self._af, value=-1)
        return pd.Series(af, name='af')

    def ns_psar(self):
        ns = self._check_fillna(self._psar, value=-1)
        return pd.Series(ns, name='ns')

    def psar(self) -> pd.Series:
        psar_series = self._check_fillna(self._psar, value=-1)
        scaled_psar = (psar_series - self._open) / self._open
        return pd.Series(scaled_psar, name="psar")

    def up_trend(self):
        up = self._check_fillna(self._up_trend, value=-1)
        return pd.Series(up, name='up')

    def indicator(self):
        indi = self._check_fillna(self._indicator, value=-1)
        return pd.Series(indi, name='indicator')


class STCIndicator(IndicatorMixin):
    """Schaff Trend Cycle (STC)

    The Schaff Trend Cycle (STC) is a charting indicator that
    is commonly used to identify market trends and provide buy
    and sell signals to traders. Developed in 1999 by noted currency
    trader Doug Schaff, STC is a type of oscillator and is based on
    the assumption that, regardless of time frame, currency trends
    accelerate and decelerate in cyclical patterns.

    https://www.investopedia.com/articles/forex/10/schaff-trend-cycle-indicator.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        window_fast(int): n period short-term.
        window_slow(int): n period long-term.
        cycle(int): cycle size
        smooth1(int): ema period over stoch_k
        smooth2(int): ema period over stoch_kd
        fillna(bool): if True, fill nan values.
    """

    def __init__(
            self,
            close: pd.Series,
            window_slow: int = 50,
            window_fast: int = 23,
            cycle: int = 10,
            smooth1: int = 3,
            smooth2: int = 3,
            fillna: bool = False,
    ):
        self._close = close
        self._window_slow = window_slow
        self._window_fast = window_fast
        self._cycle = cycle
        self._smooth1 = smooth1
        self._smooth2 = smooth2
        self._fillna = fillna
        self._run()

    def _run(self):
        _emafast = _ema(self._close, self._window_fast, self._fillna)
        _emaslow = _ema(self._close, self._window_slow, self._fillna)
        self._ns_macd = _emafast - _emaslow

        _macdmin = self._ns_macd.rolling(window=self._cycle).min()
        _macdmax = self._ns_macd.rolling(window=self._cycle).max()
        self._stoch_k = 100 * (self._ns_macd - _macdmin) / (_macdmax - _macdmin)
        self._stoch_d = _ema(self._stoch_k, self._smooth1, self._fillna)
        _stoch_d_min = self._stoch_d.rolling(window=self._cycle).min()
        _stoch_d_max = self._stoch_d.rolling(window=self._cycle).max()
        self._stoch_kd = 100 * (self._stoch_d - _stoch_d_min) / (_stoch_d_max - _stoch_d_min)
        self._stc = _ema(self._stoch_kd, self._smooth2, self._fillna)

    def ns_macd(self):
        ns_macd = self._check_fillna(self._ns_macd)
        return pd.Series(ns_macd, name='ns_macd')

    def stoch_k(self):
        stoch_k = self._check_fillna(self._stoch_k)
        return pd.Series(stoch_k, name='stoch_k')

    def stoch_d(self):
        stoch_d = self._check_fillna(self._stoch_d)
        return pd.Series(stoch_d, name='stoch_d')

    def stoch_kd(self):
        stoch_kd = self._check_fillna(self._stoch_kd)
        return pd.Series(stoch_kd, name='stoch_kd')

    def stc(self):
        stc_series = self._check_fillna(self._stc, value=-1)
        return pd.Series(stc_series, name="stc")


class MinMaxIndicator(IndicatorMixin):
    """MinMax trend

        Args:
            high(pandas.Series): dataset 'High' column.
            low(pandas.Series): dataset 'Low' column.
            window1(int): n1 low period.
            window2(int): n2 medium period.
            window3(int): n3 high period.
            fillna(bool): if True, fill nan values.
        """

    def __init__(
            self,
            high: pd.Series,
            low: pd.Series,
            opn: pd.Series,
            patr: pd.Series,
            rsi: pd.Series,
            window1: int = 9,
            window2: int = 26,
            window3: int = 52,
            fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._open = opn
        self.patr = patr
        self.rsi = rsi
        self.time_delta = high.index[1] - high.index[0]
        self._window1 = window1
        self._window2 = window2
        self._window3 = window3
        self._fillna = fillna
        self._run()

    def _run(self):
        _res1 = []
        _a_res1 = []
        _r_res1 = []
        _res2 = []
        _a_res2 = []
        _r_res2 = []
        _res3 = []
        _a_res3 = []
        _r_res3 = []
        _sup1 = []
        _a_sup1 = []
        _r_sup1 = []
        _sup2 = []
        _a_sup2 = []
        _r_sup2 = []
        _sup3 = []
        _a_sup3 = []
        _r_sup3 = []

        d_sup1 = []
        d_sup2 = []
        d_sup3 = []
        d_res1 = []
        d_res2 = []
        d_res3 = []
        d_res_indicator = []

        g_sup1 = []
        g_res1 = []
        g_res_indicator = []

        d_sup_indicator = []


        res = pd.DataFrame(columns=['res_base', 'res_line', 'res_rsi'])
        sup = pd.DataFrame(columns=['sup_base', 'sup_line', 'sup_rsi'])

        gres = pd.DataFrame(columns=['res_base'])
        gsup = pd.DataFrame(columns=['sup_base'])

        dres = pd.DataFrame(columns=['res_base'])
        dsup = pd.DataFrame(columns=['sup_base'])

        latest_point_is_low = True
        threshold_coef = 3
        threshold = threshold_coef * self.patr[0]
        pre_latest_point = {'high': self._high[0], 'low': self._low[0]}
        latest_point = (0, self._low[0])
        highest_high_rsi = self.rsi[0]
        pre_highest_high = self._low[0]
        highest_high = self._high[0]
        lowest_low_rsi = self.rsi[0]
        pre_lowest_low = self._high[0]
        lowest_low = self._low[0]

        for i in range(len(self._open)):
            if self._low[i] < lowest_low:
                lowest_low_rsi = self.rsi[i]
                pre_lowest_low = min(self._high[i - 1], self._high[i])
                lowest_low = self._low[i]
            if self._high[i] > highest_high:
                highest_high_rsi = self.rsi[i]
                pre_highest_high = max(self._low[i - 1], self._low[i])
                highest_high = self._high[i]

            if latest_point_is_low:
                if self._low[i] < latest_point[1]:
                    pre_latest_point['high'], pre_latest_point['low'] = min(self._high[i - 1], self._high[i]), max(self._low[i - 1], self._low[i])
                    latest_point = (i, self._low[i])
                    threshold = threshold_coef * self.patr[i]
                if (latest_point[1] * (1 + threshold)) <= self._high[i]:
                    sup.loc[i] = [latest_point[1], pre_latest_point['high'], self.rsi[i]]
                    sup.sort_values('sup_base', ascending=False, inplace=True)
                    dsup.loc[i] = [latest_point[1]]
                    dsup.sort_values('sup_base', ascending=False, inplace=True)
                    pre_latest_point['high'], pre_latest_point['low'] = min(self._high[i - 1], self._high[i]) , max(self._low[i - 1], self._low[i])
                    latest_point = (i, self._high[i])
                    threshold = threshold_coef * self.patr[i]
                    latest_point_is_low = False
            else:
                if self._high[i] > latest_point[1]:
                    pre_latest_point['high'], pre_latest_point['low'] = min(self._high[i - 1], self._high[i]), max(self._low[i - 1], self._low[i])
                    latest_point = (i, self._high[i])
                    threshold = threshold_coef * self.patr[i]
                if (latest_point[1] * (1 - threshold)) >= self._low[i]:
                    res.loc[i] = [latest_point[1], pre_latest_point['low'], self.rsi[i]]
                    res.sort_values('res_base', ascending=True, inplace=True)
                    dres.loc[i] = [latest_point[1]]
                    dres.sort_values('res_base', ascending=True, inplace=True)
                    pre_latest_point['high'], pre_latest_point['low'] = min(self._high[i - 1], self._high[i]), max(self._low[i - 1], self._low[i])
                    latest_point = (i, self._low[i])
                    threshold = threshold_coef * self.patr[i]
                    latest_point_is_low = True

            gsup = pd.concat([gsup, sup[sup.sup_base >= self._low[i]]], axis=0).copy()
            gsup.drop_duplicates(inplace=True)
            sup = sup[sup.sup_base <= self._low[i]]

            gres = pd.concat([gres, res[res.res_base <= self._high[i]]], axis=0).copy()
            gres.drop_duplicates(inplace=True)
            res = res[res.res_base >= self._high[i]]

            if len(res) >= 3:
                _res1.append(res.iloc[0, 0])
                _a_res1.append(res.iloc[0, 1])
                _r_res1.append(res.iloc[0, 2])
                _res2.append(res.iloc[1, 0])
                _a_res2.append(res.iloc[1, 1])
                _r_res2.append(res.iloc[1, 2])
                _res3.append(res.iloc[2, 0])
                _a_res3.append(res.iloc[2, 1])
                _r_res3.append(res.iloc[2, 2])
            elif len(res) >= 2:
                _res1.append(res.iloc[0, 0])
                _a_res1.append(res.iloc[0, 1])
                _r_res1.append(res.iloc[0, 2])
                _res2.append(res.iloc[1, 0])
                _a_res2.append(res.iloc[1, 1])
                _r_res2.append(res.iloc[1, 2])
                _res3.append(highest_high)
                _a_res3.append(pre_highest_high)
                _r_res3.append(highest_high_rsi)
            elif len(res) >= 1:
                _res1.append(res.iloc[0, 0])
                _a_res1.append(res.iloc[0, 1])
                _r_res1.append(res.iloc[0, 2])
                _res2.append(res.iloc[0, 0])
                _a_res2.append(res.iloc[0, 1])
                _r_res2.append(res.iloc[0, 2])
                _res3.append(highest_high)
                _a_res3.append(pre_highest_high)
                _r_res3.append(highest_high_rsi)
            else:
                _res1.append(highest_high)
                _a_res1.append(pre_highest_high)
                _r_res1.append(highest_high_rsi)
                _res2.append(highest_high)
                _a_res2.append(pre_highest_high)
                _r_res2.append(highest_high_rsi)
                _res3.append(highest_high)
                _a_res3.append(pre_highest_high)
                _r_res3.append(highest_high_rsi)

            if len(sup) >= 3:
                _sup1.append(sup.iloc[0, 0])
                _a_sup1.append(sup.iloc[0, 1])
                _r_sup1.append(sup.iloc[0, 2])
                _sup2.append(sup.iloc[1, 0])
                _a_sup2.append(sup.iloc[1, 1])
                _r_sup2.append(sup.iloc[1, 2])
                _sup3.append(sup.iloc[2, 0])
                _a_sup3.append(sup.iloc[2, 1])
                _r_sup3.append(sup.iloc[2, 2])
            elif len(sup) >= 2:
                _sup1.append(sup.iloc[0, 0])
                _a_sup1.append(sup.iloc[0, 1])
                _r_sup1.append(sup.iloc[0, 2])
                _sup2.append(sup.iloc[1, 0])
                _a_sup2.append(sup.iloc[1, 1])
                _r_sup2.append(sup.iloc[1, 2])
                _sup3.append(lowest_low)
                _a_sup3.append(pre_lowest_low)
                _r_sup3.append(lowest_low_rsi)
            elif len(sup) >= 1:
                _sup1.append(sup.iloc[0, 0])
                _a_sup1.append(sup.iloc[0, 1])
                _r_sup1.append(sup.iloc[0, 2])
                _sup2.append(sup.iloc[0, 0])
                _a_sup2.append(sup.iloc[0, 1])
                _r_sup2.append(sup.iloc[0, 2])
                _sup3.append(lowest_low)
                _a_sup3.append(pre_lowest_low)
                _r_sup3.append(lowest_low_rsi)
            else:
                _sup1.append(lowest_low)
                _a_sup1.append(pre_lowest_low)
                _r_sup1.append(lowest_low_rsi)
                _sup2.append(lowest_low)
                _a_sup2.append(pre_lowest_low)
                _r_sup2.append(lowest_low_rsi)
                _sup3.append(lowest_low)
                _a_sup3.append(pre_lowest_low)
                _r_sup3.append(lowest_low_rsi)

            dsup = dsup[dsup.sup_base <= self._low[i]]
            dres = dres[dres.res_base >= self._high[i]]

            " dynamic support "
            if len(dsup) >= 1 and latest_point_is_low:
                if len(dsup) >= 2:
                    d = (dsup.iloc[0, 0] - dsup.iloc[1, 0]) / (dsup.iloc[0, :].name - dsup.iloc[1, :].name)
                    dynamic_sup = dsup.iloc[0, 0] + (i - dsup.iloc[0, :].name) * d
                    if self._low[i] < dynamic_sup:
                        dsup.drop(dsup.iloc[0, :].name)

                d = (latest_point[1] - dsup.iloc[0, 0]) / (latest_point[0] - dsup.iloc[0, :].name)
                dynamic_sup1 = latest_point[1] + (i - latest_point[0] + 1) * d
                dynamic_sup2 = latest_point[1] + (i - latest_point[0] + 3) * d
                dynamic_sup3 = latest_point[1] + (i - latest_point[0] + 7) * d

            elif len(dsup) >= 2:
                if len(dsup) >= 3:
                    d = (dsup.iloc[0, 0] - dsup.iloc[1, 0]) / (dsup.iloc[0, :].name - dsup.iloc[1, :].name)
                    dynamic_sup = dsup.iloc[0, 0] + (i - dsup.iloc[0, :].name) * d
                    if self._low[i] < dynamic_sup:
                        dsup.drop(dsup.iloc[0, :].name)
                        dsup.loc[i] = self._low[i]
                        dsup.sort_values('sup_base', ascending=False, inplace=True)

                d = (dsup.iloc[0, 0] - dsup.iloc[1, 0]) / (dsup.iloc[0, :].name - dsup.iloc[1, :].name)
                dynamic_sup1 = dsup.iloc[0, 0] + (i - dsup.iloc[0, :].name + 1) * d
                dynamic_sup2 = dsup.iloc[0, 0] + (i - dsup.iloc[0, :].name + 3) * d
                dynamic_sup3 = dsup.iloc[0, 0] + (i - dsup.iloc[0, :].name + 7) * d

            else:
                dynamic_sup1 = lowest_low
                dynamic_sup2 = lowest_low
                dynamic_sup3 = lowest_low

            d_sup1.append(dynamic_sup1)
            d_sup2.append(dynamic_sup2)
            d_sup3.append(dynamic_sup3)

            " dynamic resistance "
            if len(dres) >= 1 and not latest_point_is_low:
                if len(dres) >= 2:
                    d = (dres.iloc[0, 0] - dres.iloc[1, 0]) / (dres.iloc[0, :].name - dres.iloc[1, :].name)
                    dynamic_res = dres.iloc[0, 0] + (i - dres.iloc[0, :].name) * d
                    if self._high[i] > dynamic_res:
                        dres.drop(dres.iloc[0, :].name)

                d = (latest_point[1] - dres.iloc[0, 0]) / (latest_point[0] - dres.iloc[0, :].name)
                dynamic_res1 = latest_point[1] + (i - latest_point[0] + 1) * d
                dynamic_res2 = latest_point[1] + (i - latest_point[0] + 3) * d
                dynamic_res3 = latest_point[1] + (i - latest_point[0] + 7) * d

            elif len(dres) >= 2:
                if len(dres) >= 3:
                    d = (dres.iloc[0, 0] - dres.iloc[1, 0]) / (dres.iloc[0, :].name - dres.iloc[1, :].name)
                    dynamic_res = dres.iloc[0, 0] + (i - dres.iloc[0, :].name) * d
                    if self._high[i] > dynamic_res:
                        dres.drop(dres.iloc[0, :].name)
                        dres.loc[i] = self._high[i]
                        dres.sort_values('res_base', ascending=False, inplace=True)

                d = (dres.iloc[0, 0] - dres.iloc[1, 0]) / (dres.iloc[0, :].name - dres.iloc[1, :].name)
                dynamic_res1 = dres.iloc[0, 0] + (i - dres.iloc[0, :].name + 1) * d
                dynamic_res2 = dres.iloc[0, 0] + (i - dres.iloc[0, :].name + 3) * d
                dynamic_res3 = dres.iloc[0, 0] + (i - dres.iloc[0, :].name + 7) * d

            else:
                dynamic_res1 = highest_high
                dynamic_res2 = highest_high
                dynamic_res3 = highest_high

            d_res1.append(dynamic_res1)
            d_res2.append(dynamic_res2)
            d_res3.append(dynamic_res3)

            d_res_indicator.append((self._open[i] > d_res1[i - 1]) if i != 0 else False)

            d_sup_indicator.append(d_sup1[i - 1] > (self._open[i] - (2 * self.patr[i])) if i != 0 else False)

            """ golden support """
            gsup = gsup[(gsup.sup_base < self._open[i]) | (gsup.sup_base < self._open[i-1])]
            gsup.sort_values('sup_base', ascending=False, inplace=True)
            if len(gsup) >= 1:
                g_sup1.append(gsup.iloc[0, 0])
            else:
                g_sup1.append(_sup3[-1])

            """ golden resistance """
            gres = gres[(gres.res_base > self._open[i]) | (gres.res_base > self._open[i - 1])]
            gres.sort_values('res_base', ascending=True, inplace=True)
            if len(gres) >= 1:
                g_res1.append(gres.iloc[0, 0])
            else:
                g_res1.append(_res3[-1])

            if i not in [0, 1]:
                g_res_indicator.append(self._open[i] > g_res1[i])
            else:
                g_res_indicator.append(False)


        self._sup1 = pd.Series(_sup1, index=self._open.index)
        self._a_sup1 = pd.Series(_a_sup1, index=self._open.index)
        self._r_sup1 = pd.Series(_r_sup1, index=self._open.index)
        self._sup2 = pd.Series(_sup2, index=self._open.index)
        self._a_sup2 = pd.Series(_a_sup2, index=self._open.index)
        self._r_sup2 = pd.Series(_r_sup2, index=self._open.index)
        self._sup3 = pd.Series(_sup3, index=self._open.index)
        self._a_sup3 = pd.Series(_a_sup3, index=self._open.index)
        self._r_sup3 = pd.Series(_r_sup3, index=self._open.index)
        self._d_sup1 = pd.Series(d_sup1, index=self._open.index)
        self._d_sup2 = pd.Series(d_sup2, index=self._open.index)
        self._d_sup3 = pd.Series(d_sup3, index=self._open.index)
        self._g_sup1 = pd.Series(g_sup1, index=self._open.index)

        self._res1 = pd.Series(_res1, index=self._open.index)
        self._a_res1 = pd.Series(_a_res1, index=self._open.index)
        self._r_res1 = pd.Series(_r_res1, index=self._open.index)
        self._res2 = pd.Series(_res2, index=self._open.index)
        self._a_res2 = pd.Series(_a_res2, index=self._open.index)
        self._r_res2 = pd.Series(_r_res2, index=self._open.index)
        self._res3 = pd.Series(_res3, index=self._open.index)
        self._a_res3 = pd.Series(_a_res3, index=self._open.index)
        self._r_res3 = pd.Series(_r_res3, index=self._open.index)
        self._d_res1 = pd.Series(d_res1, index=self._open.index)
        self._d_res2 = pd.Series(d_res2, index=self._open.index)
        self._d_res3 = pd.Series(d_res3, index=self._open.index)
        self._g_res1 = pd.Series(g_res1, index=self._open.index)

        self._d_resi = pd.Series(d_res_indicator, index=self._open.index)
        self._d_supi = pd.Series(d_sup_indicator, index=self._open.index)
        self._g_resi = pd.Series(g_res_indicator, index=self._open.index)

    def res1(self) -> pd.Series:
        _res1 = self._check_fillna(self._res1, value=-1)
        scaled_res1 = (_res1 - self._open) / self._open
        return pd.Series(
            scaled_res1, name=f"res_{self._window1}"
        )

    def a_res1(self) -> pd.Series:
        _a_res1 = self._check_fillna(self._a_res1, value=-1)
        scaled_a_res1 = (_a_res1 - self._open) / self._open
        return pd.Series(
            scaled_a_res1, name=f"ares_{self._window1}"
        )

    def r_res1(self):
        _r_res1 = self._check_fillna(self._r_res1, value=-1)
        return pd.Series(
            _r_res1, name=f"rres_{self._window1}"
        )

    def d_res1(self) -> pd.Series:
        _d_res1 = self._check_fillna(self._d_res1, value=-1)
        scaled_d_res1 = (_d_res1 - self._open) / self._open
        return pd.Series(
            scaled_d_res1, name=f"dres_{self._window1}"
        )

    def res2(self) -> pd.Series:
        _res2 = self._check_fillna(self._res2, value=-1)
        scaled_res2 = (_res2 - self._open) / self._open
        return pd.Series(
            scaled_res2, name=f"res_{self._window2}"
        )

    def a_res2(self) -> pd.Series:
        _a_res2 = self._check_fillna(self._a_res2, value=-1)
        scaled_a_res2 = (_a_res2 - self._open) / self._open
        return pd.Series(
            scaled_a_res2, name=f"ares_{self._window2}"
        )

    def r_res2(self):
        _r_res2 = self._check_fillna(self._r_res2, value=-1)
        return pd.Series(
            _r_res2, name=f"rres_{self._window2}"
        )

    def d_res2(self) -> pd.Series:
        _d_res2 = self._check_fillna(self._d_res2, value=-1)
        scaled_d_res2 = (_d_res2 - self._open) / self._open
        return pd.Series(
            scaled_d_res2, name=f"dres_{self._window2}"
        )

    def res3(self) -> pd.Series:
        _res3 = self._check_fillna(self._res3, value=-1)
        scaled_res3 = (_res3 - self._open) / self._open
        return pd.Series(
            scaled_res3, name=f"res_{self._window3}"
        )

    def a_res3(self) -> pd.Series:
        _a_res3 = self._check_fillna(self._a_res3, value=-1)
        scaled_a_res3 = (_a_res3 - self._open) / self._open
        return pd.Series(
            scaled_a_res3, name=f"ares_{self._window3}"
        )

    def r_res3(self):
        _r_res3 = self._check_fillna(self._r_res3, value=-1)
        return pd.Series(
            _r_res3, name=f"rres_{self._window3}"
        )

    def d_res3(self) -> pd.Series:
        _d_res3 = self._check_fillna(self._d_res3, value=-1)
        scaled_d_res3 = (_d_res3 - self._open) / self._open
        return pd.Series(
            scaled_d_res3, name=f"dres_{self._window3}"
        )

    def g_res1(self) -> pd.Series:
        _g_res1 = self._check_fillna(self._g_res1, value=-1)
        scaled_g_res1 = (_g_res1 - self._open) / self._open
        return pd.Series(
            scaled_g_res1, name=f"gres_{self._window1}"
        )

    def d_resi(self) -> pd.Series:
        _d_resi = self._check_fillna(self._d_resi, value=-1)
        return pd.Series(_d_resi, name="d_res_indicator")

    def g_resi(self) -> pd.Series:
        _g_resi = self._check_fillna(self._g_resi, value=-1)
        return pd.Series(_g_resi, name="g_res_indicator")

    def sup1(self) -> pd.Series:
        _sup1 = self._check_fillna(self._sup1, value=-1)
        scaled_sup1 = (_sup1 - self._open) / self._open
        return pd.Series(
            scaled_sup1, name=f"sup_{self._window1}"
        )

    def a_sup1(self) -> pd.Series:
        _a_sup1 = self._check_fillna(self._a_sup1, value=-1)
        scaled_a_sup1 = (_a_sup1 - self._open) / self._open
        return pd.Series(
            scaled_a_sup1, name=f"asup_{self._window1}"
        )

    def r_sup1(self) -> pd.Series:
        _r_sup1 = self._check_fillna(self._r_sup1, value=-1)
        return pd.Series(
            _r_sup1, name=f"rsup_{self._window1}"
        )

    def d_sup1(self) -> pd.Series:
        _d_sup1 = self._check_fillna(self._d_sup1, value=-1)
        scaled_d_sup1 = (_d_sup1 - self._open) / self._open
        return pd.Series(
            scaled_d_sup1, name=f"dsup_{self._window1}"
        )

    def sup2(self) -> pd.Series:
        _sup2 = self._check_fillna(self._sup2, value=-1)
        scaled_sup2 = (_sup2 - self._open) / self._open
        return pd.Series(
            scaled_sup2, name=f"sup_{self._window2}"
        )

    def a_sup2(self) -> pd.Series:
        _a_sup2 = self._check_fillna(self._a_sup2, value=-1)
        scaled_a_sup2 = (_a_sup2 - self._open) / self._open
        return pd.Series(
            scaled_a_sup2, name=f"asup_{self._window2}"
        )

    def r_sup2(self) -> pd.Series:
        _r_sup2 = self._check_fillna(self._r_sup2, value=-1)
        return pd.Series(
            _r_sup2, name=f"rsup_{self._window2}"
        )

    def d_sup2(self) -> pd.Series:
        _d_sup2 = self._check_fillna(self._d_sup2, value=-1)
        scaled_d_sup2 = (_d_sup2 - self._open) / self._open
        return pd.Series(
            scaled_d_sup2, name=f"dsup_{self._window2}"
        )

    def sup3(self) -> pd.Series:
        _sup3 = self._check_fillna(self._sup3, value=-1)
        scaled_sup3 = (_sup3 - self._open) / self._open
        return pd.Series(
            scaled_sup3, name=f"sup_{self._window3}"
        )

    def a_sup3(self) -> pd.Series:
        _a_sup3 = self._check_fillna(self._a_sup3, value=-1)
        scaled_a_sup3 = (_a_sup3 - self._open) / self._open
        return pd.Series(
            scaled_a_sup3, name=f"asup_{self._window3}"
        )

    def r_sup3(self) -> pd.Series:
        _r_sup3 = self._check_fillna(self._r_sup3, value=-1)
        return pd.Series(
            _r_sup3, name=f"rsup_{self._window3}"
        )

    def d_sup3(self) -> pd.Series:
        _d_sup3 = self._check_fillna(self._d_sup3, value=-1)
        scaled_d_sup3 = (_d_sup3 - self._open) / self._open
        return pd.Series(
            scaled_d_sup3, name=f"dsup_{self._window3}"
        )

    def g_sup1(self) -> pd.Series:
        _g_sup1 = self._check_fillna(self._g_sup1, value=-1)
        scaled_g_sup1 = (_g_sup1 - self._open) / self._open
        return pd.Series(
            scaled_g_sup1, name=f"gsup_{self._window1}"
        )

    def d_supi(self) -> pd.Series:
        d_supi = self._check_fillna(self._d_supi, value=-1)
        return pd.Series(d_supi, name=f"d_supi")


class logMACD(IndicatorMixin):
    """Moving Average Convergence Divergence (MACD)

    Is a trend-following momentum indicator that shows the relationship between
    two moving averages of prices.

    https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd

    Args:
        close(pandas.Series): dataset 'Close' column.
        window_fast(int): n period short-term.
        window_slow(int): n period long-term.
        window_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
            self,
            close: pd.Series,
            opn: pd.Series,
            window_slow: int = 26,
            window_fast: int = 12,
            window_sign: int = 9,
            fillna: bool = False,
    ):
        self._close = close
        self._open = opn
        self._window_slow = window_slow
        self._window_fast = window_fast
        self._window_sign = window_sign
        self._fillna = fillna
        self._run()

    def _run(self):
        self._emafast = _ema(self._close, self._window_fast, self._fillna)
        self._emaslow = _ema(self._close, self._window_slow, self._fillna)

        self._macd = self._emafast - self._emaslow
        self._macd_signal = _ema(self._macd, self._window_sign, self._fillna)
        self._macd_diff = self._macd - self._macd_signal
        self._s_macd_signal = _ema(self._macd_diff, self._window_sign, self._fillna)

    def s_macd(self) -> pd.Series:
        macd_series = self._check_fillna(self._macd, value=0)
        return pd.Series(macd_series, name=f"MACD_{self._window_fast}_{self._window_slow}")

    def ns_macd_signal(self):
        ns_signal = self._check_fillna(self._macd_signal, value=0)
        return pd.Series(ns_signal, name=f"ns_macd_signal")

    def s_macd_diff(self) -> pd.Series:
        macd_diff_series = self._check_fillna(self._macd_diff, value=0)
        return pd.Series(macd_diff_series, name=f"MACD_diff_{self._window_fast}_{self._window_slow}")

    def s_macd_signal(self) -> pd.Series:
        macd_signal_series = self._check_fillna(self._s_macd_signal, value=0)
        return pd.Series(macd_signal_series, name=f"MACD_sign_{self._window_fast}_{self._window_slow}")


class logEMAIndicator(IndicatorMixin):
    """EMA - Exponential Moving Average

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, opn: pd.Series, window: int = 14, fillna: bool = False):
        self._close = close
        self._open = opn
        self._window = window
        self._fillna = fillna
        self.ns_ema = _ema(self._close, self._window, self._fillna)
        self.ema = _ema(self._open, self._window, self._fillna)

    def ema_indicator(self) -> pd.Series:
        """Exponential Moving Average (EMA)

        Returns:
            pandas.Series: New feature generated.
        """
        return pd.Series(self.ema, name=f"ema_{self._window}")

    def ns_ema_indicator(self) -> pd.Series:
        return pd.Series(self.ns_ema, name=f"ema_{self._window}")


class logSMAIndicator(IndicatorMixin):
    """SMA - Simple Moving Average

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, opn: pd.Series, window: int, fillna: bool = False):
        self._close = close
        self._open = opn
        self._window = window
        self._fillna = fillna

    def sma_indicator(self) -> pd.Series:
        """Simple Moving Average (SMA)

        Returns:
            pandas.Series: New feature generated.
        """
        sma_ = _sma(self._close, self._window, self._fillna)
        return pd.Series(sma_, name=f"sma_{self._window}")


class logTRIXIndicator(IndicatorMixin):
    """Trix (TRIX)

    Shows the percent rate of change of a triple exponentially smoothed moving
    average.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:trix

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, opn: pd.Series, window: int = 15, fillna: bool = False):
        self._opn = opn
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        ema1 = _ema(self._opn, self._window, self._fillna)
        self._dema = _ema(ema1, self._window, self._fillna)
        self.s_trix = _ema(self._dema, self._window, self._fillna)
        ema1 = _ema(self._close, self._window, self._fillna)
        self._dema = _ema(ema1, self._window, self._fillna)
        self._trix = _ema(self._dema, self._window, self._fillna)

    def dema(self):
        dema_series = self._check_fillna(self._dema, value=0)
        return pd.Series(dema_series, name=f"trix_{self._window}")

    def trix(self) -> pd.Series:
        trix_series = self._check_fillna(self._trix, value=0)
        return pd.Series(trix_series, name=f"trix_{self._window}")

    def strix(self) -> pd.Series:
        strix_series = self._check_fillna(self.s_trix, value=0)
        return pd.Series(strix_series, name=f"trix_{self._window}")


class logIchimokuIndicator(IndicatorMixin):
    """Ichimoku Kinkō Hyō (Ichimoku)

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ichimoku_cloud

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        window1(int): n1 low period.
        window2(int): n2 medium period.
        window3(int): n3 high period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
            self,
            high: pd.Series,
            low: pd.Series,
            opn: pd.Series,
            window1: int = 9,
            window2: int = 26,
            window3: int = 52,
            fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._open = opn
        self._window1 = window1
        self._window2 = window2
        self._window3 = window3
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods_n1 = 0 if self._fillna else self._window1
        min_periods_n2 = 0 if self._fillna else self._window2
        self._conv = 0.5 * (
                self._high.rolling(self._window1, min_periods=min_periods_n1).max()
                + self._low.rolling(self._window1, min_periods=min_periods_n1).min()
        )
        self._base = 0.5 * (
                self._high.rolling(self._window2, min_periods=min_periods_n2).max()
                + self._low.rolling(self._window2, min_periods=min_periods_n2).min()
        )

        self._spana = 0.5 * (self._conv + self._base)

        self._spanb = 0.5 * (
                self._high.rolling(self._window3, min_periods=0).max()
                + self._low.rolling(self._window3, min_periods=0).min()
        )

    def ichimoku_conversion_line(self) -> pd.Series:
        conversion = self._check_fillna(self._conv, value=-1)
        scaled_conversion = conversion
        return pd.Series(
            scaled_conversion, name=f"ichimoku_conv_{self._window1}_{self._window2}"
        )

    def ichimoku_base_line(self) -> pd.Series:
        base = self._check_fillna(self._base, value=-1)
        scaled_base = base
        return pd.Series(scaled_base, name=f"ichimoku_base_{self._window1}_{self._window2}")

    def lead_a(self):
        lead_a = self._check_fillna(self._spana, value=-1)
        return pd.Series(lead_a, name='lead_a')

    def lead_b(self):
        lead_b = self._check_fillna(self._spanb, value=-1)
        return pd.Series(lead_b, name='lead_b')

    def visual_ichimoku_a(self) -> pd.Series:
        spana = self._spana.shift(self._window2)
        spana = self._check_fillna(spana, value=-1)
        scaled_spana = spana
        return pd.Series(scaled_spana, name=f"ichimoku_a_{self._window1}_{self._window2}")

    def visual_ichimoku_b(self) -> pd.Series:
        """Senkou Span B (Leading Span B)

        Returns:
            pandas.Series: New feature generated.
        """
        spanb = self._spanb.shift(self._window2)
        spanb = self._check_fillna(spanb, value=-1)
        scaled_spanb = spanb
        return pd.Series(scaled_spanb, name=f"ichimoku_b_{self._window1}_{self._window2}")

    @staticmethod
    def ichimoku_a(_conv, _base, _open, window2, steps):
        """Senkou Span A (Leading Span A)

        Returns:
            pandas.Series: New feature generated.
        """
        spana = 0.5 * (_conv + _base)
        for i in range(steps):
            spana.loc[max(spana.index) + 1] = None
        spana = spana.shift(window2 - steps, fill_value=spana.mean())
        spana.fillna(method='ffill', inplace=True)
        scaled_spana = pd.DataFrame(
            [spana[i: i + steps] for i in range(len(_open))],
            columns=[f'minus_{i}' for i in range(steps - 1, -1, -1)])
        return scaled_spana

    @staticmethod
    def ichimoku_b(_high, _low, _open, window2, window3, steps):
        """Senkou Span B (Leading Span B)

        Returns:
            pandas.Series: New feature generated.
        """
        spanb = 0.5 * (_high.rolling(window3, min_periods=0).max() + _low.rolling(window3, min_periods=0).min())
        for i in range(steps):
            spanb.loc[max(spanb.index) + 1] = None
        spanb = spanb.shift(window2 - steps, fill_value=spanb.mean())
        spanb.fillna(method='ffill', inplace=True)
        scaled_spanb = pd.DataFrame(
            [spanb[i: i + steps] for i in range(len(_open))],
            columns=[f'minus_{i}' for i in range(steps - 1, -1, -1)])
        return scaled_spanb


class logPSARIndicator(IndicatorMixin):
    """Parabolic Stop and Reverse (Parabolic SAR)

    The Parabolic Stop and Reverse, more commonly known as the
    Parabolic SAR,is a trend-following indicator developed by
    J. Welles Wilder. The Parabolic SAR is displayed as a single
    parabolic line (or dots) underneath the price bars in an uptrend,
    and above the price bars in a downtrend.

    https://school.stockcharts.com/doku.php?id=technical_indicators:parabolic_sar

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        step(float): the Acceleration Factor used to compute the SAR.
        max_step(float): the maximum value allowed for the Acceleration Factor.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
            self,
            high: pd.Series,
            low: pd.Series,
            close: pd.Series,
            opn: pd.Series,
            step: float = 0.02,
            max_step: float = 0.20,
            fillna: bool = False,
    ):
        self._high = high.copy()
        self._low = low.copy()
        self._close = close
        self._open = opn
        self._step = step
        self._max_step = max_step
        self._fillna = fillna
        self._run()

    def _run(self):  # noqa
        up_trend = True
        acceleration_factor = self._step
        ep = self._high.iloc[0]

        self._ep = pd.Series(index=self._close.index, dtype='float64')
        self._af = pd.Series(index=self._close.index, dtype='float64')
        self._psar = self._close.copy()
        self._up_trend = pd.Series(index=self._close.index, dtype='float64')
        self._indicator = pd.Series(0, index=self._close.index)

        for i in range(2, len(self._close)):
            reversal = False
            max_high = self._high.iloc[i]
            min_low = self._low.iloc[i]

            if up_trend:
                self._psar.iloc[i] = self._psar.iloc[i - 1] + (acceleration_factor * (ep - self._psar.iloc[i - 1]))
                if min_low < self._psar.iloc[i]:
                    reversal = True
                    self._indicator[i] = -1
                    self._psar.iloc[i] = ep
                    ep = min_low
                    acceleration_factor = self._step
                else:
                    if max_high > ep:
                        ep = max_high
                        acceleration_factor = min(acceleration_factor + self._step, self._max_step)
                    top = min(self._low.iloc[i - 1], self._low.iloc[i - 2])
                    if top < self._psar.iloc[i]:
                        self._psar.iloc[i] = top
            else:
                self._psar.iloc[i] = self._psar.iloc[i - 1] - (
                        acceleration_factor * (self._psar.iloc[i - 1] - ep)
                )

                if max_high > self._psar.iloc[i]:
                    reversal = True
                    self._indicator[i] = 1
                    self._psar.iloc[i] = ep
                    ep = max_high
                    acceleration_factor = self._step
                else:
                    if min_low < ep:
                        ep = min_low
                        acceleration_factor = min(acceleration_factor + self._step, self._max_step)

                    bottom = max(self._high.iloc[i - 1], self._high.iloc[i - 2])
                    if bottom > self._psar.iloc[i]:
                        self._psar[i] = bottom

            self._ep[i] = ep
            self._af[i] = acceleration_factor
            up_trend = up_trend != reversal  # XOR
            self._up_trend[i] = up_trend

    def ep(self):
        ep = self._check_fillna(self._ep, value=-1)
        return pd.Series(ep, name='ep')

    def af(self):
        af = self._check_fillna(self._af, value=-1)
        return pd.Series(af, name='af')

    def ns_psar(self):
        ns = self._check_fillna(self._psar, value=-1)
        return pd.Series(ns, name='ns')

    def psar(self) -> pd.Series:
        psar_series = self._check_fillna(self._psar, value=-1)
        scaled_psar = psar_series
        return pd.Series(scaled_psar, name="psar")

    def up_trend(self):
        up = self._check_fillna(self._up_trend, value=-1)
        return pd.Series(up, name='up')

    def indicator(self):
        indi = self._check_fillna(self._indicator, value=-1)
        return pd.Series(indi, name='indicator')