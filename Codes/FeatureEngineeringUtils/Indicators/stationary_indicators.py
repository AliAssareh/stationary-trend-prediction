import numpy as np
import pandas as pd

from Codes.FeatureEngineeringUtils.Indicators.momentum_ta import (
    AwesomeOscillatorIndicator,
    KAMAIndicator,
    PercentageVolumeOscillator,
    ROCIndicator,
    RSIIndicator,
    StochasticOscillator,
    StochRSIIndicator,
    TSIIndicator,
    UltimateOscillator,
)
from Codes.FeatureEngineeringUtils.Indicators.trend_ta import (
    MACD,
    ADXIndicator,
    AroonIndicator,
    CCIIndicator,
    DPOIndicator,
    EMAIndicator,
    IchimokuIndicator,
    KSTIndicator,
    MassIndex,
    PSARIndicator,
    SMAIndicator,
    STCIndicator,
    TRIXIndicator,
    VortexIndicator,
    MinMaxIndicator,
)
from Codes.FeatureEngineeringUtils.Indicators.volatility_ta import (
    AverageTrueRange,
    BollingerBands,
    DonchianChannel,
    KeltnerChannel,
    UlcerIndex,
)
from Codes.FeatureEngineeringUtils.Indicators.volume_ta import (
    AccDistIndexIndicator,
    ChaikinMoneyFlowIndicator,
    EaseOfMovementIndicator,
    ForceIndexIndicator,
    MFIIndicator,
    NegativeVolumeIndexIndicator,
    OnBalanceVolumeIndicator,
    VolumePriceTrendIndicator,
    VolumeWeightedAveragePrice,
    Volume_ema
)


def add_volume_ta(df0: pd.DataFrame, opn: str, high: str, low: str, close: str, volume: str, fillna=False,
                  colprefix=""):
    df = {}
    log_vol = df0[volume].apply(np.log)
    df[f"{colprefix}volume_log_volume"] = log_vol

    # Accumulation Distribution Index
    df[f"{colprefix}volume_adi"] = AccDistIndexIndicator(
        high=df0[high], low=df0[low], close=df0[close], volume=log_vol, fillna=fillna
    ).acc_dist_index()

    # On Balance Volume
    df[f"{colprefix}volume_obv"] = OnBalanceVolumeIndicator(
        close=df0[close], volume=log_vol, fillna=fillna
    ).on_balance_volume()

    # Chaikin Money Flow
    df[f"{colprefix}volume_cmf"] = ChaikinMoneyFlowIndicator(
        high=df0[high], low=df0[low], close=df0[close], volume=log_vol, fillna=fillna
    ).chaikin_money_flow()

    # Force Index
    indicator_fi = ForceIndexIndicator(close=df0[close], volume=log_vol, w1=13, w2=7, fillna=fillna)
    df[f"{colprefix}volume_fi_13"] = indicator_fi.force_index()
    df[f"{colprefix}volume_fi_7"] = indicator_fi.force_index2()

    # Money Flow Indicator
    indicator_mfi = MFIIndicator(high=df0[high], low=df0[low], close=df0[close], volume=log_vol, w1=14, w2=10,
                                 fillna=fillna)
    df[f"{colprefix}volume_raw_mfi"] = indicator_mfi.raw()
    df[f"{colprefix}volume_mfi_14"] = indicator_mfi.money_flow_index_14()
    df[f"{colprefix}volume_mfi_10"] = indicator_mfi.money_flow_index_10()

    # Ease of Movement
    indicator_eom = EaseOfMovementIndicator(high=df0[high], low=df0[low], volume=log_vol, window=14, fillna=fillna)
    df[f"{colprefix}volume_em"] = indicator_eom.ease_of_movement()
    df[f"{colprefix}volume_sma_em"] = indicator_eom.sma_ease_of_movement()

    # Volume Price Trend
    df[f"{colprefix}volume_vpt"] = VolumePriceTrendIndicator(
        close=df0[close], volume=log_vol, fillna=fillna
    ).volume_price_trend()

    # Negative Volume Index
    df[f"{colprefix}volume_nvi"] = NegativeVolumeIndexIndicator(
        close=df0[close], volume=log_vol, fillna=fillna
    ).negative_volume_index()

    # Volume Weighted Average Price
    indicator_vwap = VolumeWeightedAveragePrice(
        opn=df0[opn],
        high=df0[high],
        low=df0[low],
        close=df0[close],
        volume=log_vol,
        window=14,
        fillna=fillna,
    )
    df[f"{colprefix}volume_tpv"] = indicator_vwap.tpv()
    df[f"{colprefix}volume_vwap"] = indicator_vwap.volume_weighted_average_price()

    # ema
    indicator_vema = Volume_ema(volume=log_vol, window1=12, window2=26)
    df[f"{colprefix}volume_ema_12"] = indicator_vema.vema1()
    df[f"{colprefix}volume_ema_26"] = indicator_vema.vema2()

    new_df = pd.concat([df0, pd.DataFrame(df)], axis=1)

    return new_df


def add_volatility_ta(df0, opn: str, high: str, low: str, close: str, up_first='None', fillna=False, colprefix=""):
    df = {}
    # Average True Range
    if up_first == 'None':
        df[f"{colprefix}volatility_atr"] = AverageTrueRange(
            close=df0[close], high=df0[high], low=df0[low], window=10, fillna=fillna
        ).average_true_range()
    else:
        indicator_atr = AverageTrueRange(close=df0[close], high=df0[high], low=df0[low], up_first=df0[up_first],
                                         window=10)
        df[f"{colprefix}volatility_atr"] = indicator_atr.average_true_range()
        df[f"{colprefix}volatility_p_atr"] = indicator_atr.p_average_true_range()

    # Bollinger Bands
    indicator_bb = BollingerBands(
        close=df0[close], opn=df0[opn], window=20, window_dev=2, fillna=fillna
    )
    df[f"{colprefix}volatility_bbm"] = indicator_bb.bollinger_mavg()
    df[f"{colprefix}volatility_bbh"] = indicator_bb.bollinger_hband()
    df[f"{colprefix}volatility_bbl"] = indicator_bb.bollinger_lband()
    df[f"{colprefix}volatility_bbw"] = indicator_bb.bollinger_wband()
    df[f"{colprefix}volatility_bbp"] = indicator_bb.bollinger_pband()

    # Keltner Channel
    if up_first == 'None':
        indicator_kc = KeltnerChannel(close=df0[close], high=df0[high], low=df0[low], opn=df0[opn], window=10)
        df[f"{colprefix}volatility_kch"] = indicator_kc.keltner_channel_hband()
        df[f"{colprefix}volatility_kcl"] = indicator_kc.keltner_channel_lband()
        df[f"{colprefix}volatility_kcw"] = indicator_kc.keltner_channel_wband()
        df[f"{colprefix}volatility_kcp"] = indicator_kc.keltner_channel_pband()
    else:
        indicator_kc = KeltnerChannel(close=df0[close], high=df0[high], low=df0[low], opn=df0[opn],
                                      up_first=df0[up_first], window=10)
        df[f"{colprefix}volatility_kch"] = indicator_kc.keltner_channel_p_hband()
        df[f"{colprefix}volatility_kcl"] = indicator_kc.keltner_channel_p_lband()
        df[f"{colprefix}volatility_kcw"] = indicator_kc.keltner_channel_wband()
        df[f"{colprefix}volatility_kcp"] = indicator_kc.keltner_channel_pband()

    # Donchian Channel
    indicator_dc = DonchianChannel(
        high=df0[high], low=df0[low], close=df0[close], opn=df0[opn], window=20, offset=0, fillna=fillna
    )
    df[f"{colprefix}volatility_dch"] = indicator_dc.donchian_channel_hband()
    df[f"{colprefix}volatility_dcl"] = indicator_dc.donchian_channel_lband()
    df[f"{colprefix}volatility_dcm"] = indicator_dc.donchian_channel_mband()
    df[f"{colprefix}volatility_dcw"] = indicator_dc.donchian_channel_wband()
    df[f"{colprefix}volatility_dcp"] = indicator_dc.donchian_channel_pband()

    # Ulcer Index
    ui = UlcerIndex(close=df0[close], window=14, fillna=fillna)
    df[f"{colprefix}volatility_spdd"] = ui.spdd()
    df[f"{colprefix}volatility_sndd"] = ui.sndd()
    df[f"{colprefix}volatility_ui"] = ui.ulcer_index()
    df[f"{colprefix}volatility_nui"] = ui.n_ulcer_index()

    new_df = pd.concat([df0, pd.DataFrame(df)], axis=1)

    return new_df


def add_trend_ta(df0, opn: str, high: str, low: str, close: str, fillna=False, colprefix: str = ""):
    df = {}
    # SMAs
    df[f"{colprefix}trend_sma_12"] = SMAIndicator(close=df0[close], opn=df0[opn], window=12).sma_indicator()
    df[f"{colprefix}trend_sma_26"] = SMAIndicator(close=df0[close], opn=df0[opn], window=26).sma_indicator()
    df[f"{colprefix}trend_sma_50"] = SMAIndicator(close=df0[close], opn=df0[opn], window=50).sma_indicator()

    # EMAs
    ema_10 = EMAIndicator(close=df0[close], opn=df0[opn], window=10, fillna=fillna)
    df[f"{colprefix}trend_ema_10"] = ema_10.ema_indicator()
    df[f"{colprefix}trend_ns_ema_10"] = ema_10.ns_ema_indicator()

    ema_12 = EMAIndicator(close=df0[close], opn=df0[opn], window=12, fillna=fillna)
    df[f"{colprefix}trend_ema_12"] = ema_12.ema_indicator()
    df[f"{colprefix}trend_ns_ema_12"] = ema_12.ns_ema_indicator()

    ema_15 = EMAIndicator(close=df0[close], opn=df0[opn], window=15, fillna=fillna)
    df[f"{colprefix}trend_ema_15"] = ema_15.ema_indicator()
    df[f"{colprefix}trend_ns_ema_15"] = ema_15.ns_ema_indicator()

    ema_20 = EMAIndicator(close=df0[close], opn=df0[opn], window=20, fillna=fillna)
    df[f"{colprefix}trend_ema_20"] = ema_20.ema_indicator()
    df[f"{colprefix}trend_ns_ema_20"] = ema_20.ns_ema_indicator()

    ema_23 = EMAIndicator(close=df0[close], opn=df0[opn], window=23, fillna=fillna)
    df[f"{colprefix}trend_ema_23"] = ema_23.ema_indicator()
    df[f"{colprefix}trend_ns_ema_23"] = ema_23.ns_ema_indicator()

    ema_26 = EMAIndicator(close=df0[close], opn=df0[opn], window=26, fillna=fillna)
    df[f"{colprefix}trend_ema_26"] = ema_26.ema_indicator()
    df[f"{colprefix}trend_ns_ema_26"] = ema_26.ns_ema_indicator()

    ema_50 = EMAIndicator(close=df0[close], opn=df0[opn], window=50, fillna=fillna)
    df[f"{colprefix}trend_ema_50"] = ema_50.ema_indicator()
    df[f"{colprefix}trend_ns_ema_50"] = ema_50.ns_ema_indicator()

    # MACD
    indicator_macd = MACD(close=df0[close], opn=df0[opn], window_slow=26, window_fast=12, window_sign=9, fillna=fillna)
    df[f"{colprefix}trend_macd"] = indicator_macd.s_macd()
    df[f"{colprefix}trend_ns_macd_signal"] = indicator_macd.ns_macd_signal()
    df[f"{colprefix}trend_macd_signal"] = indicator_macd.s_macd_signal()
    df[f"{colprefix}trend_macd_diff"] = indicator_macd.s_macd_diff()

    # Average Directional Movement Index (ADX)
    indicator_adx = ADXIndicator(
        high=df0[high], low=df0[low], close=df0[close], window=14, fillna=fillna
    )
    df[f"{colprefix}trend_p_di"] = indicator_adx.pdi()
    df[f"{colprefix}trend_spdi"] = indicator_adx.spdi()
    df[f"{colprefix}trend_n_di"] = indicator_adx.ndi()
    df[f"{colprefix}trend_sndi"] = indicator_adx.sndi()
    df[f"{colprefix}trend_atr_14"] = indicator_adx.atr()
    df[f"{colprefix}trend_adx"] = indicator_adx.adx()

    # Vortex Indicator
    indicator_vortex = VortexIndicator(
        high=df0[high], low=df0[low], close=df0[close], window=14, fillna=fillna
    )
    df[f"{colprefix}trend_p_vm"] = indicator_vortex.p_vm()
    df[f"{colprefix}trend_n_vm"] = indicator_vortex.n_vm()
    df[f"{colprefix}trend_vortex_ind_pos"] = indicator_vortex.vortex_indicator_pos()
    df[f"{colprefix}trend_vortex_ind_neg"] = indicator_vortex.vortex_indicator_neg()

    # TRIX Indicator
    indicator_trix = TRIXIndicator(close=df0[close], opn=df0[opn], window=15, fillna=fillna)
    df[f"{colprefix}trend_dema"] = indicator_trix.dema()
    df[f"{colprefix}trend_trix"] = indicator_trix.trix()
    df[f"{colprefix}trend_strix"] = indicator_trix.strix()

    # Mass Index
    indicator_mass_index = MassIndex(high=df0[high], low=df0[low], window_fast=9, window_slow=25, fillna=fillna)
    df[f"{colprefix}trend_tr_ema"] = indicator_mass_index.tr_ema()
    df[f"{colprefix}trend_tr_dema"] = indicator_mass_index.tr_dema()
    df[f"{colprefix}trend_mass"] = indicator_mass_index.mass()
    df[f"{colprefix}trend_mass_index"] = indicator_mass_index.mass_index()

    # CCI Indicator
    indicator_cci = CCIIndicator(high=df0[high], low=df0[low], close=df0[close], window=20, constant=0.015,
                                 fillna=fillna)
    df[f"{colprefix}trend_typical"] = indicator_cci.tp()
    df[f"{colprefix}trend_cci"] = indicator_cci.cci()

    # DPO Indicator
    df[f"{colprefix}trend_dpo"] = DPOIndicator(close=df0[close], window=20, fillna=fillna).dpo()

    # KST Indicator
    indicator_kst = KSTIndicator(
        close=df0[close],
        roc1=10,
        roc2=15,
        roc3=20,
        roc4=30,
        window1=10,
        window2=10,
        window3=10,
        window4=15,
        nsig=9,
        fillna=fillna,
    )
    df[f"{colprefix}trend_roc10"] = indicator_kst.roc10()
    df[f"{colprefix}trend_roc15"] = indicator_kst.roc15()
    df[f"{colprefix}trend_roc20"] = indicator_kst.roc20()
    df[f"{colprefix}trend_roc30"] = indicator_kst.roc30()
    df[f"{colprefix}trend_kst"] = indicator_kst.kst()
    df[f"{colprefix}trend_kst_sig"] = indicator_kst.kst_sig()

    # Ichimoku Indicator
    indicator_ichi = IchimokuIndicator(high=df0[high], low=df0[low], opn=df0[opn], window1=9, window2=26, window3=52)
    df[f"{colprefix}trend_ichimoku_conv"] = indicator_ichi.ichimoku_conversion_line()
    df[f"{colprefix}trend_ichimoku_base"] = indicator_ichi.ichimoku_base_line()
    df[f"{colprefix}trend_lead_a"] = indicator_ichi.lead_a()
    df[f"{colprefix}trend_lead_b"] = indicator_ichi.lead_b()
    df[f"{colprefix}trend_ichimoku_a"] = indicator_ichi.visual_ichimoku_a()
    df[f"{colprefix}trend_ichimoku_b"] = indicator_ichi.visual_ichimoku_b()

    # Aroon Indicator
    indicator_aroon = AroonIndicator(close=df0[close], window=25, fillna=fillna)
    df[f"{colprefix}trend_aroon_up"] = indicator_aroon.aroon_up()
    df[f"{colprefix}trend_aroon_down"] = indicator_aroon.aroon_down()
    indicator_aroon = AroonIndicator(close=df0[high], window=25, fillna=fillna)
    df[f"{colprefix}trend_high_aroon_up"] = indicator_aroon.aroon_up()
    indicator_aroon = AroonIndicator(close=df0[low], window=25, fillna=fillna)
    df[f"{colprefix}trend_low_aroon_down"] = indicator_aroon.aroon_down()

    # PSAR Indicator
    indicator_psar = PSARIndicator(high=df0[high], low=df0[low], close=df0[close], opn=df0[opn], step=0.02,
                                   max_step=0.20)
    df[f"{colprefix}trend_psar_ep"] = indicator_psar.ep()
    df[f"{colprefix}trend_psar_af"] = indicator_psar.af()
    df[f"{colprefix}trend_ns_psar"] = indicator_psar.ns_psar()
    df[f'{colprefix}trend_psar'] = indicator_psar.psar()
    df[f"{colprefix}trend_psar_up_trend"] = indicator_psar.up_trend()
    df[f'{colprefix}trend_psar_indicator'] = indicator_psar.indicator()

    # Schaff Trend Cycle (STC)
    indicator_stc = STCIndicator(close=df0[close], window_slow=50, window_fast=23, cycle=10, smooth1=3, smooth2=3)
    df[f"{colprefix}trend_ns_macd"] = indicator_stc.ns_macd()
    df[f"{colprefix}trend_stoch_k"] = indicator_stc.stoch_k()
    df[f"{colprefix}trend_stoch_d"] = indicator_stc.stoch_d()
    df[f"{colprefix}trend_stoch_kd"] = indicator_stc.stoch_kd()
    df[f"{colprefix}trend_stc"] = indicator_stc.stc()

    # MinMax cycle
    indicator_minmax = MinMaxIndicator(high=df0[high], low=df0[low], opn=df0[close],
                                       patr=df0[f"{colprefix}volatility_p_atr"],
                                       rsi=df0[f"{colprefix}momentum_rsi"])
    df[f"{colprefix}trend_minmax_res1"] = indicator_minmax.res1()
    df[f"{colprefix}trend_minmax_res2"] = indicator_minmax.res2()
    df[f"{colprefix}trend_minmax_res3"] = indicator_minmax.res3()
    df[f"{colprefix}trend_minmax_a_res1"] = indicator_minmax.a_res1()
    df[f"{colprefix}trend_minmax_a_res2"] = indicator_minmax.a_res2()
    df[f"{colprefix}trend_minmax_a_res3"] = indicator_minmax.a_res3()
    df[f"{colprefix}trend_minmax_r_res1"] = indicator_minmax.r_res1()
    df[f"{colprefix}trend_minmax_r_res2"] = indicator_minmax.r_res2()
    df[f"{colprefix}trend_minmax_r_res3"] = indicator_minmax.r_res3()
    df[f"{colprefix}trend_minmax_d_res1"] = indicator_minmax.d_res1()
    df[f"{colprefix}trend_minmax_d_res2"] = indicator_minmax.d_res2()
    df[f"{colprefix}trend_minmax_d_res3"] = indicator_minmax.d_res3()
    df[f"{colprefix}trend_minmax_g_res1"] = indicator_minmax.g_res1()

    df[f"{colprefix}trend_minmax_d_resi"] = indicator_minmax.d_resi()
    df[f"{colprefix}trend_minmax_g_resi"] = indicator_minmax.g_resi()
    df[f"{colprefix}trend_minmax_d_supi"] = indicator_minmax.d_supi()


    df[f"{colprefix}trend_minmax_sup1"] = indicator_minmax.sup1()
    df[f"{colprefix}trend_minmax_sup2"] = indicator_minmax.sup2()
    df[f"{colprefix}trend_minmax_sup3"] = indicator_minmax.sup3()
    df[f"{colprefix}trend_minmax_a_sup1"] = indicator_minmax.a_sup1()
    df[f"{colprefix}trend_minmax_a_sup2"] = indicator_minmax.a_sup2()
    df[f"{colprefix}trend_minmax_a_sup3"] = indicator_minmax.a_sup3()
    df[f"{colprefix}trend_minmax_r_sup1"] = indicator_minmax.r_sup1()
    df[f"{colprefix}trend_minmax_r_sup2"] = indicator_minmax.r_sup2()
    df[f"{colprefix}trend_minmax_r_sup3"] = indicator_minmax.r_sup3()
    df[f"{colprefix}trend_minmax_d_sup1"] = indicator_minmax.d_sup1()
    df[f"{colprefix}trend_minmax_d_sup2"] = indicator_minmax.d_sup2()
    df[f"{colprefix}trend_minmax_d_sup3"] = indicator_minmax.d_sup3()
    df[f"{colprefix}trend_minmax_g_sup1"] = indicator_minmax.g_sup1()

    new_df = pd.concat([df0, pd.DataFrame(df)], axis=1)

    return new_df


def add_momentum_ta(df0, opn: str, high: str, low: str, close: str, volume='None', fillna=False, colprefix: str = ""):
    df = {}
    # Relative Strength Index (RSI)
    indicator_rsi = RSIIndicator(close=df0[close], window=14, fillna=fillna)
    df[f"{colprefix}momentum_ema_up"] = indicator_rsi.emaup()
    df[f"{colprefix}momentum_ema_dn"] = indicator_rsi.emadn()
    df[f"{colprefix}momentum_rsi"] = indicator_rsi.rsi()

    # Stoch RSI (StochRSI)
    indicator_srsi = StochRSIIndicator(close=df0[close], window=14, smooth1=3, smooth2=3, fillna=fillna)
    df[f"{colprefix}momentum_stoch_rsi"] = indicator_srsi.stochrsi()

    # TSI Indicator
    indicator_tsi = TSIIndicator(close=df0[close], window_slow=25, window_fast=13, fillna=fillna)
    df[f"{colprefix}momentum_pc_ema"] = indicator_tsi.pc_ema()
    df[f"{colprefix}momentum_pc_dma"] = indicator_tsi.pc_dema()
    df[f"{colprefix}momentum_apc_ema"] = indicator_tsi.apc_ema()
    df[f"{colprefix}momentum_apc_dma"] = indicator_tsi.apc_dema()
    df[f"{colprefix}momentum_tsi"] = indicator_tsi.tsi()

    # Ultimate Oscillator
    uo = UltimateOscillator(high=df0[high], low=df0[low], close=df0[close], window1=7, window2=14, window3=28, weight1=4.0,
                            weight2=2.0, weight3=1.0)
    df[f"{colprefix}momentum_bp"] = uo.bp()
    df[f"{colprefix}momentum_uo"] = uo.ultimate_oscillator()

    # Stoch Indicator
    indicator_so = StochasticOscillator(high=df0[high], low=df0[low], close=df0[close], window=14, smooth_window=3)
    df[f"{colprefix}momentum_stoch"] = indicator_so.stoch()
    df[f"{colprefix}momentum_stoch_signal"] = indicator_so.stoch_signal()

    # Awesome Oscillator
    indicator_ao = AwesomeOscillatorIndicator(high=df0[high], low=df0[low], opn=df0[opn], window1=5, window2=34)
    df[f"{colprefix}momentum_med"] = indicator_ao.median_price()
    df[f"{colprefix}momentum_ao"] = indicator_ao.awesome_oscillator()

    # KAMA
    indicator_kama = KAMAIndicator(close=df0[close], opn=df0[opn], window=10, pow1=2, pow2=30)
    df[f"{colprefix}momentum_c2c"] = indicator_kama.c2c()
    df[f"{colprefix}momentum_ns_kama"] = indicator_kama.kama()
    df[f"{colprefix}momentum_kama"] = indicator_kama.skama()

    # Rate Of Change
    df[f"{colprefix}momentum_roc"] = ROCIndicator(close=df0[close], window=12, fillna=fillna).roc()

    # Percentage Volume Oscillator
    if volume != 'None':
        indicator_pvo = PercentageVolumeOscillator(volume=df0[volume].apply(np.log), window_slow=26, window_fast=12,
                                                   window_sign=9)
        df[f"{colprefix}momentum_pvo"] = indicator_pvo.pvo()
        df[f"{colprefix}momentum_pvo_signal"] = indicator_pvo.pvo_signal()

    new_df = pd.concat([df0, pd.DataFrame(df)], axis=1)

    return new_df.copy()


def add_all_stationary_ta_features(df, opn: str, high: str, low: str, close: str, volume='None', up_first='None',
                                   fillna=False, colprefix: str = ""):
    """Add all technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        opn (str): Name of 'open' column.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        up_first (str|None): Name of 'up_first' column
        fillna (bool): if True, fill nan values.
        colprefix (str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    if volume != 'None':
        df = add_volume_ta(df0=df, opn=opn, high=high, low=low, close=close, volume=volume, fillna=fillna,
                           colprefix=colprefix)
    df = add_volatility_ta(df0=df, opn=opn, high=high, low=low, close=close, up_first=up_first, fillna=fillna,
                           colprefix=colprefix)
    df = add_momentum_ta(df0=df, opn=opn, high=high, low=low, close=close, volume=volume, fillna=fillna,
                         colprefix=colprefix)
    df = add_trend_ta(df0=df, opn=opn, high=high, low=low, close=close, fillna=fillna, colprefix=colprefix)
    return df
