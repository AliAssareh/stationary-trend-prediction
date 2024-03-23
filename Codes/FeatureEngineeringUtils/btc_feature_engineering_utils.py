from Codes.FeatureEngineeringUtils.Indicators.logstationary_indicators import add_all_log_stationary_ta_features
from Codes.FeatureEngineeringUtils.Indicators.stationary_indicators import add_all_stationary_ta_features
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from ta import add_all_ta_features
from pickle import dump, load
from enum import Enum
import numpy as np
import pandas as pd
import shutil
import os


class Target(Enum):
    target1 = 1
    target2 = 2
    target3 = 3


class TimeStamps(Enum):
    start = '2017-08-17 00:00:00'
    end1 = '2022-01-01 00:00:01'
    crop = '2021-12-31 04:00:00'
    end2 = '2024-01-03 00:00:01'


class StationaryFeatures:

    @staticmethod
    def generate_stationary_features_list(colprefix):
        features = ['open', 'high', 'low', 'close', 'ho_percent', 'co_percent', 'oc_percent', 'lo_percent',
                    'tr_percent', 'volume_log_volume', 'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi_7',
                    'volume_fi_13', 'volume_mfi_10', 'volume_mfi_14', 'volume_em', 'volume_sma_em', 'volume_vpt',
                    'volume_nvi', 'volume_tpv', 'volume_vwap', 'trend_macd', 'trend_macd_signal', 'trend_macd_diff',
                    'trend_sma_12', 'trend_sma_26', 'trend_sma_50', 'trend_ema_12', 'trend_ema_15', 'trend_ema_20',
                    'trend_ema_23', 'trend_ema_26', 'trend_ema_50', 'trend_p_di', 'trend_n_di', 'trend_adx',
                    'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_strix', 'trend_dpo', 'trend_roc10',
                    'trend_roc15', 'trend_roc20', 'trend_roc30', 'trend_kst', 'trend_kst_sig', 'trend_ichimoku_conv',
                    'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b', 'trend_high_aroon_up',
                    'trend_aroon_up', 'trend_aroon_down', 'trend_low_aroon_down', 'trend_psar', 'trend_psar_indicator',
                    'trend_stoch_k', 'trend_stoch_d', 'trend_stoch_kd', 'trend_stc', 'trend_minmax_res1',
                    'trend_minmax_res2', 'trend_minmax_res3', 'trend_minmax_a_res1', 'trend_minmax_a_res2',
                    'trend_minmax_a_res3', "trend_minmax_r_res1", "trend_minmax_r_res2", "trend_minmax_r_res3",
                    "trend_minmax_d_res1", "trend_minmax_d_res2", "trend_minmax_d_res3", "trend_minmax_g_res1",
                    'trend_minmax_sup1', 'trend_minmax_sup2', 'trend_minmax_sup3', 'trend_minmax_a_sup1',
                    'trend_minmax_a_sup2', 'trend_minmax_a_sup3', "trend_minmax_r_sup1", "trend_minmax_r_sup2",
                    "trend_minmax_r_sup3", "trend_minmax_d_sup1", "trend_minmax_d_sup2", "trend_minmax_d_sup3",
                    "trend_minmax_g_sup1", "trend_minmax_d_resi", "trend_minmax_g_resi", "trend_minmax_d_supi",
                    'trend_cci', 'volatility_p_atr', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl',
                    'trend_mass_index', 'volatility_bbp', 'volatility_bbw', 'volatility_kch', 'volatility_kcl',
                    'volatility_kcp', 'volatility_kcw', 'volatility_dcm', 'volatility_dch', 'volatility_dcl',
                    'volatility_dcp', 'volatility_dcw', 'volatility_ui', 'volatility_nui', 'momentum_rsi',
                    'momentum_stoch_rsi', 'momentum_tsi', 'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal',
                    'momentum_ao', 'momentum_kama', 'momentum_roc', 'momentum_pvo', 'momentum_pvo_signal'
                    ]
        return [colprefix + feature for feature in features]

    @staticmethod
    def generate_excluded_stationary_features_list(colprefix):
        features = ['open', 'high', 'low', 'close', 'volume_log_volume', 'volume_adi', 'volume_obv', 'volume_cmf',
                    'volume_fi_7', 'volume_fi_13', 'volume_mfi_10', 'volume_mfi_14', 'volume_em', 'volume_sma_em',
                    'volume_vpt', 'volume_nvi', 'volume_tpv', 'volume_vwap', 'trend_macd', 'trend_macd_signal',
                    'trend_macd_diff', 'trend_sma_12', 'trend_sma_26', 'trend_sma_50', 'trend_ema_12', 'trend_ema_15',
                    'trend_ema_20', 'trend_ema_23', 'trend_ema_26', 'trend_ema_50', 'trend_p_di', 'trend_n_di',
                    'trend_adx', 'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_strix', 'trend_dpo',
                    'trend_roc10', 'trend_roc15', 'trend_roc20', 'trend_roc30', 'trend_kst', 'trend_kst_sig',
                    'trend_ichimoku_conv', 'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b',
                    'trend_high_aroon_up', 'trend_aroon_up', 'trend_aroon_down', 'trend_low_aroon_down', 'trend_psar',
                    'trend_psar_indicator', 'trend_stoch_k', 'trend_stoch_d', 'trend_stoch_kd', 'trend_stc',
                    'trend_cci', 'volatility_p_atr', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl',
                    'trend_mass_index', 'volatility_bbp', 'volatility_bbw', 'volatility_kch', 'volatility_kcl',
                    'volatility_kcp', 'volatility_kcw', 'volatility_dcm', 'volatility_dch', 'volatility_dcl',
                    'volatility_dcp', 'volatility_dcw', 'volatility_ui', 'volatility_nui', 'momentum_rsi',
                    'momentum_stoch_rsi', 'momentum_tsi', 'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal',
                    'momentum_ao', 'momentum_kama', 'momentum_roc', 'momentum_pvo', 'momentum_pvo_signal'
                    ]
        return [colprefix + feature for feature in features]


def import_raw_data(address, index_column_name='datetime', time_format="%Y-%m-%d %H:%M:%S", from_unix=False):
    df = pd.read_csv(address)
    if from_unix:
        df['datetime'] = pd.to_datetime(df[index_column_name], unit='s')
    else:
        df['datetime'] = pd.to_datetime(df[index_column_name], format=time_format)
    df.sort_values('datetime', inplace=True)
    df.set_index('datetime', inplace=True)
    if index_column_name != 'datetime':
        df.drop(columns=[index_column_name], inplace=True)
    if 'up_first' in df.columns:
        if 'volume' in df.columns:
            df = df.loc[:, ['open', 'high', 'low', 'close', 'volume', 'up_first']]
        else:
            df = df.loc[:, ['open', 'high', 'low', 'close', 'up_first']]
    else:
        if 'volume' in df.columns:
            df = df.loc[:, ['open', 'high', 'low', 'close', 'volume']]
        else:
            df = df.loc[:, ['open', 'high', 'low', 'close']]
    return df


def import_processed_data(address, index_column_name='datetime', time_format="%Y-%m-%d %H:%M:%S"):
    df = pd.read_csv(address, low_memory=False)
    df['datetime'] = pd.to_datetime(df[index_column_name], format=time_format)
    if index_column_name != 'datetime':
        df.drop(columns=[index_column_name], inplace=True)
    df.sort_values('datetime', inplace=True)
    df.set_index('datetime', inplace=True)
    return df


def preprocess_m1_df(address, index_column_name='datetime', time_format="ISO8601"):
    raw_m1_df = import_raw_data(address, index_column_name, time_format)
    if os.path.isfile('Data//PreprocessedData//BTCUSDT_1.csv'):
        preprocessed_m1_df = import_processed_data('Data//PreprocessedData//BTCUSDT_1.csv',
                                                   index_column_name='datetime')
        last_stamp = preprocessed_m1_df.tail(1).index.to_pydatetime()[0]
        last_stamp = last_stamp - timedelta(hours=last_stamp.hour, minutes=last_stamp.minute, seconds=last_stamp.second)
        rest_of_raw_m1 = raw_m1_df.loc[last_stamp:]
        if len(rest_of_raw_m1) > 1:
            print(f'Updating the last {len(rest_of_raw_m1)} candles')
            m1_linear_interpolator = Interpolator(rest_of_raw_m1, time_frame_in_minutes=1,
                                                  first_day=datetime.strftime(last_stamp, '%Y-%m-%d %H:%M:%S'))
            m1_df = m1_linear_interpolator.return_complete_dataframe()
            preprocessed_m1_df.drop(preprocessed_m1_df.loc[last_stamp:].index, inplace=True)
            m1_df = pd.concat([preprocessed_m1_df, m1_df], axis=0)
            m1_df.index.name = 'datetime'
            m1_df.sort_index(inplace=True)
            m1_df.to_csv('Data//PreprocessedData//BTCUSDT_1.csv')
        else:
            print('BTCUSDT_1 is already up to date!')
    else:
        print('BTCUSDT_1.csv not found at PreprocessedData, preprocessing the whole series might take a while!')
        m1_linear_interpolator = Interpolator(raw_m1_df, time_frame_in_minutes=1)
        m1_df = m1_linear_interpolator.return_complete_dataframe()
        m1_df.to_csv('Data//PreprocessedData//BTCUSDT_1.csv')


class BtcPreprocessor:
    def __init__(self, timeframe, use_stationary_ta=True, method='default',
                 first_day=TimeStamps.start.value, colprefix='BTCUSDT_', time_format="ISO8601"):
        self.df = import_raw_data(f"Data//Raw//BTCUSDT_{timeframe}.csv", 'datetime', time_format)
        self.timeframe = timeframe
        self.use_stationary_ta = use_stationary_ta
        if method not in ['default', 'log']:
            raise Exception('This method of dealing with non-stationarity is not implemented yet!')
        self.method = method
        self.colprefix = colprefix
        self.first_day = first_day
        self.features_df = None

    def run(self):
        processed_df = self.interpolate_the_raw_data_and_add_up_first()

        if self.use_stationary_ta and (self.method == 'default'):
            alt_candles_df = self._create_alternative_candles(processed_df)
            indicators_df = self._generate_def_indicators(processed_df)
            self.features_df = self._create_features_df(alt_candles_df, indicators_df)
        elif self.use_stationary_ta and (self.method == 'log'):
            logalt_candles_df = self._create_log_alternative_candles(processed_df)
            logindicators_df = self._generate_log_indicators(processed_df)
            self.features_df = self._create_features_df(logalt_candles_df, logindicators_df)
        else:
            alt_candles_df = self._create_alternative_candles(processed_df)
            indicators_df = self._generate_indicators(processed_df)
            self.features_df = self._create_features_df(alt_candles_df, indicators_df)
        self.save_results()

    def interpolate_the_raw_data_and_add_up_first(self):
        linear_interpolator = Interpolator(self.df, self.timeframe, first_day=self.first_day)
        interpolated_df = linear_interpolator.return_complete_dataframe()
        up_first_detector = UpFirstDetector(interpolated_df, self.timeframe, first_day=self.first_day)
        return up_first_detector.return_new_df()

    def _create_alternative_candles(self, processed_df):
        temp_df = processed_df.copy()
        temp_df['previous_close'] = temp_df.close.shift(+1)
        temp_df['previous_open'] = temp_df.open.shift(+1)

        first_open = temp_df.open.iloc[0].copy()
        temp_df.iloc[0, temp_df.columns.get_loc('previous_close')] = first_open
        temp_df.iloc[0, temp_df.columns.get_loc('previous_open')] = first_open

        temp_df[self.colprefix + 'ho_percent'] = (temp_df.high - temp_df.open) / temp_df.open
        temp_df[self.colprefix + 'co_percent'] = (temp_df.close - temp_df.open) / temp_df.open
        temp_df[self.colprefix + 'oc_percent'] = (temp_df.open - temp_df.previous_close) / temp_df.previous_close
        temp_df[self.colprefix + 'lo_percent'] = (temp_df.low - temp_df.open) / temp_df.open
        temp_df[self.colprefix + 'true_range'] = (temp_df.high - temp_df.low)
        denom = temp_df.high.where(temp_df.up_first, temp_df.low)
        temp_df[self.colprefix + 'tr_percent'] = temp_df[self.colprefix + 'true_range'] / denom

        temp_df.rename({'open': self.colprefix + 'open', 'high': self.colprefix + 'high',
                        'low': self.colprefix + 'low', 'close': self.colprefix + 'close',
                        'volume': self.colprefix + 'volume', 'up_first': self.colprefix + 'up_first'},
                       axis=1, inplace=True)

        temp_df.drop(columns=['previous_open', 'previous_close'], inplace=True)

        return temp_df

    def _create_log_alternative_candles(self, processed_df):
        temp_df = processed_df.copy()
        log_open = temp_df.open.apply(np.log)
        log_open = log_open - log_open.shift(1).fillna(method='backfill')
        log_high = temp_df.high.apply(np.log)
        log_high = log_high - log_high.shift(1).fillna(method='backfill')
        log_low = temp_df.low.apply(np.log)
        log_low = log_low - log_low.shift(1).fillna(method='backfill')
        log_close = temp_df.close.apply(np.log)
        log_close = log_close - log_close.shift(1).fillna(method='backfill')
        temp_df['previous_close'] = temp_df.close.shift(+1)
        temp_df['previous_open'] = temp_df.open.shift(+1)

        temp_df[self.colprefix + 'ho_percent'] = log_high
        temp_df[self.colprefix + 'co_percent'] = log_close
        temp_df[self.colprefix + 'oc_percent'] = log_open
        temp_df[self.colprefix + 'lo_percent'] = log_low
        temp_df[self.colprefix + 'true_range'] = (temp_df.high - temp_df.low)
        denom = temp_df.high.where(temp_df.up_first, temp_df.low)
        temp_df[self.colprefix + 'tr_percent'] = temp_df[self.colprefix + 'true_range'] / denom

        temp_df.rename({'open': self.colprefix + 'open', 'high': self.colprefix + 'high',
                        'low': self.colprefix + 'low', 'close': self.colprefix + 'close',
                        'volume': self.colprefix + 'volume', 'up_first': self.colprefix + 'up_first'},
                       axis=1, inplace=True)

        temp_df.drop(columns=['previous_open', 'previous_close'], inplace=True)

        return temp_df

    def _generate_indicators(self, processed_df):
        temp_df = processed_df.copy()
        indicators_df = add_all_ta_features(temp_df, 'open', 'high', 'low', 'close', 'volume',
                                            colprefix=self.colprefix)
        indicators_df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'trend_psar_up',
                                    'trend_psar_down', 'up_first'], inplace=True)
        return indicators_df

    def _generate_def_indicators(self, processed_df):
        temp_df = processed_df.copy()
        indicators_df = add_all_stationary_ta_features(temp_df, 'open', 'high', 'low', 'close', 'volume',
                                                       'up_first', colprefix=self.colprefix)
        indicators_df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'up_first'], inplace=True)
        return indicators_df

    def _generate_log_indicators(self, processed_df):
        temp_df = processed_df.copy()
        indicators_df = add_all_log_stationary_ta_features(temp_df, 'open', 'high', 'low', 'close', 'volume',
                                                           'up_first', colprefix=self.colprefix)
        indicators_df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'up_first'], inplace=True)
        return indicators_df

    @staticmethod
    def _create_features_df(alt_candles_df, indicators_df):
        df = pd.concat([alt_candles_df, indicators_df], axis=1)
        return df

    def save_results(self):
        self.features_df.to_csv(f'Data//PreprocessedData//BTCUSDT_{self.timeframe}.csv')
        self.features_df.drop(self.features_df.head(70).index, inplace=True)
        self.features_df.drop(self.features_df.tail(5).index, inplace=True)
        if self.use_stationary_ta and (self.method == 'default'):
            self.features_df.to_csv(f'Data//PreprocessedData//BTCUSDT_{self.timeframe}_3.csv')
        elif self.use_stationary_ta and (self.method == 'log'):
            self.features_df.to_csv(f'Data//PreprocessedData//BTCUSDT_{self.timeframe}_log_3.csv')
        else:
            self.features_df.to_csv(f'Data//PreprocessedData//BTCUSDT_{self.timeframe}_non_3.csv')


class Interpolator:
    def __init__(self, df: pd.DataFrame, time_frame_in_minutes=1440, first_day=TimeStamps.start.value,
                 last_day=TimeStamps.end2.value, causal=True, debug=False):
        self.df = df.copy()
        self.df_columns = self.df.columns.to_list()
        self.colprefix = self.df_columns[0].split('_')[0]
        self.time_frame = time_frame_in_minutes
        self.number_of_candles_in_each_day = int(1440 / time_frame_in_minutes)
        self.first_day = datetime.strptime(first_day, '%Y-%m-%d %H:%M:%S')
        self.last_day = datetime.strptime(last_day, '%Y-%m-%d %H:%M:%S')
        self.total_number_of_missing_candles = 0
        self.number_of_candles_with_bad_index = 0
        self.number_of_candles_with_two_index = 0
        self.causal = causal
        if debug:
            self._run_with_comment()
        else:
            self._run()

    def _run_with_comment(self):
        self._assert_existence_of_last_candle()
        self._clip_the_dataframe()
        self._adjust_dataframe_indices()
        self._reset_first_candle()
        date = self.first_day
        while date < self.last_day:
            if self._given_day_has_missing_data(date):
                print(f'At the day: {date.strftime("%Y-%m-%d")}\n')
                number_of_missing_candles = 0
                if date.date() == self.last_day.date():
                    gen = self._list_of_last_days_candles(date)
                else:
                    gen = self._list_of_daily_candles(date)
                for start_of_candle in gen:
                    if start_of_candle not in self.df.index:
                        n = self._impute_the_missing_candle(start_of_candle)
                        number_of_missing_candles += n

                print(
                    f'\n{number_of_missing_candles} data {"is" if number_of_missing_candles == 1 else "are"} '
                    f'missing.The repaird data is shown below:\n')
                # print(self.df.loc[date.strftime('%Y-%m-%d')])
                print('\n\n')
            date = date + timedelta(days=1)
        self._make_sure_no_where_volume_is_zero_or_negative()
        self.df.sort_index(inplace=True)
        print('\n total number of missing data: ', self.total_number_of_missing_candles)

    def _run(self):
        self._assert_existence_of_last_candle()
        self._clip_the_dataframe()
        self._adjust_dataframe_indices()
        self._reset_first_candle()
        date = self.first_day
        while date < self.last_day:
            if self._given_day_has_missing_data(date):
                if date.date() == self.last_day.date():
                    gen = self._list_of_last_days_candles(date)
                else:
                    gen = self._list_of_daily_candles(date)
                for start_of_candle in gen:
                    if start_of_candle not in self.df.index:
                        self._impute_the_missing_candle(start_of_candle)
            date = date + timedelta(days=1)
        if 'volume' in self.df_columns:
            self._make_sure_no_where_volume_is_zero_or_negative()
        self.df.sort_index(inplace=True)

    def _assert_existence_of_last_candle(self):
        if self.last_day - timedelta(seconds=1) not in self.df.index.tolist():
            raise Exception('last candle must exist in the given dataframe!')

    def _clip_the_dataframe(self):
        self.df = self.df.loc[self.first_day: self.last_day - timedelta(seconds=1), :]

    def _adjust_dataframe_indices(self):
        list_of_df_indices = self.df.index.tolist()
        list_of_moved_duplicates_index = []
        i = 0
        for idx, item in enumerate(list_of_df_indices):
            if (item.minute % self.time_frame) + item.second + item.microsecond > 0:
                offset = timedelta(minutes=(item.minute % self.time_frame), seconds=item.second,
                                   microseconds=item.microsecond)
                new_idx = item - offset
                if new_idx in list_of_df_indices:
                    list_of_df_indices[idx] = self.last_day + timedelta(seconds=idx + 1)
                    list_of_moved_duplicates_index.append(self.last_day + timedelta(seconds=idx + 1))
                else:
                    list_of_df_indices[idx] = new_idx
                i = i + 1
        self.df.index = list_of_df_indices
        self.number_of_candles_with_bad_index = i
        j = 0
        for idx in list_of_moved_duplicates_index:
            self.df.drop(idx, inplace=True)
            j = j + 1
        self.df.index.set_names('datetime', inplace=True)
        self.number_of_candles_with_two_index = j

    def _reset_first_candle(self):
        self.df.loc[self.first_day] = self.df.iloc[0, :]

    def _given_day_has_missing_data(self, date):
        return len(self.df.loc[date.strftime('%Y-%m-%d')]) < self.number_of_candles_in_each_day

    def _list_of_daily_candles(self, date):
        for i in range(self.number_of_candles_in_each_day):
            yield date + timedelta(minutes=i * self.time_frame)

    def _list_of_last_days_candles(self, date):
        last_date = datetime.strptime(self.last_day.date().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
        number_of_candles = int(np.floor((self.last_day - last_date).seconds / (60 * self.time_frame))) + 1
        for i in range(number_of_candles):
            yield date + timedelta(minutes=i * self.time_frame)

    def _impute_the_missing_candle(self, start_of_missing_candle):
        number_of_missing_candles, start_of_next_available_candle = \
            self._find_next_available_candle_and_number_of_missing_candles(start_of_missing_candle)
        if self.causal:
            self._causal_estimate_ohlcv_and_insert_the_candles(start_of_missing_candle, number_of_missing_candles)
        else:
            self._estimate_ohlcv_and_insert_the_candles(start_of_missing_candle, start_of_next_available_candle,
                                                        number_of_missing_candles)
        return number_of_missing_candles

    def _find_next_available_candle_and_number_of_missing_candles(self, current_index):
        i = 0
        t = current_index
        while t not in self.df.index:
            i += 1
            t += timedelta(minutes=self.time_frame)
            if t > self.last_day:
                raise Exception('Last candle was missing from the dataframe!')
        return i, t

    def _causal_estimate_ohlcv_and_insert_the_candles(self, missing_candle_idx, number_of_candles):
        for _ in range(number_of_candles):
            prev_index = missing_candle_idx - timedelta(minutes=self.time_frame)

            self.df.loc[missing_candle_idx] = self.df.loc[prev_index].to_dict()

            missing_candle_idx += timedelta(minutes=self.time_frame)
            self._increase_number_of_missing_candles()

    def _estimate_ohlcv_and_insert_the_candles(self, missing_candle_idx, next_index, number_of_candles):
        for _ in range(number_of_candles):
            prev_index = missing_candle_idx - timedelta(minutes=self.time_frame)

            open_ = self.df.loc[prev_index].close
            high_ = (number_of_candles * self.df.loc[prev_index].high + self.df.loc[next_index].high) / (
                    number_of_candles + 1)
            low_ = (number_of_candles * self.df.loc[prev_index].low + self.df.loc[next_index].low) / (
                    number_of_candles + 1)
            close_ = ((number_of_candles - 1) * open_ + self.df.loc[next_index].open) / number_of_candles
            if 'volume' in self.df_columns:
                volume = (self.df.loc[prev_index].volume + self.df.loc[next_index].volume) / 2
                self.df.loc[missing_candle_idx] = {'open': open_, 'high': high_, 'low': low_, 'close': close_,
                                                   'volume': volume}
            else:
                self.df.loc[missing_candle_idx] = {'open': open_, 'high': high_, 'low': low_, 'close': close_}

            missing_candle_idx += timedelta(minutes=self.time_frame)
            number_of_candles += -1
            self._increase_number_of_missing_candles()

    def _increase_number_of_missing_candles(self):
        self.total_number_of_missing_candles += 1

    def _make_sure_no_where_volume_is_zero_or_negative(self):
        self.df.volume[self.df.volume < 1] = 0.1

    def return_complete_dataframe(self):
        return self.df


class UpFirstDetector:
    def __init__(self, df, time_frame_in_minutes=1440, first_day=TimeStamps.start.value,
                 last_day=TimeStamps.end2.value):
        self.df = df
        self.one_min_df = self._load_one_min_df()
        self.time_frame = time_frame_in_minutes
        self.first_candle = datetime.strptime(first_day, '%Y-%m-%d %H:%M:%S')
        self.last_candle = datetime.strptime(last_day, '%Y-%m-%d %H:%M:%S')
        self.run()

    def run(self):
        start_of_candle = self.first_candle
        up_first_column = []
        while start_of_candle < self.last_candle:
            if self._max_happened_before_min_in_this_candle(start_of_candle):
                up_first_column.append(True)
            else:
                up_first_column.append(False)
            start_of_candle += timedelta(minutes=self.time_frame)
        self.df['up_first'] = up_first_column

    @staticmethod
    def _load_one_min_df():
        df = pd.read_csv('Data//PreprocessedData/BTCUSDT_1.csv')
        df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%d %H:%M:%S")
        df.sort_values('datetime', inplace=True)
        df.set_index('datetime', inplace=True)
        return df

    def _max_happened_before_min_in_this_candle(self, start):
        end = start + timedelta(minutes=self.time_frame - 1)
        temp_df = self.one_min_df.loc[start.strftime('%Y-%m-%d %H:%M:%S'): end.strftime('%Y-%m-%d %H:%M:%S')]
        index_of_max = temp_df.high.idxmax()
        index_of_min = temp_df.low.idxmin()

        if index_of_max == index_of_min:
            return temp_df.loc[index_of_max, 'close'] <= temp_df.loc[index_of_max, 'open']
            # print('idx= ', index_of_max)
            # plt.figure()
            # plot_chart(temp_df)
            # plt.show()

        return index_of_max < index_of_min

    def return_new_df(self):
        return self.df


class TargetExtractor:
    def __init__(self, colprefix='BTCUSDT_'):
        l = len(colprefix)

        df_15 = import_processed_data(f'Data//PreprocessedData//BTCUSDT_15.csv', index_column_name='datetime')
        df_15.rename({name: name[l:] for name in df_15.columns.to_list()}, axis=1, inplace=True)

        self.df = df_15.loc[:, ['open', 'high', 'low', 'close']]
        self.targets_df = None

    def run(self):
        targets = self._generate_the_targets_df()
        number_of_positive_targets = len(targets[targets.target3 == 1])
        number_of_negative_targets = len(targets[targets.target3 == -1])
        total_cases = len(targets)
        print(f'for target3 the number of positive cases is: {number_of_positive_targets} :'
              f' {100 * number_of_positive_targets / total_cases}%')
        print(f'for target3 the number of Negative cases is: {number_of_negative_targets} :'
              f' {100 * number_of_negative_targets / total_cases}%')
        self.save_results(targets)

    def _generate_the_targets_df(self):
        df = self.df.copy()

        df['target3'] = df.close < df.close.shift(-1)
        df['target3'] = df['target3'].astype('int32')

        df.loc[df.target3 == 1, 'target'] = (df.close.shift(-1) - df.close) / df.close
        df.loc[df.target3 != 1, 'target'] = 0
        df.loc[df.target3 == 1, 'stop'] = 0
        df.loc[df.target3 != 1, 'stop'] = (df.close.shift(-1) - df.close) / df.close
        df.fillna(value=33, inplace=True)

        return df.loc[:, ['target3', 'target', 'stop', 'open', 'high', 'low', 'close']]

    def save_results(self, targets_df):
        target3_df = self._drop_the_formerly_repaired_nan_values_for_given_target(targets_df.copy())
        target3_df.to_csv(f'Data//PreprocessedData//BTCUSDT_def_targets3.csv')

    def _drop_the_formerly_repaired_nan_values_for_given_target(self, df):
        df.drop(df.head(70).index, inplace=True)
        df.drop(df.tail(5).index, inplace=True)
        return df


class DataMixer:
    def __init__(self, df_name, timeframe, first_day, last_day, log_features=False, exclude_new_features=False):
        self.df_name = df_name
        self.timeframe = timeframe
        self.first_day = datetime.strptime(first_day, '%Y-%m-%d %H:%M:%S')
        self.last_day = datetime.strptime(last_day, '%Y-%m-%d %H:%M:%S')
        self.use_log_features = log_features
        self.exclude_new_features = exclude_new_features
        self.features_df = None
        self.targets_df = None

    def run(self):
        features_address, targets_address = self.get_address_of_features_and_targets()

        self.features_df = self.load_features(features_address)
        if self.exclude_new_features:
            self.features_df = self.features_df.loc[:,
                               StationaryFeatures.generate_excluded_stationary_features_list('')]
        else:
            self.features_df = self.features_df.loc[:, StationaryFeatures.generate_stationary_features_list('')]
        self.targets_df = self.load_targets(targets_address)

        self.save_the_result()

    def get_address_of_features_and_targets(self):
        if self.use_log_features:
            features_address = f'Data//PreprocessedData//BTCUSDT_{self.timeframe}_log_3.csv'
        else:
            features_address = f'Data//PreprocessedData//BTCUSDT_{self.timeframe}_3.csv'
        targets_address = f'Data//PreprocessedData//BTCUSDT_def_targets3.csv'
        return features_address, targets_address

    def load_features(self, features_address):
        btc_df = import_processed_data(features_address, index_column_name='datetime')
        btc_df = btc_df.loc[self.first_day: self.last_day]
        btc_df = btc_df.fillna(method='bfill')

        return btc_df

    def load_targets(self, targets_address):
        targets_df = import_processed_data(targets_address, index_column_name='datetime')
        return targets_df.loc[self.first_day:self.last_day]

    def save_the_result(self):
        if self.use_log_features:
            self.features_df.to_csv(f'Data//MixedData//{self.df_name}_logfeatures.csv')
            shutil.copy(f'Data//MixedData//{self.df_name}_logfeatures.csv',
                        f'Data//ProcessedData//{self.df_name}_log_ready_features.csv')
        elif self.exclude_new_features:
            self.features_df.to_csv(f'Data//MixedData//{self.df_name}_excludedfeatures.csv')
            shutil.copy(f'Data//MixedData//{self.df_name}_excludedfeatures.csv',
                        f'Data//ProcessedData//{self.df_name}_excluded_ready_features.csv')
        else:
            self.features_df.to_csv(f'Data//MixedData//{self.df_name}_features.csv')
            shutil.copy(f'Data//MixedData//{self.df_name}_features.csv',
                        f'Data//ProcessedData//{self.df_name}_ready_features.csv')
        self.targets_df.to_csv(f'Data//MixedData//{self.df_name}_def_targets.csv')
        shutil.copy(f'Data//MixedData//{self.df_name}_def_targets.csv',
                    f'Data//ProcessedData//{self.df_name}_def_targets.csv')


class DefTrainTestValidationLoader:
    def __init__(self, model_name, data_name, target, raw_targets=None, training_portion=0.75, test_portion=0.4,
                 n_input_steps=21, feature_scaler_range=(-1, 1), target_scaler_range=(-1, 1), update_scaler=False,
                 silent=False):
        self.model_name = model_name
        self.silent = silent
        features_df, labels_df = self.get_features_and_labels(data_name)
        self.targets = [target]
        self.update_scaler = update_scaler
        if raw_targets is None:
            self.raw_targets = [target]
            self.common_features = []
        else:
            self.raw_targets = raw_targets
            self.common_features = [feature for feature in raw_targets if feature in features_df.columns.tolist()]

        self.full_df = self._mix_features_and_targets(features_df, labels_df)
        self.number_of_input_steps = n_input_steps
        self.feature_scaler = MinMaxScaler(feature_range=feature_scaler_range)
        self.target_scaler = MinMaxScaler(feature_range=target_scaler_range)
        self.x_shape = (-1, self.number_of_input_steps, len(features_df.columns.to_list()))

        self._run(training_portion, test_portion)

    def _run(self, training_portion, test_portion):
        df_train, df_val, df_test = self._train_test_val_split(training_portion, test_portion)

        if self.update_scaler:
            scaled_train_features, scaled_train_labels = self._scale(df_train, fit=True)
        else:
            self.feature_scaler = load(open(f'Files/feature_scaler/{self.model_name}_feature_scaler.pkl', 'rb'))
            self.target_scaler = load(open(f'Files/feature_scaler/{self.model_name}_target_scaler.pkl', 'rb'))
            scaled_train_features, scaled_train_labels = self._scale(df_train, fit=False)
        scaled_val_features, scaled_val_labels = self._scale(df_val, fit=False)
        scaled_test_features, scaled_test_labels = self._scale(df_test, fit=False)

        self.x_train, self.y_train, self.y2_train = self._reframe(scaled_train_features, scaled_train_labels)
        self.x_val, self.y_val, self.y2_val = self._reframe(scaled_val_features, scaled_val_labels)
        self.x_test, self.y_test, self.y2_test = self._reframe(scaled_test_features, scaled_test_labels)

    @staticmethod
    def get_features_and_labels(data_name):
        features_address = f'Data//ProcessedData//{data_name}_ready_features.csv'
        features_df = import_processed_data(features_address, index_column_name='datetime')
        labels_address = f'Data//ProcessedData//{data_name}_def_targets.csv'
        labels_df = import_processed_data(labels_address, index_column_name='datetime')
        return features_df, labels_df

    def _mix_features_and_targets(self, features_df, labels_df):
        df = features_df.copy()
        df[self.raw_targets] = labels_df[self.raw_targets]
        return df

    def _train_test_val_split(self, train_portion, test_portion):
        temp_df = self.full_df
        size1 = int(len(temp_df) * train_portion)
        df_train = temp_df[0:size1].copy()
        df_test_and_val = temp_df[size1 - (self.number_of_input_steps - 1):].copy()
        size2 = int(len(df_test_and_val) * (1 - test_portion))
        df_val = df_test_and_val[0:size2].copy()
        df_test = df_test_and_val[size2 - (self.number_of_input_steps - 1):].copy()
        if not self.silent:
            print(f'first candle in train set: {df_train.iloc[0, :].name}')
            print(f'last candle in train set: {df_train.iloc[len(df_train) - 1, :].name}')
            print(f'first candle in val set: {df_val.iloc[0, :].name}')
            print(f'last candle in val set: {df_val.iloc[len(df_val) - 1, :].name}')
            print(f'first candle in test set: {df_test.iloc[0, :].name}')
            print(f'last candle in test set: {df_test.iloc[len(df_test) - 1, :].name}')
        return df_train, df_val, df_test

    def _scale(self, df, fit=True):
        features_to_drop = [feature for feature in self.raw_targets if feature not in self.common_features]
        features = df.drop(columns=features_to_drop)
        targets = df.loc[:, self.targets]
        if fit:
            self.feature_scaler.fit(features.values)
            dump(self.feature_scaler, open(f'Files/feature_scaler/{self.model_name}_feature_scaler.pkl', 'wb'))
            self.target_scaler.fit(targets.values)
            dump(self.target_scaler, open(f'Files/feature_scaler/{self.model_name}_target_scaler.pkl', 'wb'))
        features[features.columns.to_list()] = self.feature_scaler.transform(features.values)
        targets[targets.columns.to_list()] = self.target_scaler.transform(targets.values)
        complete_targets = pd.DataFrame(df.loc[:, self.raw_targets])
        complete_targets[self.targets] = targets.loc[:, self.targets]
        return features, complete_targets

    def _reframe(self, features, labels):
        stacked_features_df = pd.DataFrame()
        for i in range(self.number_of_input_steps - 1, -1, -1):
            cols = features.shift(i)
            rename_dict = {x: x + f'_minus_{i}_of_n_{1}' for x in list(features.columns)}
            cols = cols.rename(columns=rename_dict)
            stacked_features_df = pd.concat([stacked_features_df, cols], axis=1)
        stacked_features_df.drop(stacked_features_df.head((self.number_of_input_steps - 1)).index, inplace=True)

        labels_df = labels.drop(labels.head((self.number_of_input_steps - 1)).index)

        return stacked_features_df, labels_df.loc[:, self.targets], labels_df.drop(self.targets, axis=1)

    def get_reframed_train_data(self):
        x = np.asarray(self.x_train.values.reshape(self.x_shape)).astype('float64')
        y = np.asarray(self.y_train.values.reshape(-1, 1, 1)).astype('float64')
        return x, y

    def get_reframed_val_data(self):
        x = np.asarray(self.x_val.values.reshape(self.x_shape)).astype('float64')
        y = np.asarray(self.y_val.values.reshape(-1, 1, 1)).astype('float64')
        return x, y

    def get_reframed_test_data(self):
        x = np.asarray(self.x_test.values.reshape(self.x_shape)).astype('float64')
        y = np.asarray(self.y_test.values.reshape(-1, 1, 1)).astype('float64')
        return x, y

    def load_train_test_validation_data(self):
        return self.get_reframed_train_data(), self.get_reframed_val_data(), self.get_reframed_test_data()


class LogTrainTestValidationLoader:
    def __init__(self, model_name, data_name, target, raw_targets=None, training_portion=0.75, test_portion=0.4,
                 n_input_steps=21, feature_scaler_range=(-1, 1), target_scaler_range=(-1, 1), update_scaler=False,
                 silent=False):
        self.model_name = model_name
        self.silent = silent
        features_df, labels_df = self.get_features_and_labels(data_name)
        self.targets = [target]
        self.update_scaler = update_scaler
        if raw_targets is None:
            self.raw_targets = [target]
            self.common_features = []
        else:
            self.raw_targets = raw_targets
            self.common_features = [feature for feature in raw_targets if feature in features_df.columns.tolist()]

        self.full_df = self._mix_features_and_targets(features_df, labels_df)
        self.number_of_input_steps = n_input_steps
        self.feature_scaler = MinMaxScaler(feature_range=feature_scaler_range)
        self.target_scaler = MinMaxScaler(feature_range=target_scaler_range)
        self.x_shape = (-1, self.number_of_input_steps, len(features_df.columns.to_list()))

        self._run(training_portion, test_portion)

    def _run(self, training_portion, test_portion):
        df_train, df_val, df_test = self._train_test_val_split(training_portion, test_portion)

        if self.update_scaler:
            scaled_train_features, scaled_train_labels = self._scale(df_train, fit=True)
        else:
            self.feature_scaler = load(open(f'Files/feature_scaler/{self.model_name}_feature_scaler.pkl', 'rb'))
            self.target_scaler = load(open(f'Files/feature_scaler/{self.model_name}_target_scaler.pkl', 'rb'))
            scaled_train_features, scaled_train_labels = self._scale(df_train, fit=False)

        scaled_val_features, scaled_val_labels = self._scale(df_val, fit=False)
        scaled_test_features, scaled_test_labels = self._scale(df_test, fit=False)

        self.x_train, self.y_train, self.y2_train = self._reframe(scaled_train_features, scaled_train_labels)
        self.x_val, self.y_val, self.y2_val = self._reframe(scaled_val_features, scaled_val_labels)
        self.x_test, self.y_test, self.y2_test = self._reframe(scaled_test_features, scaled_test_labels)

    @staticmethod
    def get_features_and_labels(data_name):
        features_address = f'Data//ProcessedData//{data_name}_log_ready_features.csv'
        features_df = import_processed_data(features_address, index_column_name='datetime')
        labels_address = f'Data//ProcessedData//{data_name}_def_targets.csv'
        labels_df = import_processed_data(labels_address, index_column_name='datetime')
        return features_df, labels_df

    def _mix_features_and_targets(self, features_df, labels_df):
        df = features_df.copy()
        df[self.raw_targets] = labels_df[self.raw_targets]
        return df

    def _train_test_val_split(self, train_portion, test_portion):
        temp_df = self.full_df
        size1 = int(len(temp_df) * train_portion)
        df_train = temp_df[0:size1].copy()
        df_test_and_val = temp_df[size1 - (self.number_of_input_steps - 1):].copy()
        size2 = int(len(df_test_and_val) * (1 - test_portion))
        df_val = df_test_and_val[0:size2].copy()
        df_test = df_test_and_val[size2 - (self.number_of_input_steps - 1):].copy()
        if not self.silent:
            print(f'first candle in train set: {df_train.iloc[0, :].name}')
            print(f'last candle in train set: {df_train.iloc[len(df_train) - 1, :].name}')
            print(f'first candle in val set: {df_val.iloc[0, :].name}')
            print(f'last candle in val set: {df_val.iloc[len(df_val) - 1, :].name}')
            print(f'first candle in test set: {df_test.iloc[0, :].name}')
            print(f'last candle in test set: {df_test.iloc[len(df_test) - 1, :].name}')
        return df_train, df_val, df_test

    def _scale(self, df, fit=True):
        features_to_drop = [feature for feature in self.raw_targets if feature not in self.common_features]
        features = df.drop(columns=features_to_drop)
        targets = df.loc[:, self.targets]
        if fit:
            self.feature_scaler.fit(features.values)
            dump(self.feature_scaler, open(f'Files/feature_scaler/{self.model_name}_feature_scaler.pkl', 'wb'))
            self.target_scaler.fit(targets.values)
            dump(self.target_scaler, open(f'Files/feature_scaler/{self.model_name}_target_scaler.pkl', 'wb'))
        features[features.columns.to_list()] = self.feature_scaler.transform(features.values)
        targets[targets.columns.to_list()] = self.target_scaler.transform(targets.values)
        complete_targets = pd.DataFrame(df.loc[:, self.raw_targets])
        complete_targets[self.targets] = targets.loc[:, self.targets]
        return features, complete_targets

    def _reframe(self, features, labels):
        stacked_features_df = pd.DataFrame()
        for i in range(self.number_of_input_steps - 1, -1, -1):
            cols = features.shift(i)
            rename_dict = {x: x + f'_minus_{i}_of_n_{1}' for x in list(features.columns)}
            cols = cols.rename(columns=rename_dict)
            stacked_features_df = pd.concat([stacked_features_df, cols], axis=1)
        stacked_features_df.drop(stacked_features_df.head((self.number_of_input_steps - 1)).index, inplace=True)

        labels_df = labels.drop(labels.head((self.number_of_input_steps - 1)).index)

        return stacked_features_df, labels_df.loc[:, self.targets], labels_df.drop(self.targets, axis=1)

    def get_reframed_train_data(self):
        x = np.asarray(self.x_train.values.reshape(self.x_shape)).astype('float64')
        y = np.asarray(self.y_train.values.reshape(-1, 1, 1)).astype('float64')
        return x, y

    def get_reframed_val_data(self):
        x = np.asarray(self.x_val.values.reshape(self.x_shape)).astype('float64')
        y = np.asarray(self.y_val.values.reshape(-1, 1, 1)).astype('float64')
        return x, y

    def get_reframed_test_data(self):
        x = np.asarray(self.x_test.values.reshape(self.x_shape)).astype('float64')
        y = np.asarray(self.y_test.values.reshape(-1, 1, 1)).astype('float64')
        return x, y

    def load_train_test_validation_data(self):
        return self.get_reframed_train_data(), self.get_reframed_val_data(), self.get_reframed_test_data()


class ExTrainTestValidationLoader:
    def __init__(self, model_name, data_name, target, raw_targets=None, training_portion=0.75, test_portion=0.4,
                 n_input_steps=21, feature_scaler_range=(-1, 1), target_scaler_range=(-1, 1), update_scaler=False,
                 silent=False):
        self.model_name = model_name
        self.silent = silent
        features_df, labels_df = self.get_features_and_labels(data_name)
        self.targets = [target]
        self.update_scaler = update_scaler
        if raw_targets is None:
            self.raw_targets = [target]
            self.common_features = []
        else:
            self.raw_targets = raw_targets
            self.common_features = [feature for feature in raw_targets if feature in features_df.columns.tolist()]

        self.full_df = self._mix_features_and_targets(features_df, labels_df)
        self.number_of_input_steps = n_input_steps
        self.feature_scaler = MinMaxScaler(feature_range=feature_scaler_range)
        self.target_scaler = MinMaxScaler(feature_range=target_scaler_range)
        self.x_shape = (-1, self.number_of_input_steps, len(features_df.columns.to_list()))

        self._run(training_portion, test_portion)

    def _run(self, training_portion, test_portion):
        df_train, df_val, df_test = self._train_test_val_split(training_portion, test_portion)

        if self.update_scaler:
            scaled_train_features, scaled_train_labels = self._scale(df_train, fit=True)
        else:
            self.feature_scaler = load(open(f'Files/feature_scaler/{self.model_name}_feature_scaler.pkl', 'rb'))
            self.target_scaler = load(open(f'Files/feature_scaler/{self.model_name}_target_scaler.pkl', 'rb'))
            scaled_train_features, scaled_train_labels = self._scale(df_train, fit=False)
        scaled_val_features, scaled_val_labels = self._scale(df_val, fit=False)
        scaled_test_features, scaled_test_labels = self._scale(df_test, fit=False)

        self.x_train, self.y_train, self.y2_train = self._reframe(scaled_train_features, scaled_train_labels)
        self.x_val, self.y_val, self.y2_val = self._reframe(scaled_val_features, scaled_val_labels)
        self.x_test, self.y_test, self.y2_test = self._reframe(scaled_test_features, scaled_test_labels)

    @staticmethod
    def get_features_and_labels(data_name):
        features_address = f'Data//ProcessedData//{data_name}_excluded_ready_features.csv'
        features_df = import_processed_data(features_address, index_column_name='datetime')
        labels_address = f'Data//ProcessedData//{data_name}_def_targets.csv'
        labels_df = import_processed_data(labels_address, index_column_name='datetime')
        return features_df, labels_df

    def _mix_features_and_targets(self, features_df, labels_df):
        df = features_df.copy()
        df[self.raw_targets] = labels_df[self.raw_targets]
        return df

    def _train_test_val_split(self, train_portion, test_portion):
        temp_df = self.full_df
        size1 = int(len(temp_df) * train_portion)
        df_train = temp_df[0:size1].copy()
        df_test_and_val = temp_df[size1 - (self.number_of_input_steps - 1):].copy()
        size2 = int(len(df_test_and_val) * (1 - test_portion))
        df_val = df_test_and_val[0:size2].copy()
        df_test = df_test_and_val[size2 - (self.number_of_input_steps - 1):].copy()
        if not self.silent:
            print(f'first candle in train set: {df_train.iloc[0, :].name}')
            print(f'last candle in train set: {df_train.iloc[len(df_train) - 1, :].name}')
            print(f'first candle in val set: {df_val.iloc[0, :].name}')
            print(f'last candle in val set: {df_val.iloc[len(df_val) - 1, :].name}')
            print(f'first candle in test set: {df_test.iloc[0, :].name}')
            print(f'last candle in test set: {df_test.iloc[len(df_test) - 1, :].name}')
        return df_train, df_val, df_test

    def _scale(self, df, fit=True):
        features_to_drop = [feature for feature in self.raw_targets if feature not in self.common_features]
        features = df.drop(columns=features_to_drop)
        targets = df.loc[:, self.targets]
        if fit:
            self.feature_scaler.fit(features.values)
            dump(self.feature_scaler, open(f'Files/feature_scaler/{self.model_name}_feature_scaler.pkl', 'wb'))
            self.target_scaler.fit(targets.values)
            dump(self.target_scaler, open(f'Files/feature_scaler/{self.model_name}_target_scaler.pkl', 'wb'))
        features[features.columns.to_list()] = self.feature_scaler.transform(features.values)
        targets[targets.columns.to_list()] = self.target_scaler.transform(targets.values)
        complete_targets = pd.DataFrame(df.loc[:, self.raw_targets])
        complete_targets[self.targets] = targets.loc[:, self.targets]
        return features, complete_targets

    def _reframe(self, features, labels):
        stacked_features_df = pd.DataFrame()
        for i in range(self.number_of_input_steps - 1, -1, -1):
            cols = features.shift(i)
            rename_dict = {x: x + f'_minus_{i}_of_n_{1}' for x in list(features.columns)}
            cols = cols.rename(columns=rename_dict)
            stacked_features_df = pd.concat([stacked_features_df, cols], axis=1)
        stacked_features_df.drop(stacked_features_df.head((self.number_of_input_steps - 1)).index, inplace=True)

        labels_df = labels.drop(labels.head((self.number_of_input_steps - 1)).index)

        return stacked_features_df, labels_df.loc[:, self.targets], labels_df.drop(self.targets, axis=1)

    def get_reframed_train_data(self):
        x = np.asarray(self.x_train.values.reshape(self.x_shape)).astype('float64')
        y = np.asarray(self.y_train.values.reshape(-1, 1, 1)).astype('float64')
        return x, y

    def get_reframed_val_data(self):
        x = np.asarray(self.x_val.values.reshape(self.x_shape)).astype('float64')
        y = np.asarray(self.y_val.values.reshape(-1, 1, 1)).astype('float64')
        return x, y

    def get_reframed_test_data(self):
        x = np.asarray(self.x_test.values.reshape(self.x_shape)).astype('float64')
        y = np.asarray(self.y_test.values.reshape(-1, 1, 1)).astype('float64')
        return x, y

    def load_train_test_validation_data(self):
        return self.get_reframed_train_data(), self.get_reframed_val_data(), self.get_reframed_test_data()
