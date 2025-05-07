"""
Модуль для обработки и подготовки данных для моделей.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Класс для обработки данных перед подачей в модели.
    """

    def __init__(self, scaling_method: str = 'minmax'):
        """
        Инициализирует процессор данных.

        Args:
            scaling_method: Метод масштабирования ('minmax' или 'standard')
        """
        self.scaling_method = scaling_method

        if scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling_method == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Неподдерживаемый метод масштабирования: {scaling_method}")

        logger.info(f"Initialized DataProcessor with {scaling_method} scaling")

    def process_data(
            self,
            df: pd.DataFrame,
            feature_columns: List[str],
            target_column: Optional[str] = None,
            test_size: float = 0.2,
            validation_size: float = 0.1,
            sequence_length: int = 60,
            prediction_horizon: int = 1,
            shuffle: bool = True,
            random_state: int = 42
    ) -> Dict:
        """
        Обрабатывает данные для обучения и тестирования.

        Args:
            df: DataFrame с данными
            feature_columns: Список колонок-признаков
            target_column: Колонка-цель (опционально)
            test_size: Доля данных для тестирования
            validation_size: Доля данных для валидации
            sequence_length: Длина последовательности (окно) для временных признаков
            prediction_horizon: Горизонт прогнозирования
            shuffle: Перемешивать ли данные
            random_state: Случайное состояние для воспроизводимости

        Returns:
            Словарь с обработанными данными
        """
        logger.info(f"Processing data with {len(feature_columns)} features")

        # Создание копии для обработки
        data = df.copy()

        # Удаление строк с пропущенными значениями
        data = data.dropna(subset=feature_columns + ([target_column] if target_column else []))

        # Масштабирование признаков
        feature_data = data[feature_columns].values
        scaled_features = self.scaler.fit_transform(feature_data)

        # Создание целевой переменной (если указана)
        if target_column:
            # Если цель - это сдвинутая вперед цена, создаем ее
            if target_column == 'future_price':
                target_data = data['close'].shift(-prediction_horizon).values[:-prediction_horizon]
                scaled_features = scaled_features[:-prediction_horizon]
            else:
                target_data = data[target_column].values

            # Создание последовательностей (для LSTM и других последовательных моделей)
            X, y = self._create_sequences(
                scaled_features,
                target_data,
                sequence_length=sequence_length
            )

            # Разделение на тренировочную, валидационную и тестовую выборки
            # Сначала выделяем тестовую часть
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=shuffle, random_state=random_state
            )

            # Затем из оставшихся данных выделяем валидационную часть
            validation_ratio = validation_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=validation_ratio, shuffle=shuffle, random_state=random_state
            )

            result = {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'scaler': self.scaler,
                'feature_columns': feature_columns,
                'target_column': target_column,
                'sequence_length': sequence_length
            }
        else:
            # Если цель не указана, просто создаем последовательности признаков (для RL)
            X = self._create_feature_sequences(scaled_features, sequence_length)

            # Разделение на тренировочную, валидационную и тестовую выборки
            total_samples = len(X)
            test_count = int(total_samples * test_size)
            validation_count = int(total_samples * validation_size)
            train_count = total_samples - test_count - validation_count

            if shuffle:
                indices = np.random.RandomState(random_state).permutation(total_samples)
                X = X[indices]

            X_train = X[:train_count]
            X_val = X[train_count:train_count + validation_count]
            X_test = X[train_count + validation_count:]

            result = {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'scaler': self.scaler,
                'feature_columns': feature_columns,
                'sequence_length': sequence_length
            }

        logger.info(
            f"Processed data shapes: {', '.join([f'{k}: {v.shape}' for k, v in result.items() if isinstance(v, np.ndarray)])}")

        return result

    def _create_sequences(
            self,
            features: np.ndarray,
            targets: np.ndarray,
            sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Создает последовательности признаков и целевых значений.

        Args:
            features: Массив признаков
            targets: Массив целевых значений
            sequence_length: Длина последовательности

        Returns:
            Кортеж (X, y) с последовательностями признаков и целевыми значениями
        """
        X, y = [], []

        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
            y.append(targets[i + sequence_length])

        return np.array(X), np.array(y)

    def _create_feature_sequences(
            self,
            features: np.ndarray,
            sequence_length: int
    ) -> np.ndarray:
        """
        Создает последовательности только из признаков.

        Args:
            features: Массив признаков
            sequence_length: Длина последовательности

        Returns:
            Массив последовательностей признаков
        """
        X = []

        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])

        return np.array(X)

    def inverse_scale(self, scaled_data: np.ndarray, column_indices: List[int]) -> np.ndarray:
        """
        Обратное масштабирование данных.

        Args:
            scaled_data: Масштабированные данные
            column_indices: Индексы колонок для обратного масштабирования

        Returns:
            Данные в исходном масштабе
        """
        # Создаем массив нулей того же размера, что и входные данные скейлера
        dummy = np.zeros((scaled_data.shape[0], self.scaler.scale_.shape[0]))

        # Заполняем нужные колонки масштабированными данными
        for i, col_idx in enumerate(column_indices):
            dummy[:, col_idx] = scaled_data[:, i]

        # Выполняем обратное масштабирование
        inverse_scaled = self.scaler.inverse_transform(dummy)

        # Возвращаем только интересующие нас колонки
        return inverse_scaled[:, column_indices]


class FeatureGenerator:
    """
    Класс для генерации признаков для моделей.
    """

    def __init__(self):
        """
        Инициализирует генератор признаков.
        """
        logger.info("Initialized FeatureGenerator")

    def add_price_features(
            self,
            df: pd.DataFrame,
            ohlcv_columns: List[str] = ['open', 'high', 'low', 'close', 'volume']
    ) -> pd.DataFrame:
        """
        Добавляет признаки на основе цен.

        Args:
            df: DataFrame с данными
            ohlcv_columns: Список колонок OHLCV

        Returns:
            DataFrame с добавленными признаками
        """
        data = df.copy()

        # Проверяем наличие необходимых колонок
        for col in ohlcv_columns:
            if col not in data.columns:
                logger.warning(f"Колонка {col} отсутствует в данных")
                return data

        # Извлекаем колонки
        open_col, high_col, low_col, close_col, volume_col = ohlcv_columns

        # Добавляем признаки на основе цен
        # Ценовой диапазон
        data['price_range'] = data[high_col] - data[low_col]

        # Отношение открытия и закрытия
        data['open_close_ratio'] = data[open_col] / data[close_col]

        # Отношение максимума и минимума
        data['high_low_ratio'] = data[high_col] / data[low_col]

        # Ценовая амплитуда (%)
        data['price_amplitude'] = (data[high_col] - data[low_col]) / data[open_col] * 100

        # Логарифмические возвраты
        data['log_return'] = np.log(data[close_col] / data[close_col].shift(1))

        # Процентное изменение
        data['pct_change'] = data[close_col].pct_change()

        # Относительная сила объема
        data['volume_strength'] = data[volume_col] * data['pct_change']

        logger.info(f"Added {len(data.columns) - len(df.columns)} price-based features")

        return data

    def add_momentum_features(
            self,
            df: pd.DataFrame,
            price_column: str = 'close',
            windows: List[int] = [5, 10, 20, 50, 100]
    ) -> pd.DataFrame:
        """
        Добавляет признаки импульса (моментума).

        Args:
            df: DataFrame с данными
            price_column: Колонка с ценой
            windows: Список размеров окон для расчета признаков

        Returns:
            DataFrame с добавленными признаками
        """
        data = df.copy()

        # Проверяем наличие колонки цены
        if price_column not in data.columns:
            logger.warning(f"Колонка {price_column} отсутствует в данных")
            return data

        # Относительное положение цены (RSI)
        for window in windows:
            # Скользящие средние
            data[f'sma_{window}'] = data[price_column].rolling(window=window).mean()

            # Экспоненциальные скользящие средние
            data[f'ema_{window}'] = data[price_column].ewm(span=window, adjust=False).mean()

            # Расстояние от текущей цены до скользящей средней
            data[f'dist_to_sma_{window}'] = data[price_column] / data[f'sma_{window}'] - 1
            data[f'dist_to_ema_{window}'] = data[price_column] / data[f'ema_{window}'] - 1

            # Моментум (процентное изменение за период)
            data[f'momentum_{window}'] = data[price_column].pct_change(periods=window)

            # Скорость изменения
            data[f'roc_{window}'] = (data[price_column] - data[price_column].shift(window)) / data[price_column].shift(
                window) * 100

        logger.info(f"Added {len(data.columns) - len(df.columns)} momentum features")

        return data

    def add_volatility_features(
            self,
            df: pd.DataFrame,
            price_column: str = 'close',
            windows: List[int] = [5, 10, 20, 50, 100]
    ) -> pd.DataFrame:
        """
        Добавляет признаки волатильности.

        Args:
            df: DataFrame с данными
            price_column: Колонка с ценой
            windows: Список размеров окон для расчета признаков

        Returns:
            DataFrame с добавленными признаками
        """
        data = df.copy()

        # Проверяем наличие колонки цены
        if price_column not in data.columns:
            logger.warning(f"Колонка {price_column} отсутствует в данных")
            return data

        # Создаем логарифмические возвраты, если их еще нет
        if 'log_return' not in data.columns:
            data['log_return'] = np.log(data[price_column] / data[price_column].shift(1))

        # Добавляем признаки волатильности для разных окон
        for window in windows:
            # Стандартное отклонение доходности
            data[f'volatility_{window}'] = data['log_return'].rolling(window=window).std() * np.sqrt(window)

            # Верхняя полоса Боллинджера
            data[f'bollinger_high_{window}'] = data[price_column].rolling(window=window).mean() + 2 * data[
                price_column].rolling(window=window).std()

            # Нижняя полоса Боллинджера
            data[f'bollinger_low_{window}'] = data[price_column].rolling(window=window).mean() - 2 * data[
                price_column].rolling(window=window).std()

            # Ширина полос Боллинджера
            data[f'bollinger_width_{window}'] = (data[f'bollinger_high_{window}'] - data[f'bollinger_low_{window}']) / \
                                                data[price_column].rolling(window=window).mean()

            # Процентный диапазон (сравнение текущего диапазона цен с историческим)
            if 'high' in data.columns and 'low' in data.columns:
                data[f'historical_range_{window}'] = (data['high'] - data['low']) / data['low'].rolling(
                    window=window).mean()

        logger.info(f"Added {len(data.columns) - len(df.columns)} volatility features")

        return data

    def add_pattern_features(
            self,
            df: pd.DataFrame,
            ohlc_columns: List[str] = ['open', 'high', 'low', 'close']
    ) -> pd.DataFrame:
        """
        Добавляет признаки свечных паттернов.
        Требует установленной библиотеки TA-Lib.

        Args:
            df: DataFrame с данными
            ohlc_columns: Список колонок OHLC

        Returns:
            DataFrame с добавленными признаками
        """
        data = df.copy()

        # Проверяем наличие необходимых колонок
        for col in ohlc_columns:
            if col not in data.columns:
                logger.warning(f"Колонка {col} отсутствует в данных")
                return data

        try:
            import talib

            # Извлекаем массивы OHLC для TA-Lib
            open_arr = data[ohlc_columns[0]].values
            high_arr = data[ohlc_columns[1]].values
            low_arr = data[ohlc_columns[2]].values
            close_arr = data[ohlc_columns[3]].values

            # Добавляем признаки свечных паттернов
            # Доджи
            data['cdl_doji'] = talib.CDLDOJI(open_arr, high_arr, low_arr, close_arr)

            # Молот
            data['cdl_hammer'] = talib.CDLHAMMER(open_arr, high_arr, low_arr, close_arr)

            # Бычье поглощение
            data['cdl_engulfing_bull'] = talib.CDLENGULFING(open_arr, high_arr, low_arr, close_arr)

            # Медвежье поглощение
            data.loc[data['cdl_engulfing_bull'] < 0, 'cdl_engulfing_bear'] = 1
            data.loc[data['cdl_engulfing_bull'] > 0, 'cdl_engulfing_bear'] = 0
            data['cdl_engulfing_bear'] = data['cdl_engulfing_bear'].fillna(0)

            # Утренняя звезда (сильный бычий паттерн)
            data['cdl_morning_star'] = talib.CDLMORNINGSTAR(open_arr, high_arr, low_arr, close_arr)

            # Вечерняя звезда (сильный медвежий паттерн)
            data['cdl_evening_star'] = talib.CDLEVENINGSTAR(open_arr, high_arr, low_arr, close_arr)

            # Три белых солдата (сильный бычий паттерн)
            data['cdl_3_white_soldiers'] = talib.CDL3WHITESOLDIERS(open_arr, high_arr, low_arr, close_arr)

            # Три черные вороны (сильный медвежий паттерн)
            data['cdl_3_black_crows'] = talib.CDL3BLACKCROWS(open_arr, high_arr, low_arr, close_arr)

            logger.info(f"Added {len(data.columns) - len(df.columns)} candlestick pattern features")

        except ImportError:
            logger.warning("TA-Lib не установлен, пропуск создания признаков свечных паттернов")

        return data

    def add_technical_indicators(
            self,
            df: pd.DataFrame,
            price_column: str = 'close',
            volume_column: str = 'volume',
            ohlc_columns: List[str] = ['open', 'high', 'low', 'close']
    ) -> pd.DataFrame:
        """
        Добавляет технические индикаторы.
        Требует установленной библиотеки TA-Lib.

        Args:
            df: DataFrame с данными
            price_column: Колонка с ценой закрытия
            volume_column: Колонка с объемом
            ohlc_columns: Список колонок OHLC

        Returns:
            DataFrame с добавленными индикаторами
        """
        data = df.copy()

        # Проверяем наличие необходимых колонок
        if price_column not in data.columns:
            logger.warning(f"Колонка {price_column} отсутствует в данных")
            return data

        if volume_column not in data.columns:
            logger.warning(f"Колонка {volume_column} отсутствует в данных")
            volume_available = False
        else:
            volume_available = True

        for col in ohlc_columns:
            if col not in data.columns:
                logger.warning(f"Колонка {col} отсутствует в данных")
                ohlc_available = False
                break
        else:
            ohlc_available = True

        try:
            import talib

            # Цена закрытия
            close_arr = data[price_column].values

            # RSI - индекс относительной силы
            data['rsi_14'] = talib.RSI(close_arr, timeperiod=14)

            # MACD - схождение-расхождение скользящих средних
            macd, macd_signal, macd_hist = talib.MACD(
                close_arr, fastperiod=12, slowperiod=26, signalperiod=9
            )
            data['macd'] = macd
            data['macd_signal'] = macd_signal
            data['macd_hist'] = macd_hist

            # Stochastic Oscillator - стохастический осциллятор
            if ohlc_available:
                high_arr = data[ohlc_columns[1]].values
                low_arr = data[ohlc_columns[2]].values

                slowk, slowd = talib.STOCH(
                    high_arr, low_arr, close_arr,
                    fastk_period=5, slowk_period=3, slowk_matype=0,
                    slowd_period=3, slowd_matype=0
                )
                data['stoch_k'] = slowk
                data['stoch_d'] = slowd

            # ADX - индекс направленного движения
            if ohlc_available:
                open_arr = data[ohlc_columns[0]].values
                data['adx'] = talib.ADX(high_arr, low_arr, close_arr, timeperiod=14)

            # OBV - балансовый объем
            if volume_available:
                volume_arr = data[volume_column].values
                data['obv'] = talib.OBV(close_arr, volume_arr)

            # CCI - индекс товарного канала
            if ohlc_available:
                data['cci'] = talib.CCI(high_arr, low_arr, close_arr, timeperiod=14)

            # ATR - средний истинный диапазон
            if ohlc_available:
                data['atr'] = talib.ATR(high_arr, low_arr, close_arr, timeperiod=14)

            # Williams %R
            if ohlc_available:
                data['willr'] = talib.WILLR(high_arr, low_arr, close_arr, timeperiod=14)

            # Chaikin A/D Line - линия накопления/распределения Чайкина
            if ohlc_available and volume_available:
                data['ad_line'] = talib.AD(high_arr, low_arr, close_arr, volume_arr)

            # Chaikin Money Flow - денежный поток Чайкина
            if ohlc_available and volume_available:
                data['mfi'] = talib.MFI(high_arr, low_arr, close_arr, volume_arr, timeperiod=14)

            logger.info(f"Added {len(data.columns) - len(df.columns)} technical indicators")

        except ImportError:
            logger.warning("TA-Lib не установлен, пропуск создания технических индикаторов")

        return data

    def create_all_features(
            self,
            df: pd.DataFrame,
            ohlcv_columns: List[str] = ['open', 'high', 'low', 'close', 'volume'],
            windows: List[int] = [5, 10, 20, 50, 100]
    ) -> pd.DataFrame:
        """
        Создает все возможные признаки.

        Args:
            df: DataFrame с данными
            ohlcv_columns: Список колонок OHLCV
            windows: Список размеров окон для расчета признаков

        Returns:
            DataFrame со всеми признаками
        """
        data = df.copy()

        # Добавляем все типы признаков
        data = self.add_price_features(data, ohlcv_columns=ohlcv_columns)
        data = self.add_momentum_features(data, price_column='close', windows=windows)
        data = self.add_volatility_features(data, price_column='close', windows=windows)
        data = self.add_pattern_features(data, ohlc_columns=ohlcv_columns[:4])
        data = self.add_technical_indicators(
            data,
            price_column='close',
            volume_column='volume',
            ohlc_columns=ohlcv_columns[:4]
        )

        # Удаляем строки с NaN
        data = data.dropna()

        logger.info(f"Created a total of {len(data.columns) - len(df.columns)} features")

        return data