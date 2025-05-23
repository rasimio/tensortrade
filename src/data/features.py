"""
Модуль для создания и анализа технических индикаторов.
"""
import logging
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TechnicalAnalysis:
    """
    Класс для расчета и анализа технических индикаторов.
    """

    def __init__(self):
        """
        Инициализирует объект технического анализа.
        """
        logger.info("Initialized TechnicalAnalysis")

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

        except ImportError:
            logger.warning("TA-Lib не установлен, пропуск создания технических индикаторов")

        return data

    def generate_trading_signals(
            self,
            df: pd.DataFrame,
            strategy: str = 'combined',
            signal_threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Генерирует торговые сигналы на основе стратегии.

        Args:
            df: DataFrame с данными и техническими индикаторами
            strategy: Стратегия для генерации сигналов ('combined', 'trend_following', 'mean_reversion', 'volatility_breakout')
            signal_threshold: Порог для генерации сигнала

        Returns:
            DataFrame с добавленными сигналами
        """
        data = df.copy()

        # Проверяем наличие необходимых индикаторов
        required_indicators = {
            'trend_following': ['sma_20', 'ema_20', 'macd', 'macd_signal', 'adx'],
            'mean_reversion': ['rsi_14', 'bollinger_high_20', 'bollinger_low_20', 'stoch_k', 'stoch_d'],
            'volatility_breakout': ['atr', 'bollinger_width_20', 'volatility_20'],
            'combined': ['macd', 'macd_signal', 'rsi_14', 'adx', 'bollinger_width_20']
        }

        # Комбинированная стратегия использует все индикаторы
        if strategy == 'combined':
            # Проверяем наличие всех необходимых индикаторов для комбинированной стратегии
            for indicator in required_indicators['combined']:
                if indicator not in data.columns:
                    logger.warning(f"Индикатор {indicator} отсутствует для комбинированной стратегии")
                    return data

            # Инициализируем столбец сигналов
            data['signal'] = 0.0

            # Трендовые сигналы
            if 'macd' in data.columns and 'macd_signal' in data.columns:
                # MACD пересекает сигнальную линию снизу вверх - бычий сигнал
                data.loc[
                    (data['macd'] > data['macd_signal']) &
                    (data['macd'].shift(1) <= data['macd_signal'].shift(1)),
                    'signal'
                ] += 0.2

                # MACD пересекает сигнальную линию сверху вниз - медвежий сигнал
                data.loc[
                    (data['macd'] < data['macd_signal']) &
                    (data['macd'].shift(1) >= data['macd_signal'].shift(1)),
                    'signal'
                ] -= 0.2

            # Сигналы перекупленности/перепроданности
            if 'rsi_14' in data.columns:
                # RSI ниже 30 - перепроданность, потенциальный разворот вверх
                data.loc[data['rsi_14'] < 30, 'signal'] += 0.15

                # RSI выше 70 - перекупленность, потенциальный разворот вниз
                data.loc[data['rsi_14'] > 70, 'signal'] -= 0.15

            # Сила тренда
            if 'adx' in data.columns:
                # Сильный тренд усиливает сигналы
                data.loc[data['adx'] > 25, 'signal'] = data.loc[data['adx'] > 25, 'signal'] * 1.2

            # Волатильность
            if 'bollinger_width_20' in data.columns:
                # Сужение полос Боллинджера часто предшествует сильному движению
                data.loc[
                    data['bollinger_width_20'] < data['bollinger_width_20'].rolling(window=10).mean() * 0.8,
                    'signal'
                ] = data.loc[
                        data['bollinger_width_20'] < data['bollinger_width_20'].rolling(window=10).mean() * 0.8,
                        'signal'
                    ] * 1.1

            # Свечные паттерны
            for pattern in ['cdl_hammer', 'cdl_morning_star', 'cdl_3_white_soldiers']:
                if pattern in data.columns:
                    # Бычьи паттерны
                    data.loc[data[pattern] > 0, 'signal'] += 0.1

            for pattern in ['cdl_evening_star', 'cdl_3_black_crows']:
                if pattern in data.columns:
                    # Медвежьи паттерны
                    data.loc[data[pattern] > 0, 'signal'] -= 0.1

            # Нормализация сигналов
            data['signal'] = data['signal'].clip(-1, 1)

        elif strategy == 'trend_following':
            # Проверяем наличие необходимых индикаторов для трендовой стратегии
            for indicator in required_indicators['trend_following']:
                if indicator not in data.columns:
                    logger.warning(f"Индикатор {indicator} отсутствует для трендовой стратегии")
                    return data

            # Инициализируем столбец сигналов
            data['signal'] = 0.0

            # Цена выше SMA - восходящий тренд
            data.loc[data['close'] > data['sma_20'], 'signal'] += 0.15

            # Цена ниже SMA - нисходящий тренд
            data.loc[data['close'] < data['sma_20'], 'signal'] -= 0.15

            # Цена пересекает EMA снизу вверх - сигнал на покупку
            data.loc[
                (data['close'] > data['ema_20']) &
                (data['close'].shift(1) <= data['ema_20'].shift(1)),
                'signal'
            ] += 0.25

            # Цена пересекает EMA сверху вниз - сигнал на продажу
            data.loc[
                (data['close'] < data['ema_20']) &
                (data['close'].shift(1) >= data['ema_20'].shift(1)),
                'signal'
            ] -= 0.25

            # MACD пересекает сигнальную линию снизу вверх - бычий сигнал
            data.loc[
                (data['macd'] > data['macd_signal']) &
                (data['macd'].shift(1) <= data['macd_signal'].shift(1)),
                'signal'
            ] += 0.3

            # MACD пересекает сигнальную линию сверху вниз - медвежий сигнал
            data.loc[
                (data['macd'] < data['macd_signal']) &
                (data['macd'].shift(1) >= data['macd_signal'].shift(1)),
                'signal'
            ] -= 0.3

            # Сильный тренд (ADX > 25) усиливает сигналы
            data.loc[data['adx'] > 25, 'signal'] = data.loc[data['adx'] > 25, 'signal'] * 1.2

            # Нормализация сигналов
            data['signal'] = data['signal'].clip(-1, 1)

        elif strategy == 'mean_reversion':
            # Проверяем наличие необходимых индикаторов для стратегии возврата к среднему
            for indicator in required_indicators['mean_reversion']:
                if indicator not in data.columns:
                    logger.warning(f"Индикатор {indicator} отсутствует для стратегии возврата к среднему")
                    return data

            # Инициализируем столбец сигналов
            data['signal'] = 0.0

            # RSI ниже 30 - перепроданность, сигнал на покупку
            data.loc[data['rsi_14'] < 30, 'signal'] += 0.3

            # RSI выше 70 - перекупленность, сигнал на продажу
            data.loc[data['rsi_14'] > 70, 'signal'] -= 0.3

            # Цена ниже нижней полосы Боллинджера - потенциальный отскок вверх
            data.loc[data['close'] < data['bollinger_low_20'], 'signal'] += 0.25

            # Цена выше верхней полосы Боллинджера - потенциальный отскок вниз
            data.loc[data['close'] > data['bollinger_high_20'], 'signal'] -= 0.25

            # Стохастический осциллятор показывает перепроданность
            data.loc[(data['stoch_k'] < 20) & (data['stoch_d'] < 20), 'signal'] += 0.2

            # Стохастический осциллятор показывает перекупленность
            data.loc[(data['stoch_k'] > 80) & (data['stoch_d'] > 80), 'signal'] -= 0.2

            # Стохастический осциллятор пересекает снизу вверх в зоне перепроданности
            data.loc[
                (data['stoch_k'] > data['stoch_d']) &
                (data['stoch_k'].shift(1) <= data['stoch_d'].shift(1)) &
                (data['stoch_k'] < 30),
                'signal'
            ] += 0.25

            # Стохастический осциллятор пересекает сверху вниз в зоне перекупленности
            data.loc[
                (data['stoch_k'] < data['stoch_d']) &
                (data['stoch_k'].shift(1) >= data['stoch_d'].shift(1)) &
                (data['stoch_k'] > 70),
                'signal'
            ] -= 0.25

            # Нормализация сигналов
            data['signal'] = data['signal'].clip(-1, 1)

        elif strategy == 'volatility_breakout':
            # Проверяем наличие необходимых индикаторов для стратегии прорыва волатильности
            for indicator in required_indicators['volatility_breakout']:
                if indicator not in data.columns:
                    logger.warning(f"Индикатор {indicator} отсутствует для стратегии прорыва волатильности")
                    return data

            # Инициализируем столбец сигналов
            data['signal'] = 0.0

            # Сужение полос Боллинджера часто предшествует сильному движению
            data.loc[
                data['bollinger_width_20'] < data['bollinger_width_20'].rolling(window=10).mean() * 0.7,
                'signal_strength'
            ] = 0.2

            # Определяем направление прорыва
            if 'sma_20' in data.columns:
                # Цена выше SMA - потенциальный прорыв вверх
                data.loc[data['close'] > data['sma_20'], 'signal'] += data['signal_strength']

                # Цена ниже SMA - потенциальный прорыв вниз
                data.loc[data['close'] < data['sma_20'], 'signal'] -= data['signal_strength']

            # Увеличение волатильности указывает на потенциальный прорыв
            data.loc[
                data['volatility_20'] > data['volatility_20'].rolling(window=10).mean() * 1.3,
                'signal'
            ] = data.loc[
                    data['volatility_20'] > data['volatility_20'].rolling(window=10).mean() * 1.3,
                    'signal'
                ] * 1.5

            # Если ATR растет, это усиливает сигнал
            data.loc[
                data['atr'] > data['atr'].shift(1) * 1.1,
                'signal'
            ] = data.loc[
                    data['atr'] > data['atr'].shift(1) * 1.1,
                    'signal'
                ] * 1.2

            # Нормализация сигналов
            data['signal'] = data['signal'].clip(-1, 1)

            # Удаляем временный столбец
            if 'signal_strength' in data.columns:
                data.drop('signal_strength', axis=1, inplace=True)

        else:
            logger.warning(f"Неизвестная стратегия: {strategy}")
            return data

        # Генерация дискретных сигналов на основе порога
        data['buy_signal'] = (data['signal'] > signal_threshold).astype(int)
        data['sell_signal'] = (data['signal'] < -signal_threshold).astype(int)
        data['hold_signal'] = ((data['signal'] >= -signal_threshold) &
                               (data['signal'] <= signal_threshold)).astype(int)

        logger.info(f"Generated trading signals using {strategy} strategy")

        return data

    def calculate_optimal_position_size(
            self,
            df: pd.DataFrame,
            risk_per_trade: float = 0.02,
            atr_multiplier: float = 2.0,
            max_position_size: float = 0.25,
            volatility_adjustment: bool = True
    ) -> pd.DataFrame:
        """
        Рассчитывает оптимальный размер позиции на основе риска.

        Args:
            df: DataFrame с торговыми сигналами
            risk_per_trade: Процент от капитала, который можно рисковать на одной сделке (0.02 = 2%)
            atr_multiplier: Множитель для ATR при установке стоп-лосса
            max_position_size: Максимальный размер позиции как процент от капитала (0.25 = 25%)
            volatility_adjustment: Учитывать ли волатильность при расчете размера позиции

        Returns:
            DataFrame с добавленными размерами позиций
        """
        data = df.copy()

        # Проверяем наличие необходимых столбцов
        if 'atr' not in data.columns:
            logger.warning("ATR отсутствует в данных, расчет размера позиции будет менее точным")
            data['atr'] = (data['high'] - data['low']).rolling(window=14).mean()

        # Инициализируем столбец размера позиции
        data['position_size'] = 0.0

        # Рассчитываем размер стоп-лосса на основе ATR
        data['stop_loss_size'] = data['atr'] * atr_multiplier

        # Учитываем силу сигнала (если есть)
        if 'signal' in data.columns:
            signal_strength = data['signal'].abs()
        else:
            signal_strength = 1.0

        # Учитываем волатильность (если запрошено)
        if volatility_adjustment and 'volatility_20' in data.columns:
            # Нормализуем волатильность относительно среднего значения
            volatility_ratio = data['volatility_20'] / data['volatility_20'].rolling(window=100).mean()
            # Обратная зависимость: выше волатильность - меньше размер позиции
            volatility_factor = 1.0 / volatility_ratio
            # Ограничиваем фактор волатильности
            volatility_factor = volatility_factor.clip(0.5, 2.0)
        else:
            volatility_factor = 1.0

        # Рассчитываем размер позиции
        # Формула: (Капитал * Риск на сделку) / Размер стоп-лосса * Поправка на волатильность * Сила сигнала
        data['position_size'] = risk_per_trade / (data['stop_loss_size'] / data['close'])

        # Применяем корректировки
        if isinstance(volatility_factor, pd.Series):
            data['position_size'] = data['position_size'] * volatility_factor

        if isinstance(signal_strength, pd.Series):
            data['position_size'] = data['position_size'] * signal_strength

        # Ограничиваем максимальный размер позиции
        data['position_size'] = data['position_size'].clip(0, max_position_size)

        # Устанавливаем размер позиции в соответствии с сигналами
        data.loc[data['buy_signal'] == 0, 'position_size'] = 0
        data.loc[data['sell_signal'] == 1, 'position_size'] = -data.loc[data['sell_signal'] == 1, 'position_size']

        logger.info(f"Calculated position sizes with risk {risk_per_trade * 100}% per trade")

        return data

    def evaluate_signal_quality(
            self,
            df: pd.DataFrame,
            lookahead_periods: int = 5,
            profit_target_multiplier: float = 2.0,
            stop_loss_multiplier: float = 1.0
    ) -> pd.DataFrame:
        """
        Оценивает качество торговых сигналов путем анализа будущих цен.

        Args:
            df: DataFrame с торговыми сигналами
            lookahead_periods: Количество периодов для анализа в будущее
            profit_target_multiplier: Множитель для целевой прибыли относительно стоп-лосса
            stop_loss_multiplier: Множитель для стоп-лосса

        Returns:
            DataFrame с добавленными оценками качества сигналов
        """
        data = df.copy()

        # Проверяем наличие необходимых столбцов
        if 'buy_signal' not in data.columns or 'sell_signal' not in data.columns:
            logger.warning("Сигналы покупки/продажи отсутствуют в данных")
            return data

        if 'atr' not in data.columns:
            logger.warning("ATR отсутствует в данных, используем простой расчет стоп-лосса")
            data['atr'] = (data['high'] - data['low']).rolling(window=14).mean()

        # Инициализируем столбцы для оценки
        data['signal_outcome'] = 0  # -1 = убыток, 0 = без результата, 1 = прибыль
        data['profit_loss_ratio'] = 0.0

        # Для каждой точки данных с сигналом покупки
        for i in range(len(data)):
            if data.iloc[i]['buy_signal'] == 1:
                # Цена входа
                entry_price = data.iloc[i]['close']

                # Размер стоп-лосса
                stop_loss_size = data.iloc[i]['atr'] * stop_loss_multiplier

                # Уровни стоп-лосса и тейк-профита
                stop_loss = entry_price - stop_loss_size
                take_profit = entry_price + (stop_loss_size * profit_target_multiplier)

                # Анализируем будущие цены
                if i + lookahead_periods < len(data):
                    future_slice = data.iloc[i + 1:i + lookahead_periods + 1]

                    # Проверяем, достигли ли цены уровня стоп-лосса или тейк-профита
                    hit_stop_loss = (future_slice['low'] <= stop_loss).any()
                    hit_take_profit = (future_slice['high'] >= take_profit).any()

                    # Определяем, что произошло раньше
                    if hit_stop_loss and hit_take_profit:
                        stop_loss_index = future_slice[future_slice['low'] <= stop_loss].index.min()
                        take_profit_index = future_slice[future_slice['high'] >= take_profit].index.min()

                        if stop_loss_index < take_profit_index:
                            data.at[data.index[i], 'signal_outcome'] = -1
                            data.at[data.index[i], 'profit_loss_ratio'] = -1.0
                        else:
                            data.at[data.index[i], 'signal_outcome'] = 1
                            data.at[data.index[i], 'profit_loss_ratio'] = profit_target_multiplier
                    elif hit_stop_loss:
                        data.at[data.index[i], 'signal_outcome'] = -1
                        data.at[data.index[i], 'profit_loss_ratio'] = -1.0
                    elif hit_take_profit:
                        data.at[data.index[i], 'signal_outcome'] = 1
                        data.at[data.index[i], 'profit_loss_ratio'] = profit_target_multiplier
                    else:
                        # Если ни один уровень не достигнут, используем конечную цену
                        final_price = future_slice.iloc[-1]['close']
                        profit_loss = (final_price - entry_price) / stop_loss_size

                        if profit_loss > 0:
                            data.at[data.index[i], 'signal_outcome'] = 1
                        elif profit_loss < 0:
                            data.at[data.index[i], 'signal_outcome'] = -1

                        data.at[data.index[i], 'profit_loss_ratio'] = profit_loss

            # Для каждой точки данных с сигналом продажи
            elif data.iloc[i]['sell_signal'] == 1:
                # Цена входа
                entry_price = data.iloc[i]['close']

                # Размер стоп-лосса
                stop_loss_size = data.iloc[i]['atr'] * stop_loss_multiplier

                # Уровни стоп-лосса и тейк-профита (обратные для короткой позиции)
                stop_loss = entry_price + stop_loss_size
                take_profit = entry_price - (stop_loss_size * profit_target_multiplier)

                # Анализируем будущие цены
                if i + lookahead_periods < len(data):
                    future_slice = data.iloc[i + 1:i + lookahead_periods + 1]

                    # Проверяем, достигли ли цены уровня стоп-лосса или тейк-профита
                    hit_stop_loss = (future_slice['high'] >= stop_loss).any()
                    hit_take_profit = (future_slice['low'] <= take_profit).any()

                    # Определяем, что произошло раньше
                    if hit_stop_loss and hit_take_profit:
                        stop_loss_index = future_slice[future_slice['high'] >= stop_loss].index.min()
                        take_profit_index = future_slice[future_slice['low'] <= take_profit].index.min()

                        if stop_loss_index < take_profit_index:
                            data.at[data.index[i], 'signal_outcome'] = -1
                            data.at[data.index[i], 'profit_loss_ratio'] = -1.0
                        else:
                            data.at[data.index[i], 'signal_outcome'] = 1
                            data.at[data.index[i], 'profit_loss_ratio'] = profit_target_multiplier
                    elif hit_stop_loss:
                        data.at[data.index[i], 'signal_outcome'] = -1
                        data.at[data.index[i], 'profit_loss_ratio'] = -1.0
                    elif hit_take_profit:
                        data.at[data.index[i], 'signal_outcome'] = 1
                        data.at[data.index[i], 'profit_loss_ratio'] = profit_target_multiplier
                    else:
                        # Если ни один уровень не достигнут, используем конечную цену
                        final_price = future_slice.iloc[-1]['close']
                        profit_loss = (entry_price - final_price) / stop_loss_size

                        if profit_loss > 0:
                            data.at[data.index[i], 'signal_outcome'] = 1
                        elif profit_loss < 0:
                            data.at[data.index[i], 'signal_outcome'] = -1

                        data.at[data.index[i], 'profit_loss_ratio'] = profit_loss

        # Рассчитываем общую статистику для сигналов
        total_signals = data['buy_signal'].sum() + data['sell_signal'].sum()
        winning_signals = (data['signal_outcome'] == 1).sum()
        losing_signals = (data['signal_outcome'] == -1).sum()

        if total_signals > 0:
            win_rate = winning_signals / total_signals

            # Средний размер выигрыша и проигрыша
            avg_win = data.loc[data['signal_outcome'] == 1, 'profit_loss_ratio'].mean()
            avg_loss = data.loc[data['signal_outcome'] == -1, 'profit_loss_ratio'].mean()

            # Отношение среднего выигрыша к среднему проигрышу
            if losing_signals > 0 and avg_loss != 0:
                win_loss_ratio = abs(avg_win / avg_loss)
            else:
                win_loss_ratio = float('inf')

            # Ожидаемая прибыль на сделку
            expected_profit_per_trade = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

            logger.info(
                f"Signal quality: Win rate = {win_rate:.2f}, Win/Loss ratio = {win_loss_ratio:.2f}, Exp. profit per trade = {expected_profit_per_trade:.2f}")

        return data