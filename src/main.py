"""
Основной модуль для запуска приложения TensorTrade.
"""
import os
import logging
import argparse
import pandas as pd
import numpy as np
import asyncio
import json
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from src.data.collectors import BinanceDataCollector
from src.data.processors import DataProcessor, FeatureGenerator
from src.models.base_model import BaseModel, ModelFactory
from src.models.lstm import LSTMModel
from src.models.reinforcement import RLModel
from src.models.technical import TechnicalModel
from src.models.ensemble import EnsembleModel
from src.models.hybrid import HybridModel
from src.backtest.engine import BacktestEngine
from src.backtest.reporting import BacktestReport
from src.simulation.server import WSServer
from src.simulation.market import MarketSimulator
from src.simulation.trader import TradingSimulator, LiveTrader

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/tensortrade.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class TensorTrade:
    """
    Основной класс для управления приложением TensorTrade.
    """

    def __init__(self, config_path: str = "config.json"):
        """
        Инициализирует приложение TensorTrade.

        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config_path = config_path

        # Загружаем конфигурацию
        self.config = self.load_config()

        # Создаем директории для хранения данных и моделей
        os.makedirs(self.config.get("data_dir", "data"), exist_ok=True)
        os.makedirs(self.config.get("models_dir", "models"), exist_ok=True)
        os.makedirs(self.config.get("output_dir", "output"), exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        # Инициализируем компоненты
        self.data_collector = None
        self.data_processor = None
        self.feature_generator = None
        self.model = None
        self.backtest_engine = None
        self.market_simulator = None
        self.trading_simulator = None
        self.live_trader = None
        self.wss_server = None

        # Данные
        self.data = None
        self.processed_data = None
        self.feature_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

        logger.info("Приложение TensorTrade инициализировано")

    def load_config(self) -> Dict:
        """
        Загружает конфигурацию из файла.

        Returns:
            Словарь с конфигурацией
        """
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)

            logger.info(f"Конфигурация загружена из {self.config_path}")

            return config

        except FileNotFoundError:
            logger.warning(f"Файл конфигурации {self.config_path} не найден, используются параметры по умолчанию")

            # Конфигурация по умолчанию
            default_config = {
                "api_key": "",
                "api_secret": "",
                "use_testnet": True,
                "symbol": "BTCUSDT",
                "interval": "1h",
                "data_start_date": "2023-01-01",
                "data_end_date": "2023-12-31",
                "data_dir": "data",
                "models_dir": "models",
                "output_dir": "output",
                "initial_balance": 10000.0,
                "fee": 0.001,
                "slippage": 0.0005,
                "position_size": 0.1,
                "stop_loss": 0.02,
                "take_profit": 0.05,
                "model_type": "hybrid",
                "model_params": {
                    "lstm_params": {
                        "lstm_units": 100,
                        "dense_units": 64,
                        "dropout_rate": 0.2,
                        "learning_rate": 0.001,
                        "batch_size": 32,
                        "epochs": 100,
                        "sequence_length": 60
                    },
                    "rl_params": {
                        "algorithm": "ppo",
                        "learning_rate": 0.0003,
                        "gamma": 0.99,
                        "batch_size": 64,
                        "env_params": {
                            "reward_scaling": 1.0
                        }
                    },
                    "technical_params": {
                        "classifier": "rf",
                        "n_estimators": 100,
                        "max_depth": 10
                    },
                    "alpha": 0.3,
                    "beta": 0.4,
                    "gamma": 0.3
                }
            }

            # Сохраняем конфигурацию по умолчанию
            with open(self.config_path, "w") as f:
                json.dump(default_config, f, indent=4)

            logger.info(f"Создана конфигурация по умолчанию и сохранена в {self.config_path}")

            return default_config

        except json.JSONDecodeError:
            logger.error(f"Ошибка при чтении файла конфигурации {self.config_path}")
            return {}

    def collect_data(
            self,
            symbol: Optional[str] = None,
            interval: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Собирает данные с биржи Binance.

        Args:
            symbol: Символ торговой пары
            interval: Интервал свечей
            start_date: Начальная дата сбора данных
            end_date: Конечная дата сбора данных
            save_path: Путь для сохранения данных

        Returns:
            DataFrame с собранными данными
        """
        # Получаем параметры из конфигурации, если не переданы
        print("collecting..")
        symbol = symbol or self.config.get("symbol", "BTCUSDT")
        interval = interval or self.config.get("interval", "1h")
        start_date = start_date or self.config.get("data_start_date", "2023-01-01")
        end_date = end_date or self.config.get("data_end_date", datetime.now().strftime("%Y-%m-%d"))

        # Если путь для сохранения не задан, создаем его
        if save_path is None:
            data_dir = self.config.get("data_dir", "data")
            save_path = os.path.join(data_dir, f"{symbol}_{interval}_{start_date}_{end_date}.csv")

        # Инициализируем коллектор данных
        if self.data_collector is None:
            self.data_collector = BinanceDataCollector(
                api_key=self.config.get("api_key", ""),
                api_secret=self.config.get("api_secret", ""),
                testnet=self.config.get("use_testnet", True)
            )

        # Собираем данные
        logger.info(f"Сбор данных для {symbol} с интервалом {interval} с {start_date} по {end_date}")

        data = self.data_collector.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            save_path=save_path
        )

        # Сохраняем данные
        self.data = data

        logger.info(f"Собрано {len(data)} записей")

        return data

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Загружает данные из CSV-файла.

        Args:
            file_path: Путь к CSV-файлу

        Returns:
            DataFrame с загруженными данными
        """
        try:
            # Загружаем данные
            data = pd.read_csv(file_path)

            # Если есть колонка timestamp, преобразуем ее в datetime и устанавливаем как индекс
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)

            # Сохраняем данные
            self.data = data

            logger.info(f"Загружено {len(data)} записей из {file_path}")

            return data

        except Exception as e:
            logger.error(f"Ошибка при загрузке данных из {file_path}: {e}")
            return pd.DataFrame()

    def preprocess_data(
            self,
            data: Optional[pd.DataFrame] = None,
            scaling_method: str = 'minmax'
    ) -> pd.DataFrame:
        """
        Выполняет предобработку данных.

        Args:
            data: DataFrame с данными (если None, используется self.data)
            scaling_method: Метод масштабирования ('minmax' или 'standard')

        Returns:
            DataFrame с обработанными данными
        """
        # Если данные не переданы, используем загруженные
        data = data if data is not None else self.data

        if data is None or data.empty:
            logger.error("Нет данных для обработки")
            return pd.DataFrame()

        # Инициализируем процессор данных
        if self.data_processor is None:
            self.data_processor = DataProcessor(scaling_method=scaling_method)

        # Инициализируем генератор признаков
        if self.feature_generator is None:
            self.feature_generator = FeatureGenerator()

        # Создаем признаки
        logger.info("Создание признаков")

        feature_data = self.feature_generator.create_all_features(data)

        # Сохраняем данные с признаками
        self.feature_data = feature_data

        logger.info(f"Создано {len(feature_data.columns) - len(data.columns)} признаков")

        # Разделяем данные на обучающую, валидационную и тестовую выборки
        train_size = int(len(feature_data) * 0.7)
        val_size = int(len(feature_data) * 0.15)

        self.train_data = feature_data.iloc[:train_size]
        self.val_data = feature_data.iloc[train_size:train_size + val_size]
        self.test_data = feature_data.iloc[train_size + val_size:]

        logger.info(
            f"Разделение данных: обучающая - {len(self.train_data)}, валидационная - {len(self.val_data)}, тестовая - {len(self.test_data)}")

        return feature_data

    def build_model(
            self,
            model_type: Optional[str] = None,
            model_params: Optional[Dict] = None
    ) -> BaseModel:
        """
        Создает модель заданного типа.

        Args:
            model_type: Тип модели ('lstm', 'reinforcement', 'technical', 'ensemble', 'hybrid')
            model_params: Параметры модели

        Returns:
            Модель
        """
        # Получаем параметры из конфигурации, если не переданы
        model_type = model_type or self.config.get("model_type", "hybrid")
        model_params = model_params or self.config.get("model_params", {})

        # Создаем модель
        logger.info(f"Создание модели типа {model_type}")

        model = ModelFactory.create_model(
            model_type=model_type,
            name=f"{model_type}_model",
            model_params=model_params
        )

        # Сохраняем модель
        self.model = model

        return model

    def train_model(
            self,
            model: Optional[BaseModel] = None,
            train_data: Optional[pd.DataFrame] = None,
            val_data: Optional[pd.DataFrame] = None,
            feature_columns: Optional[List[str]] = None,
            target_column: Optional[str] = None,
            **kwargs
    ) -> Dict:
        """
        Обучает модель на данных.

        Args:
            model: Модель для обучения (если None, используется self.model)
            train_data: Данные для обучения (если None, используется self.train_data)
            val_data: Данные для валидации (если None, используется self.val_data)
            feature_columns: Список колонок с признаками
            target_column: Колонка с целевой переменной
            **kwargs: Дополнительные аргументы для обучения

        Returns:
            Словарь с результатами обучения
        """
        # Если модель не передана, используем созданную
        model = model if model is not None else self.model

        if model is None:
            logger.error("Модель не создана")
            return {}

        # Если данные не переданы, используем подготовленные
        train_data = train_data if train_data is not None else self.train_data
        val_data = val_data if val_data is not None else self.val_data

        if train_data is None or train_data.empty:
            logger.error("Нет данных для обучения")
            return {}

        # Если признаки не переданы, используем все числовые колонки, кроме целевой
        if feature_columns is None:
            # Исключаем целевую колонку из признаков
            exclude_columns = [target_column] if target_column else []

            # Также исключаем колонки с сигналами
            exclude_columns.extend(['signal', 'buy_signal', 'sell_signal', 'hold_signal'])

            # Выбираем все числовые колонки, кроме исключенных
            feature_columns = [
                col for col in train_data.select_dtypes(include=[np.number]).columns
                if col not in exclude_columns
            ]

        # Подготавливаем данные для обучения
        logger.info(f"Подготовка данных для обучения модели: {len(feature_columns)} признаков")

        # Определяем, какой тип модели у нас и подготавливаем данные соответственно
        if isinstance(model, LSTMModel):
            # Для LSTM нужны последовательности
            sequence_length = model.model_params.get("sequence_length", 60)

            # Создаем последовательности для обучения
            X_train, y_train = [], []

            if target_column is None:
                # Если целевая колонка не задана, используем будущую цену
                target_column = 'close'
                prediction_horizon = kwargs.get("prediction_horizon", 1)

                # Используем цены закрытия с задержкой как целевую переменную
                train_target = train_data[target_column].shift(-prediction_horizon).values[:-prediction_horizon]
                train_features = train_data[feature_columns].values[:-prediction_horizon]
            else:
                train_target = train_data[target_column].values
                train_features = train_data[feature_columns].values

            # Создаем последовательности
            for i in range(len(train_features) - sequence_length):
                X_train.append(train_features[i:i + sequence_length])
                y_train.append(train_target[i + sequence_length - 1])

            X_train = np.array(X_train)
            y_train = np.array(y_train)

            # Создаем последовательности для валидации, если есть валидационные данные
            X_val, y_val = None, None

            if val_data is not None and not val_data.empty:
                X_val, y_val = [], []

                if target_column == 'close' and 'prediction_horizon' in kwargs:
                    # Если используем будущую цену
                    prediction_horizon = kwargs["prediction_horizon"]
                    val_target = val_data[target_column].shift(-prediction_horizon).values[:-prediction_horizon]
                    val_features = val_data[feature_columns].values[:-prediction_horizon]
                else:
                    val_target = val_data[target_column].values
                    val_features = val_data[feature_columns].values

                # Создаем последовательности
                for i in range(len(val_features) - sequence_length):
                    X_val.append(val_features[i:i + sequence_length])
                    y_val.append(val_target[i + sequence_length - 1])

                X_val = np.array(X_val)
                y_val = np.array(y_val)

        elif isinstance(model, RLModel):
            # Для RL используем весь DataFrame
            X_train, y_train = None, None
            X_val, y_val = None, None

            # Устанавливаем имена признаков
            model.set_feature_names(feature_columns)

        elif isinstance(model, HybridModel):
            # Для гибридной модели подготавливаем данные для всех компонентов
            X_train, y_train = None, None
            X_val, y_val = None, None

            # Устанавливаем имена признаков
            model.set_feature_names(feature_columns)

        else:
            # Для остальных моделей используем обычный подход
            X_train = train_data[feature_columns].values

            if target_column is not None:
                y_train = train_data[target_column].values
            else:
                # Если целевая колонка не задана, используем сигналы
                if 'signal' in train_data.columns:
                    y_train = train_data['signal'].values
                else:
                    # Если нет сигналов, используем изменение цены как целевую переменную
                    train_data['price_change'] = np.where(
                        train_data['close'].shift(-1) > train_data['close'], 1,
                        np.where(train_data['close'].shift(-1) < train_data['close'], 2, 0)
                    )
                    y_train = train_data['price_change'].values[:-1]
                    X_train = X_train[:-1]

            # Валидационные данные
            X_val, y_val = None, None

            if val_data is not None and not val_data.empty:
                X_val = val_data[feature_columns].values

                if target_column is not None:
                    y_val = val_data[target_column].values
                else:
                    # Если целевая колонка не задана, используем сигналы
                    if 'signal' in val_data.columns:
                        y_val = val_data['signal'].values
                    else:
                        # Если нет сигналов, используем изменение цены как целевую переменную
                        val_data['price_change'] = np.where(
                            val_data['close'].shift(-1) > val_data['close'], 1,
                            np.where(val_data['close'].shift(-1) < val_data['close'], 2, 0)
                        )
                        y_val = val_data['price_change'].values[:-1]
                        X_val = X_val[:-1]

        # Обучаем модель
        logger.info(f"Обучение модели {model.name}")

        # Разные подходы к обучению в зависимости от типа модели
        if isinstance(model, RLModel):
            # Для RL передаем DataFrame
            training_results = model.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                data=train_data,
                feature_columns=feature_columns,
                **kwargs
            )

        elif isinstance(model, HybridModel):
            # Для гибридной модели передаем DataFrame
            training_results = model.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                data=train_data,
                feature_columns=feature_columns,
                **kwargs
            )

        else:
            # Для остальных моделей используем обычный подход
            training_results = model.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                **kwargs
            )

        # Сохраняем модель
        models_dir = self.config.get("models_dir", "models")
        model_path = os.path.join(models_dir, f"{model.name}.pkl")
        model.save(model_path)

        logger.info(f"Модель {model.name} сохранена в {model_path}")

        return training_results

    def backtest_model(
            self,
            model: Optional[BaseModel] = None,
            test_data: Optional[pd.DataFrame] = None,
            feature_columns: Optional[List[str]] = None,
            price_column: str = 'close',
            initial_balance: Optional[float] = None,
            fee: Optional[float] = None,
            slippage: Optional[float] = None,
            position_size: Optional[float] = None,
            stop_loss: Optional[float] = None,
            take_profit: Optional[float] = None,
            save_report: bool = True
    ) -> Dict:
        """
        Выполняет бэктестирование модели.

        Args:
            model: Модель для бэктестирования (если None, используется self.model)
            test_data: Данные для бэктестирования (если None, используется self.test_data)
            feature_columns: Список колонок с признаками
            price_column: Колонка с ценами
            initial_balance: Начальный баланс
            fee: Комиссия за сделку
            slippage: Проскальзывание
            position_size: Размер позиции
            stop_loss: Уровень стоп-лосса
            take_profit: Уровень тейк-профита
            save_report: Сохранять ли отчет

        Returns:
            Словарь с результатами бэктестирования
        """
        # Если модель не передана, используем созданную
        model = model if model is not None else self.model

        if model is None:
            logger.error("Модель не создана")
            return {}

        # Если данные не переданы, используем подготовленные
        test_data = test_data if test_data is not None else self.test_data

        if test_data is None or test_data.empty:
            logger.error("Нет данных для бэктестирования")
            return {}

        # Если признаки не переданы, используем сохраненные в модели
        feature_columns = feature_columns or model.feature_names

        if feature_columns is None:
            # Если признаки не сохранены в модели, используем все числовые колонки
            feature_columns = test_data.select_dtypes(include=[np.number]).columns.tolist()

            # Исключаем колонки с сигналами
            exclude_columns = ['signal', 'buy_signal', 'sell_signal', 'hold_signal']
            feature_columns = [col for col in feature_columns if col not in exclude_columns]

        # Получаем параметры из конфигурации, если не переданы
        initial_balance = initial_balance or self.config.get("initial_balance", 10000.0)
        fee = fee or self.config.get("fee", 0.001)
        slippage = slippage or self.config.get("slippage", 0.0005)
        position_size = position_size or self.config.get("position_size", 0.1)
        stop_loss = stop_loss or self.config.get("stop_loss", 0.02)
        take_profit = take_profit or self.config.get("take_profit", 0.05)

        # Создаем симулятор торговли
        trading_simulator = TradingSimulator(
            model=model,
            initial_balance=initial_balance,
            fee=fee,
            slippage=slippage,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            output_dir=self.config.get("output_dir", "output")
        )

        # Выполняем бэктестирование
        logger.info(f"Бэктестирование модели {model.name}")

        backtest_results = trading_simulator.run_backtest(
            data=test_data,
            price_column=price_column,
            strategy_name=model.name,
            save_report=save_report
        )

        # Сохраняем симулятор
        self.trading_simulator = trading_simulator

        # Выводим основные метрики
        metrics = backtest_results.get('metrics', {})

        logger.info(f"Результаты бэктестирования:")
        logger.info(f"Начальный баланс: {metrics.get('initial_balance', initial_balance):.2f}")
        logger.info(f"Конечный баланс: {metrics.get('final_balance', 0):.2f}")
        logger.info(f"Общая доходность: {metrics.get('total_return_pct', 0):.2f}%")
        logger.info(f"Годовая доходность: {metrics.get('annual_return_pct', 0):.2f}%")
        logger.info(f"Максимальная просадка: {metrics.get('max_drawdown_pct', 0):.2f}%")
        logger.info(f"Соотношение выигрышей: {metrics.get('win_rate_pct', 0):.2f}%")

        return backtest_results

    def run_simulation(
            self,
            model: Optional[BaseModel] = None,
            symbol: Optional[str] = None,
            interval: Optional[str] = None,
            duration: Optional[int] = None,
            initial_balance: Optional[float] = None,
            fee: Optional[float] = None,
            slippage: Optional[float] = None,
            position_size: Optional[float] = None,
            stop_loss: Optional[float] = None,
            take_profit: Optional[float] = None,
            use_websocket: bool = True,
            ws_host: str = 'localhost',
            ws_port: int = 8765
    ) -> None:
        """
        Запускает симуляцию торговли с использованием WebSocket сервера.

        Args:
            model: Модель для симуляции (если None, используется self.model)
            symbol: Символ торговой пары
            interval: Интервал свечей
            duration: Продолжительность симуляции в секундах (если None, выполняется до остановки)
            initial_balance: Начальный баланс
            fee: Комиссия за сделку
            slippage: Проскальзывание
            position_size: Размер позиции
            stop_loss: Уровень стоп-лосса
            take_profit: Уровень тейк-профита
            use_websocket: Использовать ли WebSocket
            ws_host: Хост для WebSocket сервера
            ws_port: Порт для WebSocket сервера
        """
        # Если модель не передана, используем созданную
        model = model if model is not None else self.model

        if model is None:
            logger.error("Модель не создана")
            return

        # Получаем параметры из конфигурации, если не переданы
        symbol = symbol or self.config.get("symbol", "BTCUSDT")
        interval = interval or self.config.get("interval", "1m")
        initial_balance = initial_balance or self.config.get("initial_balance", 10000.0)
        fee = fee or self.config.get("fee", 0.001)
        slippage = slippage or self.config.get("slippage", 0.0005)
        position_size = position_size or self.config.get("position_size", 0.1)
        stop_loss = stop_loss or self.config.get("stop_loss", 0.02)
        take_profit = take_profit or self.config.get("take_profit", 0.05)

        # Запускаем WebSocket сервер в отдельном потоке, если нужно
        if use_websocket:
            # Если есть исторические данные, используем их для симуляции
            historical_data = self.data if self.data is not None else None

            # Запускаем сервер
            self.wss_server = WSServer(
                host=ws_host,
                port=ws_port,
                symbol=symbol,
                interval=interval,
                historical_data=historical_data
            )

            # Запускаем сервер в отдельном потоке
            import threading
            server_thread = threading.Thread(target=self.wss_server.start)
            server_thread.daemon = True
            server_thread.start()

            # Ждем запуска сервера
            import time
            time.sleep(2)

            # URL для WebSocket клиента
            ws_url = f"ws://{ws_host}:{ws_port}"
        else:
            ws_url = None

        # Создаем симулятор торговли
        self.trading_simulator = TradingSimulator(
            model=model,
            initial_balance=initial_balance,
            fee=fee,
            slippage=slippage,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            use_websocket=use_websocket,
            ws_url=ws_url,
            ws_symbol=symbol,
            output_dir=self.config.get("output_dir", "output")
        )

        # Запускаем симуляцию
        logger.info(f"Запуск симуляции для модели {model.name}")

        # Запускаем симуляцию асинхронно
        if use_websocket:
            try:
                # Запускаем цикл событий
                loop = asyncio.get_event_loop()

                # Если цикл событий уже запущен, используем его
                if loop.is_running():
                    asyncio.create_task(self.trading_simulator.run_live_simulation(duration=duration))
                else:
                    loop.run_until_complete(self.trading_simulator.run_live_simulation(duration=duration))

            except Exception as e:
                logger.error(f"Ошибка при запуске симуляции: {e}")
        else:
            # Если данных нет, выходим
            if self.data is None:
                logger.error("Нет данных для симуляции")
                return

            # Запускаем симуляцию на исторических данных
            self.trading_simulator.run_simulation(data=self.data)

        logger.info("Симуляция завершена")

    def start_live_trading(
            self,
            model: Optional[BaseModel] = None,
            symbol: Optional[str] = None,
            interval: Optional[str] = None,
            initial_balance: Optional[float] = None,
            fee: Optional[float] = None,
            position_size: Optional[float] = None,
            stop_loss: Optional[float] = None,
            take_profit: Optional[float] = None,
            is_demo: bool = True
    ) -> None:
        """
        Запускает живую торговлю.

        Args:
            model: Модель для торговли (если None, используется self.model)
            symbol: Символ торговой пары
            interval: Интервал свечей
            initial_balance: Начальный баланс
            fee: Комиссия за сделку
            position_size: Размер позиции
            stop_loss: Уровень стоп-лосса
            take_profit: Уровень тейк-профита
            is_demo: Использовать ли демо-режим
        """
        # Если модель не передана, используем созданную
        model = model if model is not None else self.model

        if model is None:
            logger.error("Модель не создана")
            return

        # Получаем параметры из конфигурации, если не переданы
        symbol = symbol or self.config.get("symbol", "BTCUSDT")
        interval = interval or self.config.get("interval", "1m")
        initial_balance = initial_balance or self.config.get("initial_balance", 10000.0)
        fee = fee or self.config.get("fee", 0.001)
        position_size = position_size or self.config.get("position_size", 0.1)
        stop_loss = stop_loss or self.config.get("stop_loss", 0.02)
        take_profit = take_profit or self.config.get("take_profit", 0.05)

        # Создаем API биржи, если не используется демо-режим
        exchange_api = None

        if not is_demo:
            try:
                from binance.client import Client

                # Создаем клиент биржи
                api_key = self.config.get("api_key", "")
                api_secret = self.config.get("api_secret", "")
                use_testnet = self.config.get("use_testnet", True)

                if not api_key or not api_secret:
                    logger.error("API ключ и секрет не заданы в конфигурации")
                    return

                exchange_api = Client(api_key, api_secret, testnet=use_testnet)

                logger.info(f"Создан клиент биржи {'тестовой сети' if use_testnet else 'основной сети'}")

            except ImportError:
                logger.error("Не удалось импортировать модуль binance.client, используется демо-режим")
                is_demo = True

            except Exception as e:
                logger.error(f"Ошибка при создании клиента биржи: {e}")
                is_demo = True

        # Создаем трейдера
        self.live_trader = LiveTrader(
            model=model,
            exchange_api=exchange_api,
            symbol=symbol,
            interval=interval,
            initial_balance=initial_balance,
            fee=fee,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            is_demo=is_demo,
            use_websocket=True,
            ws_url=f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{interval}",
            output_dir=self.config.get("output_dir", "output")
        )

        # Запускаем торговлю
        logger.info(f"Запуск {'демо' if is_demo else 'реальной'} торговли для модели {model.name}")

        self.live_trader.start_trading()

    def stop_live_trading(self) -> None:
        """
        Останавливает живую торговлю.
        """
        if self.live_trader is None:
            logger.warning("Живая торговля не запущена")
            return

        # Останавливаем торговлю
        self.live_trader.stop_trading()

        # Сохраняем результаты
        results_dir = self.live_trader.save_trading_results()

        logger.info(f"Живая торговля остановлена, результаты сохранены в {results_dir}")

    def optimize_model(
            self,
            model_type: Optional[str] = None,
            param_grid: Optional[Dict] = None,
            train_data: Optional[pd.DataFrame] = None,
            val_data: Optional[pd.DataFrame] = None,
            feature_columns: Optional[List[str]] = None,
            target_column: Optional[str] = None,
            cv: int = 3,
            scoring: str = 'f1_weighted',
            n_jobs: int = -1
    ) -> Dict:
        """
        Оптимизирует параметры модели.

        Args:
            model_type: Тип модели ('lstm', 'reinforcement', 'technical', 'ensemble', 'hybrid')
            param_grid: Сетка параметров для поиска
            train_data: Данные для обучения (если None, используется self.train_data)
            val_data: Данные для валидации (если None, используется self.val_data)
            feature_columns: Список колонок с признаками
            target_column: Колонка с целевой переменной
            cv: Число фолдов для кросс-валидации
            scoring: Метрика для оптимизации
            n_jobs: Число процессов для параллельного выполнения

        Returns:
            Словарь с результатами оптимизации
        """
        # Получаем параметры из конфигурации, если не переданы
        model_type = model_type or self.config.get("model_type", "technical")

        # Если param_grid не задан, используем параметры по умолчанию
        if param_grid is None:
            if model_type == "technical":
                param_grid = {
                    "classifier": ["rf", "gb"],
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 15]
                }
            elif model_type == "lstm":
                param_grid = {
                    "lstm_units": [50, 100, 150],
                    "dense_units": [32, 64, 128],
                    "dropout_rate": [0.1, 0.2, 0.3]
                }
            elif model_type == "hybrid":
                param_grid = {
                    "alpha": [0.2, 0.3, 0.4],
                    "beta": [0.3, 0.4, 0.5],
                    "gamma": [0.2, 0.3, 0.4]
                }
            else:
                logger.error(f"Оптимизация параметров не поддерживается для модели типа {model_type}")
                return {}

        # Если данные не переданы, используем подготовленные
        train_data = train_data if train_data is not None else self.train_data
        val_data = val_data if val_data is not None else self.val_data

        if train_data is None or train_data.empty:
            logger.error("Нет данных для оптимизации")
            return {}

        # Если признаки не переданы, используем все числовые колонки, кроме целевой
        if feature_columns is None:
            # Исключаем целевую колонку из признаков
            exclude_columns = [target_column] if target_column else []

            # Также исключаем колонки с сигналами
            exclude_columns.extend(['signal', 'buy_signal', 'sell_signal', 'hold_signal'])

            # Выбираем все числовые колонки, кроме исключенных
            feature_columns = [
                col for col in train_data.select_dtypes(include=[np.number]).columns
                if col not in exclude_columns
            ]

        # Оптимизация для разных типов моделей
        if model_type == "technical":
            # Создаем модель
            model = self.build_model(model_type=model_type)

            # Подготавливаем данные
            X, y = model.prepare_data_from_dataframe(train_data)

            # Оптимизируем параметры
            logger.info(f"Оптимизация параметров для модели {model_type}")

            optimization_results = model.optimize_parameters(
                df=train_data,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs
            )

            # Сохраняем оптимальные параметры
            best_params = optimization_results.get('best_params', {})

            logger.info(f"Лучшие параметры: {best_params}")

            # Обновляем параметры модели
            model.model_params.update(best_params)

            # Обучаем модель с оптимальными параметрами
            self.train_model(
                model=model,
                train_data=train_data,
                val_data=val_data,
                feature_columns=feature_columns,
                target_column=target_column
            )

            # Сохраняем модель
            self.model = model

            return optimization_results

        elif model_type == "hybrid":
            # Создаем модель
            model = self.build_model(model_type=model_type)

            # Оптимизируем веса модели
            logger.info(f"Оптимизация весов для гибридной модели")

            # Сначала обучаем модель с параметрами по умолчанию
            self.train_model(
                model=model,
                train_data=train_data,
                val_data=val_data,
                feature_columns=feature_columns,
                target_column=target_column
            )

            # Затем оптимизируем веса
            optimization_results = model.optimize_weights(
                df=val_data if val_data is not None else train_data,
                feature_columns=feature_columns,
                alpha_range=param_grid.get("alpha", [0.2, 0.3, 0.4]),
                beta_range=param_grid.get("beta", [0.3, 0.4, 0.5]),
                gamma_range=param_grid.get("gamma", [0.2, 0.3, 0.4]),
                threshold_range=param_grid.get("decision_threshold", [0.3, 0.4, 0.5, 0.6])
            )

            # Сохраняем оптимальные параметры
            best_params = optimization_results.get('best_params', {})

            logger.info(f"Лучшие параметры: {best_params}")

            return optimization_results

        else:
            logger.error(f"Оптимизация параметров не реализована для модели типа {model_type}")
            return {}


def main():
    """
    Основная функция для запуска приложения.
    """
    # Настройка парсера аргументов командной строки
    parser = argparse.ArgumentParser(description="TensorTrade - торговый бот на основе нейросетей")

    parser.add_argument("--config", type=str, default="config.json", help="Путь к файлу конфигурации")
    parser.add_argument("--mode", type=str, choices=["collect", "train", "backtest", "simulate", "trade", "optimize"],
                        default="backtest", help="Режим работы")
    parser.add_argument("--symbol", type=str, help="Символ торговой пары")
    parser.add_argument("--interval", type=str, help="Интервал свечей")
    parser.add_argument("--start_date", type=str, help="Начальная дата сбора данных")
    parser.add_argument("--end_date", type=str, help="Конечная дата сбора данных")
    parser.add_argument("--model_type", type=str, help="Тип модели")
    parser.add_argument("--data_file", type=str, help="Путь к файлу с данными")
    parser.add_argument("--model_file", type=str, help="Путь к файлу модели")
    parser.add_argument("--demo", action="store_true", help="Использовать демо-режим")

    args = parser.parse_args()

    # Создаем приложение
    app = TensorTrade(config_path=args.config)

    # Выполняем действия в зависимости от режима
    if args.mode == "collect":
        # Собираем данные
        symbol = args.symbol or app.config.get("symbol")
        interval = args.interval or app.config.get("interval")
        start_date = args.start_date or app.config.get("data_start_date")
        end_date = args.end_date or app.config.get("data_end_date")

        app.collect_data(symbol=symbol, interval=interval, start_date=start_date, end_date=end_date)

    elif args.mode == "train":
        # Загружаем данные
        if args.data_file:
            app.load_data(args.data_file)

        # Предобрабатываем данные
        app.preprocess_data()

        # Создаем и обучаем модель
        model_type = args.model_type or app.config.get("model_type")
        model = app.build_model(model_type=model_type)
        app.train_model(model=model)

    elif args.mode == "backtest":
        # Загружаем данные
        if args.data_file:
            app.load_data(args.data_file)

        # Предобрабатываем данные
        app.preprocess_data()

        # Загружаем или создаем модель
        if args.model_file:
            # Загружаем модель из файла
            model_type = args.model_type or app.config.get("model_type")
            model = ModelFactory.create_model(model_type=model_type, name="loaded_model")
            model = model.load(args.model_file)
            app.model = model
        else:
            # Создаем и обучаем модель
            model_type = args.model_type or app.config.get("model_type")
            model = app.build_model(model_type=model_type)
            app.train_model(model=model)

        # Выполняем бэктестирование
        app.backtest_model(model=model)

    elif args.mode == "simulate":
        # Загружаем данные
        if args.data_file:
            app.load_data(args.data_file)

        # Загружаем или создаем модель
        if args.model_file:
            # Загружаем модель из файла
            model_type = args.model_type or app.config.get("model_type")
            model = ModelFactory.create_model(model_type=model_type, name="loaded_model")
            model = model.load(args.model_file)
            app.model = model
        else:
            # Создаем и обучаем модель
            model_type = args.model_type or app.config.get("model_type")
            model = app.build_model(model_type=model_type)
            app.train_model(model=model)

        # Запускаем симуляцию
        symbol = args.symbol or app.config.get("symbol")
        interval = args.interval or app.config.get("interval")

        app.run_simulation(model=model, symbol=symbol, interval=interval)

    elif args.mode == "trade":
        # Загружаем или создаем модель
        if args.model_file:
            # Загружаем модель из файла
            model_type = args.model_type or app.config.get("model_type")
            model = ModelFactory.create_model(model_type=model_type, name="loaded_model")
            model = model.load(args.model_file)
            app.model = model
        else:
            logger.error("Не указан файл модели для торговли")
            return

        # Запускаем живую торговлю
        symbol = args.symbol or app.config.get("symbol")
        interval = args.interval or app.config.get("interval")

        app.start_live_trading(model=model, symbol=symbol, interval=interval, is_demo=args.demo)

        # Ждем, пока пользователь не прервет процесс
        try:
            import time
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            app.stop_live_trading()

    elif args.mode == "optimize":
        # Загружаем данные
        if args.data_file:
            app.load_data(args.data_file)

        # Предобрабатываем данные
        app.preprocess_data()

        # Оптимизируем параметры модели
        model_type = args.model_type or app.config.get("model_type")
        app.optimize_model(model_type=model_type)


if __name__ == "__main__":
    main()