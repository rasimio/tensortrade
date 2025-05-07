"""
Модуль для симуляции торговли с использованием различных стратегий и моделей.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
import os
import json
import time
import asyncio
import websockets
from concurrent.futures import ThreadPoolExecutor

from src.models.base_model import BaseModel
from src.simulation.market import MarketSimulator
from src.backtest.engine import BacktestEngine
from src.backtest.reporting import BacktestReport

logger = logging.getLogger(__name__)


class TradingSimulator:
    """
    Класс для симуляции торговли с использованием различных стратегий и моделей.
    """

    def __init__(
            self,
            model: Optional[BaseModel] = None,
            strategy_func: Optional[Callable] = None,
            market_simulator: Optional[MarketSimulator] = None,
            initial_balance: float = 10000.0,
            fee: float = 0.001,
            slippage: float = 0.0005,
            position_size: float = 1.0,
            stop_loss: Optional[float] = None,
            take_profit: Optional[float] = None,
            leverage: float = 1.0,
            allow_short: bool = False,
            use_websocket: bool = False,
            ws_url: str = 'ws://localhost:8765',
            ws_symbol: str = 'BTCUSDT',
            output_dir: str = 'output/simulations'
    ):
        """
        Инициализирует симулятор торговли.

        Args:
            model: Модель для прогнозирования (опционально)
            strategy_func: Функция стратегии (опционально)
            market_simulator: Симулятор рынка (опционально)
            initial_balance: Начальный баланс
            fee: Комиссия за сделку (например, 0.001 = 0.1%)
            slippage: Проскальзывание (например, 0.0005 = 0.05%)
            position_size: Размер позиции (доля от баланса)
            stop_loss: Уровень стоп-лосса (например, 0.02 = 2%)
            take_profit: Уровень тейк-профита (например, 0.05 = 5%)
            leverage: Плечо (например, 1.0 = без плеча, 2.0 = 2x плечо)
            allow_short: Разрешены ли короткие позиции
            use_websocket: Использовать ли WebSocket для получения данных
            ws_url: URL для WebSocket соединения
            ws_symbol: Символ для WebSocket
            output_dir: Директория для сохранения результатов
        """
        self.model = model
        self.strategy_func = strategy_func
        self.initial_balance = initial_balance
        self.fee = fee
        self.slippage = slippage
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.leverage = leverage
        self.allow_short = allow_short
        self.use_websocket = use_websocket
        self.ws_url = ws_url
        self.ws_symbol = ws_symbol
        self.output_dir = output_dir

        # Создаем директорию для вывода, если она не существует
        os.makedirs(output_dir, exist_ok=True)

        # Создаем или используем переданный симулятор рынка
        if market_simulator is None:
            self.market_simulator = MarketSimulator(
                initial_balance=initial_balance,
                fee=fee,
                slippage=slippage,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=leverage,
                allow_short=allow_short,
                use_websocket=use_websocket,
                ws_url=ws_url,
                ws_symbol=ws_symbol
            )
        else:
            self.market_simulator = market_simulator

        # Если модель задана, но функция стратегии не задана, создаем стратегию на основе модели
        if self.model is not None and self.strategy_func is None:
            self.strategy_func = self._create_strategy_from_model()

        # Thread pool для асинхронных операций
        self.executor = ThreadPoolExecutor(max_workers=5)

        # Статистика торговли
        self.trade_stats = {}

        logger.info(f"Инициализирован симулятор торговли")

    def _create_strategy_from_model(self, threshold: float = 0.5) -> Callable:
        """
        Создает функцию стратегии на основе модели.

        Args:
            threshold: Порог для сигналов

        Returns:
            Функция стратегии
        """

        def strategy(data: Dict, simulator: MarketSimulator) -> None:
            """
            Стратегия на основе модели.

            Args:
                data: Данные тика
                simulator: Симулятор рынка
            """
            try:
                # Получаем признаки для модели
                features = np.array([
                    data.get('open', data['close']),
                    data.get('high', data['close']),
                    data.get('low', data['close']),
                    data['close'],
                    data.get('volume', 0.0)
                ]).reshape(1, -1)

                # Делаем прогноз
                prediction = self.model.predict(features)

                # Преобразуем прогноз в действие
                if isinstance(prediction, np.ndarray):
                    if prediction.size > 1:
                        # Если прогноз - массив классов или вероятностей
                        action = np.argmax(prediction) - 1  # -1: продать, 0: держать, 1: купить
                    else:
                        # Если прогноз - скаляр
                        if prediction > threshold:
                            action = 1  # купить
                        elif prediction < -threshold:
                            action = -1  # продать
                        else:
                            action = 0  # держать
                else:
                    # Если прогноз - скаляр
                    if prediction > threshold:
                        action = 1  # купить
                    elif prediction < -threshold:
                        action = -1  # продать
                    else:
                        action = 0  # держать

                # Выполняем действие
                if action == 1:
                    # Сигнал на покупку
                    if simulator.position <= 0:
                        simulator.buy()
                elif action == -1:
                    # Сигнал на продажу
                    if simulator.position >= 0:
                        simulator.sell()

            except Exception as e:
                logger.error(f"Ошибка при выполнении стратегии на основе модели: {e}")

        return strategy

    def run_simulation(
            self,
            data: Optional[pd.DataFrame] = None,
            start_index: int = 0,
            end_index: Optional[int] = None,
            step: int = 1,
            save_results: bool = True
    ) -> Dict:
        """
        Запускает симуляцию торговли на исторических данных.

        Args:
            data: DataFrame с историческими данными (опционально)
            start_index: Индекс начала симуляции
            end_index: Индекс конца симуляции (если None, используется весь доступный диапазон)
            step: Шаг по индексам
            save_results: Сохранять ли результаты

        Returns:
            Словарь с результатами симуляции
        """
        if self.strategy_func is None:
            logger.error("Стратегия не задана")
            return {}

        # Если данные переданы, используем их
        if data is not None:
            self.market_simulator.data = data

        # Запускаем симуляцию
        results = self.market_simulator.run_simulation(
            strategy_func=self.strategy_func,
            start_index=start_index,
            end_index=end_index,
            step=step
        )

        # Сохраняем статистику
        self.trade_stats = results.get('stats', {})

        # Сохраняем результаты, если нужно
        if save_results:
            self._save_simulation_results(results)

        return results

    async def run_live_simulation(
            self,
            duration: Optional[int] = None,
            save_results: bool = True
    ) -> Dict:
        """
        Запускает симуляцию в режиме реального времени с использованием WebSocket.

        Args:
            duration: Продолжительность симуляции в секундах (если None, выполняется до остановки)
            save_results: Сохранять ли результаты

        Returns:
            Словарь с результатами симуляции
        """
        if self.strategy_func is None:
            logger.error("Стратегия не задана")
            return {}

        if not self.use_websocket:
            logger.warning("WebSocket не активирован, используйте use_websocket=True")
            return {}

        # Запускаем симуляцию
        results = await self.market_simulator.run_live_simulation(
            strategy_func=self.strategy_func,
            duration=duration
        )

        # Сохраняем статистику
        self.trade_stats = results.get('stats', {})

        # Сохраняем результаты, если нужно
        if save_results:
            self._save_simulation_results(results)

        return results

    def _save_simulation_results(self, results: Dict) -> None:
        """
        Сохраняет результаты симуляции.

        Args:
            results: Словарь с результатами симуляции
        """
        # Создаем директорию для результатов текущей симуляции
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sim_dir = os.path.join(self.output_dir, f"sim_{timestamp}")
        os.makedirs(sim_dir, exist_ok=True)

        # Сохраняем статистику
        with open(os.path.join(sim_dir, "stats.json"), "w") as f:
            json.dump(results.get('stats', {}), f, indent=4)

        # Сохраняем историю сделок
        trades_df = pd.DataFrame(results.get('trades', []))
        if not trades_df.empty:
            trades_df.to_csv(os.path.join(sim_dir, "trades.csv"), index=False)

        # Сохраняем историю капитала
        equity_df = pd.DataFrame(results.get('equity_history', []))
        if not equity_df.empty:
            equity_df.to_csv(os.path.join(sim_dir, "equity.csv"), index=False)

        logger.info(f"Результаты симуляции сохранены в {sim_dir}")

    def generate_report(
            self,
            results: Optional[Dict] = None,
            output_dir: Optional[str] = None,
            strategy_name: str = "Strategy"
    ) -> str:
        """
        Генерирует отчет о результатах симуляции.

        Args:
            results: Словарь с результатами симуляции (опционально, если None, используются последние результаты)
            output_dir: Директория для сохранения отчета (опционально)
            strategy_name: Название стратегии

        Returns:
            Путь к сгенерированному отчету
        """
        # Если результаты не переданы, проверяем наличие статистики торговли
        if results is None:
            if not self.trade_stats:
                logger.error("Нет результатов для генерации отчета")
                return ""

            # Получаем данные из симулятора рынка
            data = self.market_simulator.get_equity_curve()
            trades = self.market_simulator.trades
            metrics = self.trade_stats
        else:
            # Формируем данные из переданных результатов
            equity_history = results.get('equity_history', [])
            trades = results.get('trades', [])
            metrics = results.get('stats', {})

            # Создаем DataFrame из истории капитала
            data = pd.DataFrame(equity_history)

            # Устанавливаем временную метку в качестве индекса
            if 'timestamp' in data.columns:
                data.set_index('timestamp', inplace=True)

        # Если директория не задана, используем директорию по умолчанию
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "reports")
            os.makedirs(output_dir, exist_ok=True)

        # Создаем отчет
        report = BacktestReport(
            data=data,
            trades=trades,
            metrics=metrics,
            strategy_name=strategy_name,
            output_dir=output_dir
        )

        # Генерируем полный отчет
        report_paths = report.generate_full_report()

        logger.info(f"Отчет сохранен в {report_paths['html_report']}")

        return report_paths['html_report']

    def run_backtest(
            self,
            data: pd.DataFrame,
            price_column: str = 'close',
            timestamp_column: Optional[str] = None,
            strategy_name: str = "Strategy",
            save_report: bool = True
    ) -> Dict:
        """
        Запускает бэктестирование стратегии.

        Args:
            data: DataFrame с историческими данными
            price_column: Колонка с ценами
            timestamp_column: Колонка с временными метками
            strategy_name: Название стратегии
            save_report: Сохранять ли отчет

        Returns:
            Словарь с результатами бэктестирования
        """
        if self.strategy_func is None:
            logger.error("Стратегия не задана")
            return {}

        # Создаем объект для бэктестирования
        backtest_engine = BacktestEngine(
            initial_balance=self.initial_balance,
            fee=self.fee,
            slippage=self.slippage,
            position_size=self.position_size,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            leverage=self.leverage,
            allow_short=self.allow_short
        )

        # Запускаем бэктестирование
        backtest_results = backtest_engine.run_backtest(
            data=data,
            custom_logic=lambda df: self._apply_strategy_to_dataframe(df, self.strategy_func),
            price_column=price_column,
            timestamp_column=timestamp_column
        )

        # Сохраняем результаты в метрики
        self.trade_stats = backtest_results['metrics']

        # Генерируем отчет, если нужно
        if save_report:
            report_dir = os.path.join(self.output_dir, "backtests")
            os.makedirs(report_dir, exist_ok=True)

            backtest_report = BacktestReport(
                data=backtest_results['data'],
                trades=backtest_results['trades'],
                metrics=backtest_results['metrics'],
                strategy_name=strategy_name,
                output_dir=report_dir
            )

            report_paths = backtest_report.generate_full_report()

            logger.info(f"Отчет о бэктестировании сохранен в {report_paths['html_report']}")

            # Добавляем пути к отчетам в результаты
            backtest_results['report_paths'] = report_paths

        return backtest_results

    def _apply_strategy_to_dataframe(self, df: pd.DataFrame, strategy_func: Callable) -> pd.DataFrame:
        """
        Применяет стратегию к DataFrame.

        Args:
            df: DataFrame с историческими данными
            strategy_func: Функция стратегии

        Returns:
            DataFrame с добавленными сигналами
        """
        # Создаем копию DataFrame
        result_df = df.copy()

        # Добавляем колонки для сигналов
        result_df['signal'] = 0  # 0: держать, 1: покупать, 2: продавать

        # Создаем симулятор рынка для использования в стратегии
        simulator = MarketSimulator(
            data=result_df,
            initial_balance=self.initial_balance,
            fee=self.fee,
            slippage=self.slippage,
            position_size=self.position_size,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            leverage=self.leverage,
            allow_short=self.allow_short
        )

        # Проходим по каждой строке DataFrame
        for i in range(len(result_df)):
            # Создаем словарь с данными текущего тика
            tick_data = {
                'timestamp': result_df.index[i] if hasattr(result_df.index[0], 'strftime') else i,
                'open': result_df.iloc[i].get('open', result_df.iloc[i]['close']),
                'high': result_df.iloc[i].get('high', result_df.iloc[i]['close']),
                'low': result_df.iloc[i].get('low', result_df.iloc[i]['close']),
                'close': result_df.iloc[i]['close'],
                'volume': result_df.iloc[i].get('volume', 0.0)
            }

            # Сохраняем текущую позицию
            current_position = simulator.position

            # Вызываем функцию стратегии
            strategy_func(tick_data, simulator)

            # Определяем сигнал на основе изменения позиции
            if simulator.position > current_position:
                result_df.iloc[i, result_df.columns.get_loc('signal')] = 1  # купить
            elif simulator.position < current_position:
                result_df.iloc[i, result_df.columns.get_loc('signal')] = 2  # продать

        return result_df

    def compare_strategies(
            self,
            strategy_funcs: List[Callable],
            strategy_names: List[str],
            data: pd.DataFrame,
            price_column: str = 'close',
            timestamp_column: Optional[str] = None,
            save_report: bool = True
    ) -> Dict:
        """
        Сравнивает несколько стратегий на одних данных.

        Args:
            strategy_funcs: Список функций стратегий
            strategy_names: Список названий стратегий
            data: DataFrame с историческими данными
            price_column: Колонка с ценами
            timestamp_column: Колонка с временными метками
            save_report: Сохранять ли отчет

        Returns:
            Словарь с результатами сравнения
        """
        if len(strategy_funcs) != len(strategy_names):
            logger.error("Количество стратегий и названий должно совпадать")
            return {}

        # Результаты для каждой стратегии
        strategy_results = []

        # Запускаем бэктестирование для каждой стратегии
        for i, (strategy_func, strategy_name) in enumerate(zip(strategy_funcs, strategy_names)):
            logger.info(f"Бэктестирование стратегии {strategy_name}")

            # Сохраняем текущую стратегию
            original_strategy = self.strategy_func
            self.strategy_func = strategy_func

            # Запускаем бэктестирование
            backtest_results = self.run_backtest(
                data=data,
                price_column=price_column,
                timestamp_column=timestamp_column,
                strategy_name=strategy_name,
                save_report=save_report
            )

            # Восстанавливаем оригинальную стратегию
            self.strategy_func = original_strategy

            # Добавляем результаты в список
            strategy_results.append({
                'name': strategy_name,
                'results': backtest_results
            })

        # Создаем сравнительный отчет
        if save_report and len(strategy_results) >= 2:
            # Директория для сравнительного отчета
            comparison_dir = os.path.join(self.output_dir, "comparisons")
            os.makedirs(comparison_dir, exist_ok=True)

            # Сравниваем первые две стратегии (можно расширить для всех пар)
            from src.backtest.reporting import compare_backtest_results

            comparison = compare_backtest_results(
                results1=strategy_results[0]['results'],
                results2=strategy_results[1]['results'],
                name1=strategy_results[0]['name'],
                name2=strategy_results[1]['name'],
                output_dir=comparison_dir
            )

            logger.info(f"Сравнительный отчет сохранен в {comparison_dir}")

        return {
            'strategy_results': strategy_results
        }


class LiveTrader:
    """
    Класс для живой торговли с использованием моделей и стратегий.
    """

    def __init__(
            self,
            model: Optional[BaseModel] = None,
            strategy_func: Optional[Callable] = None,
            exchange_api: Optional[Any] = None,
            symbol: str = 'BTCUSDT',
            interval: str = '1m',
            initial_balance: float = 10000.0,
            fee: float = 0.001,
            position_size: float = 0.1,
            stop_loss: Optional[float] = None,
            take_profit: Optional[float] = None,
            leverage: float = 1.0,
            allow_short: bool = False,
            is_demo: bool = True,
            use_websocket: bool = True,
            ws_url: str = 'wss://stream.binance.com:9443/ws',
            output_dir: str = 'output/trading'
    ):
        """
        Инициализирует трейдера для живой торговли.

        Args:
            model: Модель для прогнозирования (опционально)
            strategy_func: Функция стратегии (опционально)
            exchange_api: API биржи (опционально)
            symbol: Символ торговой пары
            interval: Интервал свечей
            initial_balance: Начальный баланс
            fee: Комиссия за сделку (например, 0.001 = 0.1%)
            position_size: Размер позиции (доля от баланса)
            stop_loss: Уровень стоп-лосса (например, 0.02 = 2%)
            take_profit: Уровень тейк-профита (например, 0.05 = 5%)
            leverage: Плечо (например, 1.0 = без плеча, 2.0 = 2x плечо)
            allow_short: Разрешены ли короткие позиции
            is_demo: Использовать ли демо-режим (без реальных сделок)
            use_websocket: Использовать ли WebSocket для получения данных
            ws_url: URL для WebSocket соединения
            output_dir: Директория для сохранения результатов
        """
        self.model = model
        self.strategy_func = strategy_func
        self.exchange_api = exchange_api
        self.symbol = symbol
        self.interval = interval
        self.initial_balance = initial_balance
        self.fee = fee
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.leverage = leverage
        self.allow_short = allow_short
        self.is_demo = is_demo
        self.use_websocket = use_websocket
        self.ws_url = ws_url
        self.output_dir = output_dir

        # Создаем директорию для вывода, если она не существует
        os.makedirs(output_dir, exist_ok=True)

        # Если модель задана, но функция стратегии не задана, создаем стратегию на основе модели
        if self.model is not None and self.strategy_func is None:
            self.strategy_func = self._create_strategy_from_model()

        # WebSocket клиент
        self.ws = None
        self.ws_running = False

        # Thread pool для асинхронных операций
        self.executor = ThreadPoolExecutor(max_workers=5)

        # Состояние торговли
        self.balance = initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.trades = []
        self.is_running = False

        # Текущие данные
        self.current_data = {}

        # Обработчики событий
        self.on_tick_handlers = []
        self.on_trade_handlers = []

        logger.info(f"Инициализирован трейдер для {'демо' if is_demo else 'реальной'} торговли с {symbol}")

    def _create_strategy_from_model(self, threshold: float = 0.5) -> Callable:
        """
        Создает функцию стратегии на основе модели.

        Args:
            threshold: Порог для сигналов

        Returns:
            Функция стратегии
        """

        def strategy(data: Dict, simulator: MarketSimulator) -> None:
            """
            Стратегия на основе модели.

            Args:
                data: Данные тика
                simulator: Симулятор рынка
            """
            try:
                # Проверяем, что модель существует и обучена
                if self.model is None:
                    logger.warning("Модель не инициализирована")
                    return

                if not hasattr(self.model, 'is_trained') or not self.model.is_trained:
                    logger.warning("Модель не обучена")
                    return

                # Получаем признаки для модели
                features = np.array([
                    data.get('open', data['close']),
                    data.get('high', data['close']),
                    data.get('low', data['close']),
                    data['close'],
                    data.get('volume', 0.0)
                ]).reshape(1, -1)

                # Делаем прогноз с обработкой исключений
                try:
                    prediction = self.model.predict(features)
                except Exception as predict_error:
                    logger.error(f"Ошибка при прогнозировании: {predict_error}")
                    return

                # Преобразуем прогноз в действие
                if isinstance(prediction, np.ndarray):
                    if prediction.size > 1:
                        # Если прогноз - массив классов или вероятностей
                        action = np.argmax(prediction) - 1  # -1: продать, 0: держать, 1: купить
                    else:
                        # Если прогноз - скаляр
                        if prediction > threshold:
                            action = 1  # купить
                        elif prediction < -threshold:
                            action = -1  # продать
                        else:
                            action = 0  # держать
                else:
                    # Если прогноз - скаляр
                    if prediction > threshold:
                        action = 1  # купить
                    elif prediction < -threshold:
                        action = -1  # продать
                    else:
                        action = 0  # держать

                # Выполняем действие
                if action == 1:
                    # Сигнал на покупку
                    if simulator.position <= 0:
                        simulator.buy()
                elif action == -1:
                    # Сигнал на продажу
                    if simulator.position >= 0:
                        simulator.sell()

            except Exception as e:
                logger.error(f"Ошибка при выполнении стратегии на основе модели: {e}")

        return strategy

    def register_on_tick_handler(self, handler: Callable) -> None:
        """
        Регистрирует обработчик события "тик".

        Args:
            handler: Функция-обработчик события
        """
        self.on_tick_handlers.append(handler)

    def register_on_trade_handler(self, handler: Callable) -> None:
        """
        Регистрирует обработчик события "сделка".

        Args:
            handler: Функция-обработчик события
        """
        self.on_trade_handlers.append(handler)

    def _handle_tick(self, tick_data: Dict) -> None:
        """
        Обрабатывает событие "тик".

        Args:
            tick_data: Данные тика
        """
        self.current_data = tick_data

        # Вызываем все обработчики события "тик"
        for handler in self.on_tick_handlers:
            try:
                handler(tick_data)
            except Exception as e:
                logger.error(f"Ошибка в обработчике события 'тик': {e}")

        # Если торговля запущена и стратегия задана, вызываем ее
        if self.is_running and self.strategy_func is not None:
            try:
                # Получаем сигнал от стратегии
                signal = self.strategy_func(tick_data)

                # Выполняем действие в зависимости от сигнала
                if signal == 1:  # купить
                    if self.position <= 0:
                        self.buy()
                elif signal == 2:  # продать
                    if self.position >= 0:
                        self.sell()

            except Exception as e:
                logger.error(f"Ошибка при выполнении стратегии: {e}")

    def _handle_trade(self, trade_data: Dict) -> None:
        """
        Обрабатывает событие "сделка".

        Args:
            trade_data: Данные сделки
        """
        # Добавляем сделку в историю
        self.trades.append(trade_data)

        # Обновляем состояние
        self.balance = trade_data.get('balance', self.balance)
        self.position = trade_data.get('position', self.position)
        self.entry_price = trade_data.get('price', self.entry_price) if trade_data.get('type') in ['buy',
                                                                                                   'short'] else 0.0

        # Вызываем все обработчики события "сделка"
        for handler in self.on_trade_handlers:
            try:
                handler(trade_data)
            except Exception as e:
                logger.error(f"Ошибка в обработчике события 'сделка': {e}")

    def connect_to_exchange(self) -> bool:
        """
        Подключается к бирже.

        Returns:
            True, если подключение успешно, иначе False
        """
        if self.exchange_api is None:
            logger.warning("API биржи не задано")
            return False

        # Проверяем подключение
        try:
            # Получаем информацию об аккаунте
            account_info = self.exchange_api.get_account()

            # Устанавливаем начальный баланс
            self.balance = float(account_info.get('totalWalletBalance', self.initial_balance))

            # Проверяем текущие позиции
            positions = self.exchange_api.get_position_risk(symbol=self.symbol)
            if positions:
                self.position = float(positions[0].get('positionAmt', 0))
                self.entry_price = float(positions[0].get('entryPrice', 0))

            logger.info(f"Подключение к бирже успешно: баланс={self.balance}, позиция={self.position}")

            return True

        except Exception as e:
            logger.error(f"Ошибка при подключении к бирже: {e}")
            return False

    def connect_websocket(self) -> None:
        """
        Подключается к WebSocket серверу.
        """
        if not self.use_websocket:
            logger.warning("WebSocket не используется")
            return

        # Запускаем WebSocket клиент в отдельном потоке
        self.executor.submit(self._run_websocket_client)

        logger.info(f"WebSocket клиент запущен, подключение к {self.ws_url}")

    def _run_websocket_client(self) -> None:
        """
        Запускает WebSocket клиент в асинхронном режиме.
        """
        # Создаем и запускаем цикл событий для WebSocket
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._websocket_client())
        except Exception as e:
            logger.error(f"Ошибка в WebSocket клиенте: {e}")
        finally:
            loop.close()

    async def _websocket_client(self) -> None:
        """
        Асинхронный WebSocket клиент.
        """
        self.ws_running = True

        # Формируем URL для WebSocket подписки
        ws_subscribe_url = f"{self.ws_url}/{self.symbol.lower()}@kline_{self.interval}"

        while self.ws_running:
            try:
                async with websockets.connect(ws_subscribe_url) as websocket:
                    self.ws = websocket

                    # Обрабатываем входящие сообщения
                    while self.ws_running:
                        try:
                            message = await websocket.recv()
                            await self._process_websocket_message(message)
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("WebSocket соединение закрыто")
                            break

            except Exception as e:
                logger.error(f"Ошибка при подключении к WebSocket: {e}")
                # Ждем перед повторной попыткой
                await asyncio.sleep(5)

    async def _process_websocket_message(self, message: str) -> None:
        """
        Обрабатывает сообщение от WebSocket сервера.

        Args:
            message: Полученное сообщение
        """
        try:
            data = json.loads(message)

            # Проверяем, что это данные свечи
            if 'k' in data:
                kline_data = data['k']

                # Преобразуем данные в формат тика
                tick_data = {
                    'timestamp': datetime.fromtimestamp(kline_data.get('t', 0) / 1000),
                    'symbol': data.get('s', self.symbol),
                    'open': float(kline_data.get('o', 0)),
                    'high': float(kline_data.get('h', 0)),
                    'low': float(kline_data.get('l', 0)),
                    'close': float(kline_data.get('c', 0)),
                    'volume': float(kline_data.get('v', 0))
                }

                # Обрабатываем тик
                self._handle_tick(tick_data)

        except json.JSONDecodeError:
            logger.warning(f"Получено некорректное JSON-сообщение: {message}")
        except Exception as e:
            logger.error(f"Ошибка при обработке сообщения WebSocket: {e}")

    def disconnect_websocket(self) -> None:
        """
        Отключается от WebSocket сервера.
        """
        if not self.use_websocket or not self.ws_running:
            return

        self.ws_running = False

        logger.info("WebSocket клиент остановлен")

    def start_trading(self) -> None:
        """
        Запускает торговлю.
        """
        if self.is_running:
            logger.warning("Торговля уже запущена")
            return

        # Проверяем, задана ли стратегия
        if self.strategy_func is None:
            logger.error("Стратегия не задана")
            return

        # Если не используется демо-режим, подключаемся к бирже
        if not self.is_demo and self.exchange_api is not None:
            if not self.connect_to_exchange():
                logger.error("Не удалось подключиться к бирже")
                return

        # Подключаемся к WebSocket, если используется
        if self.use_websocket:
            self.connect_websocket()

        # Запускаем торговлю
        self.is_running = True

        logger.info(f"Торговля запущена в {'демо' if self.is_demo else 'реальном'} режиме")

    def stop_trading(self) -> None:
        """
        Останавливает торговлю.
        """
        if not self.is_running:
            return

        # Останавливаем торговлю
        self.is_running = False

        # Отключаемся от WebSocket, если используется
        if self.use_websocket:
            self.disconnect_websocket()

        logger.info("Торговля остановлена")

    def buy(self, price: Optional[float] = None, size: Optional[float] = None) -> Dict:
        """
        Открывает длинную позицию.

        Args:
            price: Цена входа (если None, используется текущая цена)
            size: Размер позиции (если None, используется значение position_size)

        Returns:
            Словарь с информацией о сделке
        """
        if not self.is_running:
            logger.warning("Торговля не запущена")
            return {}

        # Если уже есть короткая позиция, закрываем ее сначала
        if self.position < 0:
            self.close_position(price)

        # Получаем текущую цену, если не задана
        if price is None:
            price = self.current_data.get('close', 0.0)

        # Получаем текущее время
        timestamp = self.current_data.get('timestamp', datetime.now())

        # Получаем размер позиции, если не задан
        if size is None:
            size = self.position_size

        # Если не используется демо-режим, выполняем реальную сделку
        if not self.is_demo and self.exchange_api is not None:
            try:
                # Рассчитываем размер позиции в базовой валюте
                position_value = self.balance * size
                position_amount = position_value / price

                # Создаем ордер
                order = self.exchange_api.create_order(
                    symbol=self.symbol,
                    side='BUY',
                    type='MARKET',
                    quantity=position_amount
                )

                # Получаем детали ордера
                order_details = self.exchange_api.get_order(
                    symbol=self.symbol,
                    orderId=order['orderId']
                )

                # Обновляем состояние
                self.position += float(order_details['executedQty'])
                self.entry_price = float(order_details['price'])
                self.balance -= float(order_details['cummulativeQuoteQty'])

                # Создаем информацию о сделке
                trade_info = {
                    'type': 'buy',
                    'timestamp': timestamp,
                    'price': self.entry_price,
                    'position': float(order_details['executedQty']),
                    'value': float(order_details['cummulativeQuoteQty']),
                    'fee': float(order_details.get('commission', 0)),
                    'balance': self.balance,
                    'order_id': order['orderId']
                }

                # Вызываем обработчик события "сделка"
                self._handle_trade(trade_info)

                logger.info(f"Открыта длинная позиция: цена={self.entry_price:.2f}, размер={self.position:.6f}")

                return trade_info

            except Exception as e:
                logger.error(f"Ошибка при создании ордера на покупку: {e}")
                return {}

        else:
            # Демо-режим, симулируем сделку
            # Учитываем проскальзывание
            effective_price = price * (1 + self.fee)

            # Рассчитываем размер позиции
            position_value = self.balance * size

            # При использовании плеча
            if self.leverage > 1.0:
                position_value *= self.leverage

            # Учитываем комиссию
            effective_position_value = position_value / (1 + self.fee)
            position_amount = effective_position_value / effective_price

            # Обновляем баланс
            self.balance -= position_value / self.leverage

            # Если баланс стал отрицательным, откатываем сделку
            if self.balance < 0:
                self.balance += position_value / self.leverage
                logger.warning("Недостаточно средств для покупки")
                return {}

            # Обновляем позицию
            self.position += position_amount
            self.entry_price = effective_price

            # Создаем информацию о сделке
            trade_info = {
                'type': 'buy',
                'timestamp': timestamp,
                'price': effective_price,
                'position': position_amount,
                'value': position_value,
                'fee': position_value * self.fee,
                'balance': self.balance
            }

            # Вызываем обработчик события "сделка"
            self._handle_trade(trade_info)

            logger.info(f"[ДЕМО] Открыта длинная позиция: цена={effective_price:.2f}, размер={position_amount:.6f}")

            return trade_info

    def sell(self, price: Optional[float] = None, size: Optional[float] = None) -> Dict:
        """
        Открывает короткую позицию или закрывает длинную.

        Args:
            price: Цена входа/выхода (если None, используется текущая цена)
            size: Размер позиции (если None, используется значение position_size или вся текущая позиция)

        Returns:
            Словарь с информацией о сделке
        """
        if not self.is_running:
            logger.warning("Торговля не запущена")
            return {}

        # Получаем текущую цену, если не задана
        if price is None:
            price = self.current_data.get('close', 0.0)

        # Получаем текущее время
        timestamp = self.current_data.get('timestamp', datetime.now())

        # Если есть длинная позиция, закрываем ее
        if self.position > 0:
            return self.close_position(price)

        # Если короткие позиции не разрешены, выходим
        if not self.allow_short:
            logger.warning("Короткие позиции не разрешены")
            return {}

        # Получаем размер позиции, если не задан
        if size is None:
            size = self.position_size

        # Если не используется демо-режим, выполняем реальную сделку
        if not self.is_demo and self.exchange_api is not None:
            try:
                # Рассчитываем размер позиции в базовой валюте
                position_value = self.balance * size
                position_amount = position_value / price

                # Создаем ордер
                order = self.exchange_api.create_order(
                    symbol=self.symbol,
                    side='SELL',
                    type='MARKET',
                    quantity=position_amount
                )

                # Получаем детали ордера
                order_details = self.exchange_api.get_order(
                    symbol=self.symbol,
                    orderId=order['orderId']
                )

                # Обновляем состояние
                self.position -= float(order_details['executedQty'])
                self.entry_price = float(order_details['price'])

                # Создаем информацию о сделке
                trade_info = {
                    'type': 'short',
                    'timestamp': timestamp,
                    'price': self.entry_price,
                    'position': -float(order_details['executedQty']),
                    'value': float(order_details['cummulativeQuoteQty']),
                    'fee': float(order_details.get('commission', 0)),
                    'balance': self.balance,
                    'order_id': order['orderId']
                }

                # Вызываем обработчик события "сделка"
                self._handle_trade(trade_info)

                logger.info(f"Открыта короткая позиция: цена={self.entry_price:.2f}, размер={-self.position:.6f}")

                return trade_info

            except Exception as e:
                logger.error(f"Ошибка при создании ордера на продажу: {e}")
                return {}

        else:
            # Демо-режим, симулируем сделку
            # Учитываем проскальзывание (для продажи - отрицательное)
            effective_price = price * (1 - self.fee)

            # Рассчитываем размер позиции
            position_value = self.balance * size

            # При использовании плеча
            if self.leverage > 1.0:
                position_value *= self.leverage

            # Учитываем комиссию
            effective_position_value = position_value / (1 + self.fee)
            position_amount = effective_position_value / effective_price

            # Обновляем позицию
            self.position = -position_amount
            self.entry_price = effective_price

            # Создаем информацию о сделке
            trade_info = {
                'type': 'short',
                'timestamp': timestamp,
                'price': effective_price,
                'position': -position_amount,
                'value': position_value,
                'fee': position_value * self.fee,
                'balance': self.balance
            }

            # Вызываем обработчик события "сделка"
            self._handle_trade(trade_info)

            logger.info(f"[ДЕМО] Открыта короткая позиция: цена={effective_price:.2f}, размер={-self.position:.6f}")

            return trade_info

    def close_position(self, price: Optional[float] = None) -> Dict:
        """
        Закрывает текущую позицию.

        Args:
            price: Цена выхода (если None, используется текущая цена)

        Returns:
            Словарь с информацией о сделке
        """
        if not self.is_running:
            logger.warning("Торговля не запущена")
            return {}

        # Если нет открытой позиции, выходим
        if self.position == 0:
            logger.warning("Нет открытой позиции для закрытия")
            return {}

        # Получаем текущую цену, если не задана
        if price is None:
            price = self.current_data.get('close', 0.0)

        # Получаем текущее время
        timestamp = self.current_data.get('timestamp', datetime.now())

        # Если не используется демо-режим, выполняем реальную сделку
        if not self.is_demo and self.exchange_api is not None:
            try:
                # Определяем направление
                side = 'SELL' if self.position > 0 else 'BUY'

                # Создаем ордер
                order = self.exchange_api.create_order(
                    symbol=self.symbol,
                    side=side,
                    type='MARKET',
                    quantity=abs(self.position)
                )

                # Получаем детали ордера
                order_details = self.exchange_api.get_order(
                    symbol=self.symbol,
                    orderId=order['orderId']
                )

                # Рассчитываем прибыль/убыток
                if self.position > 0:
                    # Для длинной позиции
                    profit_loss = float(order_details['cummulativeQuoteQty']) - (self.position * self.entry_price)
                else:
                    # Для короткой позиции
                    profit_loss = (abs(self.position) * self.entry_price) - float(order_details['cummulativeQuoteQty'])

                # Обновляем баланс
                self.balance += float(order_details['cummulativeQuoteQty']) if self.position > 0 else -float(
                    order_details['cummulativeQuoteQty'])

                # Создаем информацию о сделке
                trade_info = {
                    'type': 'close_long' if self.position > 0 else 'close_short',
                    'timestamp': timestamp,
                    'price': float(order_details['price']),
                    'position': self.position,
                    'value': float(order_details['cummulativeQuoteQty']),
                    'fee': float(order_details.get('commission', 0)),
                    'profit_loss': profit_loss,
                    'balance': self.balance,
                    'order_id': order['orderId']
                }

                # Обнуляем позицию
                self.position = 0
                self.entry_price = 0

                # Вызываем обработчик события "сделка"
                self._handle_trade(trade_info)

                logger.info(
                    f"Закрыта позиция: цена={float(order_details['price']):.2f}, прибыль/убыток={profit_loss:.2f}")

                return trade_info

            except Exception as e:
                logger.error(f"Ошибка при закрытии позиции: {e}")
                return {}

        else:
            # Демо-режим, симулируем сделку
            # Учитываем проскальзывание
            if self.position > 0:
                # Для закрытия длинной позиции - отрицательное
                effective_price = price * (1 - self.fee)
            else:
                # Для закрытия короткой позиции - положительное
                effective_price = price * (1 + self.fee)

            # Рассчитываем стоимость позиции
            position_value = abs(self.position) * effective_price

            # Учитываем комиссию
            fee = position_value * self.fee
            net_position_value = position_value - fee

            # Рассчитываем прибыль/убыток
            if self.position > 0:
                # Для длинной позиции
                profit_loss = net_position_value - (self.position * self.entry_price)
            else:
                # Для короткой позиции
                profit_loss = (abs(self.position) * self.entry_price) - net_position_value

            # Обновляем баланс
            if self.position > 0:
                # Для длинной позиции
                self.balance += net_position_value
            else:
                # Для короткой позиции
                self.balance -= net_position_value

            # Создаем информацию о сделке
            trade_info = {
                'type': 'close_long' if self.position > 0 else 'close_short',
                'timestamp': timestamp,
                'price': effective_price,
                'position': self.position,
                'value': position_value,
                'fee': fee,
                'profit_loss': profit_loss,
                'balance': self.balance
            }

            # Обнуляем позицию
            self.position = 0
            self.entry_price = 0

            # Вызываем обработчик события "сделка"
            self._handle_trade(trade_info)

            logger.info(f"[ДЕМО] Закрыта позиция: цена={effective_price:.2f}, прибыль/убыток={profit_loss:.2f}")

            return trade_info

    def save_trading_results(self) -> str:
        """
        Сохраняет результаты торговли в файл.

        Returns:
            Путь к сохраненному файлу
        """
        # Создаем директорию для результатов
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.output_dir, f"trading_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)

        # Сохраняем историю сделок
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            trades_file = os.path.join(results_dir, "trades.csv")
            trades_df.to_csv(trades_file, index=False)

            logger.info(f"История сделок сохранена в {trades_file}")

            return results_dir

        return ""