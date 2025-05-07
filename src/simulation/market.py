"""
Модуль для симуляции рынка и тестирования торговых стратегий.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
import websockets
import asyncio
import json
import time
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class MarketSimulator:
    """
    Класс для симуляции рынка и тестирования торговых стратегий.
    """

    def __init__(
            self,
            data: Optional[pd.DataFrame] = None,
            initial_balance: float = 10000.0,
            fee: float = 0.001,
            slippage: float = 0.0005,
            position_size: float = 1.0,
            stop_loss: Optional[float] = None,
            take_profit: Optional[float] = None,
            leverage: float = 1.0,
            allow_short: bool = False,
            price_column: str = 'close',
            timestamp_column: Optional[str] = None,
            use_websocket: bool = False,
            ws_url: str = 'ws://localhost:8765',
            ws_symbol: str = 'BTCUSDT'
    ):
        """
        Инициализирует симулятор рынка.

        Args:
            data: DataFrame с историческими данными (опционально)
            initial_balance: Начальный баланс
            fee: Комиссия за сделку (например, 0.001 = 0.1%)
            slippage: Проскальзывание (например, 0.0005 = 0.05%)
            position_size: Размер позиции (доля от баланса)
            stop_loss: Уровень стоп-лосса (например, 0.02 = 2%)
            take_profit: Уровень тейк-профита (например, 0.05 = 5%)
            leverage: Плечо (например, 1.0 = без плеча, 2.0 = 2x плечо)
            allow_short: Разрешены ли короткие позиции
            price_column: Колонка с ценами
            timestamp_column: Колонка с временными метками
            use_websocket: Использовать ли WebSocket для получения данных
            ws_url: URL для WebSocket соединения
            ws_symbol: Символ для WebSocket
        """
        self.data = data
        self.initial_balance = initial_balance
        self.fee = fee
        self.slippage = slippage
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.leverage = leverage
        self.allow_short = allow_short
        self.price_column = price_column
        self.timestamp_column = timestamp_column
        self.use_websocket = use_websocket
        self.ws_url = ws_url
        self.ws_symbol = ws_symbol

        # Состояние симуляции
        self.reset()

        # WebSocket клиент
        self.ws = None
        self.ws_running = False
        self.ws_data_queue = asyncio.Queue()

        # Обработчики событий
        self.on_tick_handlers = []
        self.on_trade_handlers = []

        # Текущие данные
        self.current_data = {}

        # Thread pool для асинхронных операций
        self.executor = ThreadPoolExecutor(max_workers=5)

        logger.info(f"Инициализирован симулятор рынка с начальным балансом {initial_balance}")

    def reset(self) -> None:
        """
        Сбрасывает состояние симуляции.
        """
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.trades = []
        self.equity_history = []
        self.current_index = 0
        self.is_running = False

        # Инициализируем торговую статистику
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'max_drawdown': 0.0
        }

        if self.data is not None:
            self.equity_history = [{'timestamp': self.data.index[0] if hasattr(self.data.index[0], 'strftime') else 0,
                                    'equity': self.initial_balance}]

        logger.info("Состояние симуляции сброшено")

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

    def _handle_trade(self, trade_data: Dict) -> None:
        """
        Обрабатывает событие "сделка".

        Args:
            trade_data: Данные сделки
        """
        # Вызываем все обработчики события "сделка"
        for handler in self.on_trade_handlers:
            try:
                handler(trade_data)
            except Exception as e:
                logger.error(f"Ошибка в обработчике события 'сделка': {e}")

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

        while self.ws_running:
            try:
                async with websockets.connect(self.ws_url) as websocket:
                    self.ws = websocket

                    # Подписываемся на поток данных
                    await websocket.send(json.dumps({
                        'type': 'subscribe',
                        'symbol': self.ws_symbol,
                        'interval': '1m'  # По умолчанию используем 1-минутный интервал
                    }))

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

            # Обрабатываем разные типы сообщений
            if data.get('type') == 'kline':
                # Получены данные свечи
                kline_data = data.get('data', {})

                # Преобразуем данные в формат тика
                tick_data = {
                    'timestamp': datetime.fromtimestamp(kline_data.get('startTime', 0) / 1000),
                    'symbol': kline_data.get('symbol', self.ws_symbol),
                    'open': kline_data.get('open', 0.0),
                    'high': kline_data.get('high', 0.0),
                    'low': kline_data.get('low', 0.0),
                    'close': kline_data.get('close', 0.0),
                    'volume': kline_data.get('volume', 0.0)
                }

                # Добавляем данные в очередь
                await self.ws_data_queue.put(tick_data)

                # Если симуляция запущена, обрабатываем тик
                if self.is_running:
                    self._handle_tick(tick_data)

            elif data.get('type') == 'subscription':
                # Подтверждение подписки
                logger.info(f"Подписка на {data.get('symbol')} подтверждена")

            elif data.get('type') == 'error':
                # Ошибка
                logger.error(f"Получена ошибка от WebSocket сервера: {data.get('message')}")

            elif data.get('type') == 'pong':
                # Ответ на пинг
                latency = int(time.time() * 1000) - data.get('timestamp', 0)
                logger.debug(f"Задержка WebSocket: {latency} мс")

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
            logger.warning("Симуляция не запущена")
            return {}

        # Если уже есть короткая позиция, закрываем ее сначала
        if self.position < 0:
            self.close_position(price)

        # Получаем текущую цену, если не задана
        if price is None:
            if self.use_websocket:
                price = self.current_data.get('close', 0.0)
            else:
                price = self.data.iloc[self.current_index][self.price_column]

        # Получаем текущее время
        if self.use_websocket:
            timestamp = self.current_data.get('timestamp', datetime.now())
        else:
            if self.timestamp_column is not None and self.timestamp_column in self.data.columns:
                timestamp = self.data.iloc[self.current_index][self.timestamp_column]
            elif hasattr(self.data.index[0], 'strftime'):
                timestamp = self.data.index[self.current_index]
            else:
                timestamp = datetime.now()

        # Учитываем проскальзывание
        effective_price = price * (1 + self.slippage)

        # Получаем размер позиции, если не задан
        if size is None:
            size = self.position_size

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

        # Добавляем сделку в историю
        self.trades.append(trade_info)

        # Обновляем статистику
        self.stats['total_trades'] += 1

        # Вызываем обработчик события "сделка"
        self._handle_trade(trade_info)

        # Обновляем историю капитала
        equity = self.balance + (self.position * effective_price)
        self.equity_history.append({'timestamp': timestamp, 'equity': equity})

        logger.info(
            f"Открыта длинная позиция: цена={effective_price:.2f}, размер={position_amount:.6f}, стоимость={position_value:.2f}")

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
            logger.warning("Симуляция не запущена")
            return {}

        # Получаем текущую цену, если не задана
        if price is None:
            if self.use_websocket:
                price = self.current_data.get('close', 0.0)
            else:
                price = self.data.iloc[self.current_index][self.price_column]

        # Получаем текущее время
        if self.use_websocket:
            timestamp = self.current_data.get('timestamp', datetime.now())
        else:
            if self.timestamp_column is not None and self.timestamp_column in self.data.columns:
                timestamp = self.data.iloc[self.current_index][self.timestamp_column]
            elif hasattr(self.data.index[0], 'strftime'):
                timestamp = self.data.index[self.current_index]
            else:
                timestamp = datetime.now()

        # Если есть длинная позиция, закрываем ее
        if self.position > 0:
            return self.close_position(price)

        # Если короткие позиции не разрешены, выходим
        if not self.allow_short:
            logger.warning("Короткие позиции не разрешены")
            return {}

        # Учитываем проскальзывание (для продажи - отрицательное)
        effective_price = price * (1 - self.slippage)

        # Получаем размер позиции, если не задан
        if size is None:
            size = self.position_size

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
            'position': position_amount,
            'value': position_value,
            'fee': position_value * self.fee,
            'balance': self.balance
        }

        # Добавляем сделку в историю
        self.trades.append(trade_info)

        # Обновляем статистику
        self.stats['total_trades'] += 1

        # Вызываем обработчик события "сделка"
        self._handle_trade(trade_info)

        # Обновляем историю капитала
        equity = self.balance - (self.position * effective_price)
        self.equity_history.append({'timestamp': timestamp, 'equity': equity})

        logger.info(
            f"Открыта короткая позиция: цена={effective_price:.2f}, размер={position_amount:.6f}, стоимость={position_value:.2f}")

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
            logger.warning("Симуляция не запущена")
            return {}

        # Если нет открытой позиции, выходим
        if self.position == 0:
            logger.warning("Нет открытой позиции для закрытия")
            return {}

        # Получаем текущую цену, если не задана
        if price is None:
            if self.use_websocket:
                price = self.current_data.get('close', 0.0)
            else:
                price = self.data.iloc[self.current_index][self.price_column]

        # Получаем текущее время
        if self.use_websocket:
            timestamp = self.current_data.get('timestamp', datetime.now())
        else:
            if self.timestamp_column is not None and self.timestamp_column in self.data.columns:
                timestamp = self.data.iloc[self.current_index][self.timestamp_column]
            elif hasattr(self.data.index[0], 'strftime'):
                timestamp = self.data.index[self.current_index]
            else:
                timestamp = datetime.now()

        # Учитываем проскальзывание
        if self.position > 0:
            # Для закрытия длинной позиции - отрицательное
            effective_price = price * (1 - self.slippage)
        else:
            # Для закрытия короткой позиции - положительное
            effective_price = price * (1 + self.slippage)

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

        # Обновляем статистику
        if profit_loss > 0:
            self.stats['winning_trades'] += 1
            self.stats['total_profit'] += profit_loss
        else:
            self.stats['losing_trades'] += 1
            self.stats['total_loss'] += abs(profit_loss)

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

        # Добавляем сделку в историю
        self.trades.append(trade_info)

        # Сбрасываем позицию
        self.position = 0
        self.entry_price = 0

        # Вызываем обработчик события "сделка"
        self._handle_trade(trade_info)

        # Обновляем историю капитала
        self.equity_history.append({'timestamp': timestamp, 'equity': self.balance})

        # Обновляем максимальную просадку
        self._update_max_drawdown()

        logger.info(f"Закрыта позиция: цена={effective_price:.2f}, прибыль/убыток={profit_loss:.2f}")

        return trade_info

    def _update_max_drawdown(self) -> None:
        """
        Обновляет максимальную просадку.
        """
        if len(self.equity_history) < 2:
            return

        # Получаем список значений капитала
        equity_values = [entry['equity'] for entry in self.equity_history]

        # Рассчитываем максимальную просадку
        max_equity = equity_values[0]
        max_drawdown = 0

        for equity in equity_values:
            if equity > max_equity:
                max_equity = equity
            else:
                drawdown = (max_equity - equity) / max_equity
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

        self.stats['max_drawdown'] = max_drawdown

    def _check_stop_loss_take_profit(self, price: float) -> bool:
        """
        Проверяет условия стоп-лосса и тейк-профита.

        Args:
            price: Текущая цена

        Returns:
            True, если была закрыта позиция, иначе False
        """
        if self.position == 0:
            return False

        # Проверяем стоп-лосс
        if self.stop_loss is not None:
            if self.position > 0:
                # Для длинной позиции
                stop_loss_price = self.entry_price * (1 - self.stop_loss)
                if price <= stop_loss_price:
                    logger.info(f"Сработал стоп-лосс: цена={price:.2f}, стоп-лосс={stop_loss_price:.2f}")
                    self.close_position(price)
                    return True
            else:
                # Для короткой позиции
                stop_loss_price = self.entry_price * (1 + self.stop_loss)
                if price >= stop_loss_price:
                    logger.info(f"Сработал стоп-лосс: цена={price:.2f}, стоп-лосс={stop_loss_price:.2f}")
                    self.close_position(price)
                    return True

        # Проверяем тейк-профит
        if self.take_profit is not None:
            if self.position > 0:
                # Для длинной позиции
                take_profit_price = self.entry_price * (1 + self.take_profit)
                if price >= take_profit_price:
                    logger.info(f"Сработал тейк-профит: цена={price:.2f}, тейк-профит={take_profit_price:.2f}")
                    self.close_position(price)
                    return True
            else:
                # Для короткой позиции
                take_profit_price = self.entry_price * (1 - self.take_profit)
                if price <= take_profit_price:
                    logger.info(f"Сработал тейк-профит: цена={price:.2f}, тейк-профит={take_profit_price:.2f}")
                    self.close_position(price)
                    return True

        return False

    def run_simulation(
            self,
            strategy_func: Callable,
            start_index: int = 0,
            end_index: Optional[int] = None,
            step: int = 1
    ) -> Dict:
        """
        Запускает симуляцию с заданной стратегией.

        Args:
            strategy_func: Функция стратегии, принимающая данные и симулятор
            start_index: Индекс начала симуляции
            end_index: Индекс конца симуляции (если None, используется весь доступный диапазон)
            step: Шаг по индексам

        Returns:
            Словарь с результатами симуляции
        """
        if self.use_websocket:
            logger.warning("Для симуляции с WebSocket используйте метод run_live_simulation")
            return {}

        if self.data is None:
            logger.error("Нет данных для симуляции")
            return {}

        # Сбрасываем состояние симуляции
        self.reset()

        # Устанавливаем индексы
        self.current_index = start_index
        if end_index is None:
            end_index = len(self.data) - 1

        # Запускаем симуляцию
        self.is_running = True

        logger.info(f"Запуск симуляции с индекса {start_index} до {end_index} с шагом {step}")

        try:
            # Проходим по всем данным
            for i in range(start_index, end_index + 1, step):
                self.current_index = i

                # Получаем текущие данные
                current_row = self.data.iloc[i]

                # Создаем словарь с данными текущего тика
                tick_data = {
                    'timestamp': self.data.index[i] if hasattr(self.data.index[0], 'strftime') else i,
                    'open': current_row.get('open', current_row[self.price_column]),
                    'high': current_row.get('high', current_row[self.price_column]),
                    'low': current_row.get('low', current_row[self.price_column]),
                    'close': current_row[self.price_column],
                    'volume': current_row.get('volume', 0.0)
                }

                # Проверяем стоп-лосс и тейк-профит
                self._check_stop_loss_take_profit(tick_data['close'])

                # Вызываем обработчик события "тик"
                self._handle_tick(tick_data)

                # Вызываем функцию стратегии
                strategy_func(tick_data, self)

                # Обновляем историю капитала
                if self.position == 0:
                    equity = self.balance
                else:
                    equity = self.balance + (self.position * tick_data['close'])

                self.equity_history.append({'timestamp': tick_data['timestamp'], 'equity': equity})

                # Обновляем максимальную просадку
                self._update_max_drawdown()

        except Exception as e:
            logger.error(f"Ошибка при выполнении симуляции: {e}")

        finally:
            # Завершаем симуляцию
            self.is_running = False

            # Закрываем позицию, если она открыта
            if self.position != 0:
                self.close_position()

            # Рассчитываем итоговые метрики
            self._calculate_final_metrics()

            logger.info(
                f"Симуляция завершена: баланс={self.balance:.2f}, прибыль={self.balance - self.initial_balance:.2f}")

        # Возвращаем результаты симуляции
        return self._get_simulation_results()

    async def run_live_simulation(
            self,
            strategy_func: Callable,
            duration: Optional[int] = None
    ) -> Dict:
        """
        Запускает симуляцию в режиме реального времени с использованием WebSocket.

        Args:
            strategy_func: Функция стратегии, принимающая данные и симулятор
            duration: Продолжительность симуляции в секундах (если None, выполняется до остановки)

        Returns:
            Словарь с результатами симуляции
        """
        if not self.use_websocket:
            logger.warning("WebSocket не активирован, используйте use_websocket=True")
            return {}

        # Сбрасываем состояние симуляции
        self.reset()

        # Подключаемся к WebSocket, если еще не подключены
        if not self.ws_running:
            self.connect_websocket()

            # Ждем подключения
            await asyncio.sleep(2)

        # Запускаем симуляцию
        self.is_running = True

        logger.info(f"Запуск симуляции в режиме реального времени")

        try:
            # Устанавливаем время начала
            start_time = time.time()

            # Выполняем симуляцию
            while self.is_running:
                # Проверяем продолжительность
                if duration is not None and time.time() - start_time > duration:
                    logger.info(f"Достигнута заданная продолжительность: {duration} секунд")
                    break

                # Получаем данные из очереди (с таймаутом)
                try:
                    tick_data = await asyncio.wait_for(self.ws_data_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Если нет данных, продолжаем ожидание
                    continue

                # Проверяем стоп-лосс и тейк-профит
                self._check_stop_loss_take_profit(tick_data['close'])

                # Вызываем функцию стратегии
                strategy_func(tick_data, self)

                # Обновляем историю капитала
                if self.position == 0:
                    equity = self.balance
                else:
                    equity = self.balance + (self.position * tick_data['close'])

                self.equity_history.append({'timestamp': tick_data['timestamp'], 'equity': equity})

                # Обновляем максимальную просадку
                self._update_max_drawdown()

        except Exception as e:
            logger.error(f"Ошибка при выполнении симуляции в режиме реального времени: {e}")

        finally:
            # Завершаем симуляцию
            self.is_running = False

            # Закрываем позицию, если она открыта
            if self.position != 0:
                self.close_position()

            # Рассчитываем итоговые метрики
            self._calculate_final_metrics()

            logger.info(
                f"Симуляция завершена: баланс={self.balance:.2f}, прибыль={self.balance - self.initial_balance:.2f}")

        # Возвращаем результаты симуляции
        return self._get_simulation_results()

    def stop_simulation(self) -> None:
        """
        Останавливает текущую симуляцию.
        """
        if not self.is_running:
            return

        self.is_running = False

        logger.info("Симуляция остановлена")

    def _calculate_final_metrics(self) -> None:
        """
        Рассчитывает итоговые метрики.
        """
        # Получаем список значений капитала
        equity_values = [entry['equity'] for entry in self.equity_history]

        # Рассчитываем доходность
        if len(equity_values) > 1:
            total_return = (equity_values[-1] / equity_values[0]) - 1
        else:
            total_return = 0

        # Добавляем метрики в статистику
        self.stats['initial_balance'] = self.initial_balance
        self.stats['final_balance'] = self.balance
        self.stats['total_return'] = total_return

        # Добавляем годовую доходность, если есть временные метки
        if len(self.equity_history) > 1 and hasattr(self.equity_history[0]['timestamp'], 'strftime'):
            start_time = self.equity_history[0]['timestamp']
            end_time = self.equity_history[-1]['timestamp']

            days = (end_time - start_time).days
            if days > 0:
                annualized_return = (1 + total_return) ** (365 / days) - 1
                self.stats['annualized_return'] = annualized_return

    def _get_simulation_results(self) -> Dict:
        """
        Формирует словарь с результатами симуляции.

        Returns:
            Словарь с результатами симуляции
        """
        return {
            'stats': self.stats,
            'trades': self.trades,
            'equity_history': self.equity_history,
            'balance': self.balance,
            'position': self.position
        }

    def get_equity_curve(self) -> pd.DataFrame:
        """
        Возвращает кривую капитала в виде DataFrame.

        Returns:
            DataFrame с историей капитала
        """
        if not self.equity_history:
            return pd.DataFrame()

        # Создаем DataFrame из истории капитала
        equity_df = pd.DataFrame(self.equity_history)

        # Устанавливаем временную метку в качестве индекса
        if 'timestamp' in equity_df.columns:
            equity_df.set_index('timestamp', inplace=True)

        return equity_df

    def get_trade_history(self) -> pd.DataFrame:
        """
        Возвращает историю сделок в виде DataFrame.

        Returns:
            DataFrame с историей сделок
        """
        if not self.trades:
            return pd.DataFrame()

        # Создаем DataFrame из истории сделок
        trades_df = pd.DataFrame(self.trades)

        # Устанавливаем временную метку в качестве индекса
        if 'timestamp' in trades_df.columns:
            trades_df.set_index('timestamp', inplace=True)

        return trades_df

    def load_data_from_csv(self, file_path: str, timestamp_column: Optional[str] = None) -> None:
        """
        Загружает данные из CSV-файла.

        Args:
            file_path: Путь к CSV-файлу
            timestamp_column: Имя колонки с временными метками
        """
        try:
            # Загружаем данные
            self.data = pd.read_csv(file_path)

            # Если задан timestamp_column, преобразуем его в datetime и устанавливаем как индекс
            if timestamp_column and timestamp_column in self.data.columns:
                self.data[timestamp_column] = pd.to_datetime(self.data[timestamp_column])
                self.data.set_index(timestamp_column, inplace=True)
                self.timestamp_column = None  # Уже используется как индекс
            else:
                self.timestamp_column = timestamp_column

            logger.info(f"Загружены данные из {file_path}: {len(self.data)} записей")

        except Exception as e:
            logger.error(f"Ошибка при загрузке данных из {file_path}: {e}")


def create_strategy_from_model(model: BaseModel, threshold: float = 0.5) -> Callable:
    """
    Создает функцию стратегии на основе модели.

    Args:
        model: Модель для прогнозирования
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
        # Получаем признаки для модели
        features = np.array([
            data['open'], data['high'], data['low'], data['close'], data['volume']
        ]).reshape(1, -1)

        # Делаем прогноз
        prediction = model.predict(features)

        # Интерпретируем прогноз
        if prediction > threshold:
            # Сигнал на покупку
            if simulator.position <= 0:
                simulator.buy()
        elif prediction < -threshold:
            # Сигнал на продажу
            if simulator.position >= 0:
                simulator.sell()

    return strategy


def simple_moving_average_strategy(fast_period: int = 10, slow_period: int = 30) -> Callable:
    """
    Создает стратегию на основе пересечения скользящих средних.

    Args:
        fast_period: Период быстрой скользящей средней
        slow_period: Период медленной скользящей средней

    Returns:
        Функция стратегии
    """
    # Инициализируем буферы для цен
    price_buffer = []

    def strategy(data: Dict, simulator: MarketSimulator) -> None:
        """
        Стратегия на основе пересечения скользящих средних.

        Args:
            data: Данные тика
            simulator: Симулятор рынка
        """
        nonlocal price_buffer

        # Добавляем цену в буфер
        price_buffer.append(data['close'])

        # Если буфер недостаточно заполнен, выходим
        if len(price_buffer) < slow_period:
            return

        # Обрезаем буфер
        if len(price_buffer) > slow_period:
            price_buffer = price_buffer[-slow_period:]

        # Рассчитываем скользящие средние
        fast_ma = sum(price_buffer[-fast_period:]) / fast_period
        slow_ma = sum(price_buffer) / slow_period

        # Предыдущие скользящие средние
        if len(price_buffer) > 1:
            prev_fast_ma = sum(price_buffer[-fast_period - 1:-1]) / fast_period
            prev_slow_ma = sum(price_buffer[:-1]) / slow_period

            # Проверяем пересечение снизу вверх (сигнал на покупку)
            if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma:
                if simulator.position <= 0:
                    simulator.buy()

            # Проверяем пересечение сверху вниз (сигнал на продажу)
            elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma:
                if simulator.position >= 0:
                    simulator.sell()

    return strategy