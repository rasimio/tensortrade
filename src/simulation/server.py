"""
Модуль WSS сервера для симуляции торговли в режиме реального времени.
"""
import asyncio
import json
import logging
import websockets
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
import random
import time

logger = logging.getLogger(__name__)


class MarketDataSimulator:
    """
    Класс для симуляции данных рынка.
    """

    def __init__(
            self,
            symbol: str = 'BTCUSDT',
            interval: str = '1m',
            start_price: float = 50000.0,
            volatility: float = 0.002,
            drift: float = 0.0001,
            historical_data: Optional[pd.DataFrame] = None
    ):
        """
        Инициализирует симулятор данных рынка.

        Args:
            symbol: Символ торговой пары
            interval: Интервал свечей
            start_price: Начальная цена
            volatility: Волатильность (стандартное отклонение дневных изменений)
            drift: Дрейф (среднее дневных изменений)
            historical_data: Исторические данные (опционально)
        """
        self.symbol = symbol
        self.interval = interval
        self.volatility = volatility
        self.drift = drift

        self.historical_data = historical_data
        self.use_historical_data = historical_data is not None

        if not self.use_historical_data:
            self.current_price = start_price
            self.current_time = datetime.now()
        else:
            self.data_index = 0
            self.current_time = self.historical_data.index[0]
            self.current_price = self.historical_data.iloc[0]['close']

        # Интервал в секундах
        self.interval_seconds = self._interval_to_seconds(interval)

        logger.info(f"Инициализирован симулятор данных рынка для {symbol} с интервалом {interval}")

    def _interval_to_seconds(self, interval: str) -> int:
        """
        Преобразует интервал в секунды.

        Args:
            interval: Интервал в формате Binance ('1m', '5m', '1h', '1d', и т.д.)

        Returns:
            Количество секунд
        """
        interval_map = {
            's': 1,
            'm': 60,
            'h': 3600,
            'd': 86400,
            'w': 604800
        }

        unit = interval[-1]
        value = int(interval[:-1])

        if unit in interval_map:
            return value * interval_map[unit]
        else:
            raise ValueError(f"Неподдерживаемый интервал: {interval}")

    def _generate_random_price(self) -> float:
        """
        Генерирует случайное изменение цены.

        Returns:
            Новая цена
        """
        # Геометрическое броуновское движение
        random_change = np.random.normal(0, 1)
        price_change = self.current_price * (self.drift + self.volatility * random_change)

        new_price = self.current_price + price_change

        # Цена не может быть отрицательной
        if new_price <= 0:
            new_price = self.current_price * 0.9  # 10% снижение

        return new_price

    def generate_kline(self) -> Dict:
        """
        Генерирует данные свечи.

        Returns:
            Словарь с данными свечи
        """
        if self.use_historical_data:
            # Если используем исторические данные, берем их
            if self.data_index >= len(self.historical_data):
                # Если достигли конца данных, начинаем сначала
                self.data_index = 0
                logger.info("Достигнут конец исторических данных, начинаем сначала")

            current_data = self.historical_data.iloc[self.data_index]

            kline_data = {
                'symbol': self.symbol,
                'interval': self.interval,
                'startTime': int(self.historical_data.index[self.data_index].timestamp() * 1000),
                'endTime': int((self.historical_data.index[self.data_index] + timedelta(
                    seconds=self.interval_seconds)).timestamp() * 1000),
                'open': float(current_data['open']),
                'high': float(current_data['high']),
                'low': float(current_data['low']),
                'close': float(current_data['close']),
                'volume': float(current_data['volume']),
                'trades': random.randint(100, 1000),
                'isFinal': True
            }

            # Обновляем текущую цену и время
            self.current_price = current_data['close']
            self.current_time = self.historical_data.index[self.data_index]

            # Увеличиваем индекс
            self.data_index += 1

            return kline_data
        else:
            # Если не используем исторические данные, генерируем случайные
            new_price = self._generate_random_price()

            # Генерируем случайные OHLC
            price_range = abs(new_price - self.current_price) * 2

            # Определяем направление
            if new_price > self.current_price:
                # Восходящая свеча
                open_price = self.current_price
                close_price = new_price
                high_price = new_price + price_range * random.uniform(0, 0.1)
                low_price = self.current_price - price_range * random.uniform(0, 0.1)
            else:
                # Нисходящая свеча
                open_price = self.current_price
                close_price = new_price
                high_price = self.current_price + price_range * random.uniform(0, 0.1)
                low_price = new_price - price_range * random.uniform(0, 0.1)

            # Убеждаемся, что high >= max(open, close) и low <= min(open, close)
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)

            # Генерируем случайный объем
            volume = self.current_price * random.uniform(1, 10)

            # Время начала и конца свечи
            start_time = self.current_time
            end_time = start_time + timedelta(seconds=self.interval_seconds)

            kline_data = {
                'symbol': self.symbol,
                'interval': self.interval,
                'startTime': int(start_time.timestamp() * 1000),
                'endTime': int(end_time.timestamp() * 1000),
                'open': float(open_price),
                'high': float(high_price),
                'low': float(low_price),
                'close': float(close_price),
                'volume': float(volume),
                'trades': random.randint(100, 1000),
                'isFinal': True
            }

            # Обновляем текущую цену и время
            self.current_price = close_price
            self.current_time = end_time

            return kline_data


class WSServer:
    """
    WebSocket сервер для симуляции торговли в режиме реального времени.
    """

    def __init__(
            self,
            host: str = 'localhost',
            port: int = 8765,
            symbol: str = 'BTCUSDT',
            interval: str = '1m',
            historical_data: Optional[pd.DataFrame] = None,
            simulator_params: Optional[Dict] = None
    ):
        """
        Инициализирует WebSocket сервер.

        Args:
            host: Хост для сервера
            port: Порт для сервера
            symbol: Символ торговой пары
            interval: Интервал свечей
            historical_data: Исторические данные (опционально)
            simulator_params: Параметры для симулятора рынка (опционально)
        """
        self.host = host
        self.port = port
        self.symbol = symbol
        self.interval = interval

        # Создаем симулятор рынка
        if simulator_params is None:
            simulator_params = {}

        self.market_simulator = MarketDataSimulator(
            symbol=symbol,
            interval=interval,
            historical_data=historical_data,
            **simulator_params
        )

        # Набор подключенных клиентов
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()

        # Интервал в секундах
        self.interval_seconds = self.market_simulator._interval_to_seconds(interval)

        # Флаг для остановки сервера
        self.running = False

        logger.info(f"Инициализирован WebSocket сервер на {host}:{port} для {symbol} с интервалом {interval}")

    async def register(self, websocket: websockets.WebSocketServerProtocol) -> None:
        """
        Регистрирует нового клиента.

        Args:
            websocket: WebSocket соединение клиента
        """
        self.connected_clients.add(websocket)
        logger.info(f"Клиент подключен: {websocket.remote_address}")

    async def unregister(self, websocket: websockets.WebSocketServerProtocol) -> None:
        """
        Удаляет клиента при отключении.

        Args:
            websocket: WebSocket соединение клиента
        """
        self.connected_clients.remove(websocket)
        logger.info(f"Клиент отключен: {websocket.remote_address}")

    async def send_to_clients(self, message: Dict) -> None:
        """
        Отправляет сообщение всем подключенным клиентам.

        Args:
            message: Сообщение для отправки
        """
        if not self.connected_clients:
            return

        # Преобразуем сообщение в JSON
        json_message = json.dumps(message)

        # Отправляем сообщение всем клиентам
        await asyncio.gather(
            *[client.send(json_message) for client in self.connected_clients]
        )

    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str) -> None:
        """
        Обрабатывает соединение клиента.

        Args:
            websocket: WebSocket соединение клиента
            path: Путь запроса
        """
        await self.register(websocket)

        try:
            async for message in websocket:
                # Обрабатываем сообщение от клиента
                try:
                    data = json.loads(message)

                    # Обрабатываем разные типы сообщений
                    if data.get('type') == 'subscribe':
                        # Клиент подписывается на поток данных
                        logger.info(
                            f"Клиент подписался на {data.get('symbol', self.symbol)} {data.get('interval', self.interval)}")

                        # Отправляем подтверждение подписки
                        await websocket.send(json.dumps({
                            'type': 'subscription',
                            'symbol': data.get('symbol', self.symbol),
                            'interval': data.get('interval', self.interval),
                            'status': 'success'
                        }))

                    elif data.get('type') == 'unsubscribe':
                        # Клиент отписывается от потока данных
                        logger.info(
                            f"Клиент отписался от {data.get('symbol', self.symbol)} {data.get('interval', self.interval)}")

                        # Отправляем подтверждение отписки
                        await websocket.send(json.dumps({
                            'type': 'unsubscription',
                            'symbol': data.get('symbol', self.symbol),
                            'interval': data.get('interval', self.interval),
                            'status': 'success'
                        }))

                    elif data.get('type') == 'ping':
                        # Клиент отправляет пинг
                        await websocket.send(json.dumps({
                            'type': 'pong',
                            'timestamp': int(time.time() * 1000)
                        }))

                    else:
                        # Неизвестный тип сообщения
                        logger.warning(f"Получено неизвестное сообщение: {data}")

                        # Отправляем ошибку
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': f"Неизвестный тип сообщения: {data.get('type')}"
                        }))

                except json.JSONDecodeError:
                    logger.warning(f"Получено некорректное JSON-сообщение: {message}")

                    # Отправляем ошибку
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': "Некорректное JSON-сообщение"
                    }))

        finally:
            await self.unregister(websocket)

    async def generate_market_data(self) -> None:
        """
        Генерирует и отправляет данные рынка клиентам.
        """
        while self.running:
            # Генерируем данные свечи
            kline_data = self.market_simulator.generate_kline()

            # Создаем сообщение
            message = {
                'type': 'kline',
                'data': kline_data
            }

            # Отправляем сообщение клиентам
            await self.send_to_clients(message)

            # Ждем до следующего интервала
            await asyncio.sleep(self.interval_seconds)

    async def run(self) -> None:
        """
        Запускает WebSocket сервер.
        """
        self.running = True

        # Создаем задачу для генерации данных рынка
        market_data_task = asyncio.create_task(self.generate_market_data())

        # Запускаем сервер
        logger.info(f"Запуск WebSocket сервера на {self.host}:{self.port}")

        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # Ждем вечно

    def start(self) -> None:
        """
        Запускает сервер в блокирующем режиме.
        """
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            logger.info("Сервер остановлен по запросу пользователя")
        finally:
            self.running = False


async def load_historical_data(file_path: str) -> pd.DataFrame:
    """
    Загружает исторические данные из CSV-файла.

    Args:
        file_path: Путь к CSV-файлу

    Returns:
        DataFrame с историческими данными
    """
    df = pd.read_csv(file_path)

    # Преобразуем временные метки
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    return df


async def main():
    """
    Основная функция для запуска сервера.
    """
    import argparse

    # Настройка логгирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='WebSocket сервер для симуляции торговли')
    parser.add_argument('--host', type=str, default='localhost', help='Хост для сервера')
    parser.add_argument('--port', type=int, default=8765, help='Порт для сервера')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Символ торговой пары')
    parser.add_argument('--interval', type=str, default='1m', help='Интервал свечей')
    parser.add_argument('--data-file', type=str, help='Путь к CSV-файлу с историческими данными')
    parser.add_argument('--start-price', type=float, default=50000.0,
                        help='Начальная цена (если не используются исторические данные)')
    parser.add_argument('--volatility', type=float, default=0.002,
                        help='Волатильность (если не используются исторические данные)')
    parser.add_argument('--drift', type=float, default=0.0001, help='Дрейф (если не используются исторические данные)')

    args = parser.parse_args()

    # Загружаем исторические данные, если указан файл
    historical_data = None
    if args.data_file:
        try:
            historical_data = await load_historical_data(args.data_file)
            logger.info(f"Загружены исторические данные из {args.data_file}: {len(historical_data)} записей")
        except Exception as e:
            logger.error(f"Ошибка при загрузке исторических данных: {e}")
            return

    # Создаем и запускаем сервер
    server = WSServer(
        host=args.host,
        port=args.port,
        symbol=args.symbol,
        interval=args.interval,
        historical_data=historical_data,
        simulator_params={
            'start_price': args.start_price,
            'volatility': args.volatility,
            'drift': args.drift
        }
    )

    await server.run()


if __name__ == "__main__":
    asyncio.run(main())