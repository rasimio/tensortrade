"""
Модуль для сбора данных с биржи Binance.
"""
import os
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np

from binance.client import Client
from binance.exceptions import BinanceAPIException

logger = logging.getLogger(__name__)


class BinanceDataCollector:
    """
    Класс для сбора исторических и текущих данных с биржи Binance.
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            api_secret: Optional[str] = None,
            testnet: bool = False
    ):
        """
        Инициализирует коллектор данных Binance.

        Args:
            api_key: API ключ Binance (опционально)
            api_secret: API секрет Binance (опционально)
            testnet: Использовать ли тестовую сеть Binance
        """
        self.api_key = api_key or os.environ.get('BINANCE_API_KEY')
        self.api_secret = api_secret or os.environ.get('BINANCE_API_SECRET')
        self.testnet = testnet

        self.client = Client(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=testnet
        )

        logger.info(f"Initialized Binance data collector (testnet: {testnet})")

    def get_historical_klines(
            self,
            symbol: str,
            interval: str,
            start_date: Union[str, datetime],
            end_date: Optional[Union[str, datetime]] = None,
            limit: int = 1000,
            save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Получает исторические свечные данные с Binance.

        Args:
            symbol: Торговая пара (например, 'BTCUSDT')
            interval: Интервал свечей ('1m', '5m', '15m', '1h', '4h', '1d', и т.д.)
            start_date: Начальная дата сбора данных
            end_date: Конечная дата сбора данных (по умолчанию - текущее время)
            limit: Максимальное количество свечей за один запрос
            save_path: Путь для сохранения данных (опционально)

        Returns:
            DataFrame с историческими свечными данными
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')

        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

        # Преобразование в строку для API
        start_str = str(int(start_date.timestamp() * 1000))
        end_str = str(int(end_date.timestamp() * 1000))

        logger.info(f"Fetching historical klines for {symbol} at {interval} from {start_date} to {end_date}")

        # Данные для хранения результатов
        all_klines = []

        # Получение данных с разбивкой по страницам для преодоления ограничений API
        current_start = start_str
        while True:
            try:
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=int(current_start),
                    endTime=int(end_str),
                    limit=limit
                )

                if not klines:
                    break

                all_klines.extend(klines)

                # Обновление стартового времени для следующего запроса
                current_start = str(klines[-1][0] + 1)

                # Если получили меньше, чем лимит - значит, достигли конца данных
                if len(klines) < limit:
                    break

                # Пауза для предотвращения превышения лимитов API
                time.sleep(0.5)

            except BinanceAPIException as e:
                logger.error(f"Binance API error: {e}")
                break

        # Преобразование данных в DataFrame
        if all_klines:
            df = pd.DataFrame(all_klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            # Преобразование типов данных
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

            numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                               'quote_asset_volume', 'taker_buy_base_asset_volume',
                               'taker_buy_quote_asset_volume']

            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

            # Установка временной метки в качестве индекса
            df.set_index('timestamp', inplace=True)

            # Удаление ненужных столбцов
            df.drop('ignore', axis=1, inplace=True)

            logger.info(f"Retrieved {len(df)} records for {symbol}")

            # Сохранение данных в файл, если указан путь
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                df.to_csv(save_path)
                logger.info(f"Saved data to {save_path}")

            return df

        logger.warning(f"No data retrieved for {symbol}")
        return pd.DataFrame()

    def get_recent_trades(
            self,
            symbol: str,
            limit: int = 1000
    ) -> pd.DataFrame:
        """
        Получает последние сделки по торговой паре.

        Args:
            symbol: Торговая пара (например, 'BTCUSDT')
            limit: Количество сделок для получения

        Returns:
            DataFrame с данными о последних сделках
        """
        try:
            trades = self.client.get_recent_trades(symbol=symbol, limit=limit)

            df = pd.DataFrame(trades)

            # Преобразование типов данных
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            numeric_columns = ['price', 'qty', 'quoteQty']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

            # Установка временной метки в качестве индекса
            df.set_index('time', inplace=True)

            logger.info(f"Retrieved {len(df)} recent trades for {symbol}")

            return df

        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            return pd.DataFrame()

    def get_ticker_price(self, symbol: str) -> float:
        """
        Получает текущую цену для торговой пары.

        Args:
            symbol: Торговая пара (например, 'BTCUSDT')

        Returns:
            Текущая цена
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            return 0.0

    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict:
        """
        Получает информацию о бирже или конкретной торговой паре.

        Args:
            symbol: Торговая пара (опционально)

        Returns:
            Словарь с информацией о бирже или торговой паре
        """
        try:
            if symbol:
                exchange_info = self.client.get_exchange_info(symbol=symbol)
            else:
                exchange_info = self.client.get_exchange_info()
            return exchange_info
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            return {}

    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Получает информацию о конкретной торговой паре.

        Args:
            symbol: Торговая пара (например, 'BTCUSDT')

        Returns:
            Словарь с информацией о торговой паре
        """
        try:
            exchange_info = self.client.get_exchange_info(symbol=symbol)
            if 'symbols' in exchange_info and exchange_info['symbols']:
                return exchange_info['symbols'][0]
            return {}
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            return {}