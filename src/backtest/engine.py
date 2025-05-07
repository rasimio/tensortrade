"""
Движок бэктестинга для оценки эффективности торговых стратегий.
"""
import logging
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import json
import os

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Движок для проведения бэктестирования торговых стратегий.
    """

    def __init__(
            self,
            initial_balance: float = 10000.0,
            fee: float = 0.001,
            slippage: float = 0.0,
            position_size: float = 1.0,
            stop_loss: Optional[float] = None,
            take_profit: Optional[float] = None,
            leverage: float = 1.0,
            margin_call_threshold: float = 0.2,
            allow_short: bool = False
    ):
        """
        Инициализирует движок бэктестирования.

        Args:
            initial_balance: Начальный баланс
            fee: Комиссия за сделку (например, 0.001 = 0.1%)
            slippage: Проскальзывание (например, 0.001 = 0.1%)
            position_size: Размер позиции (доля от баланса)
            stop_loss: Уровень стоп-лосса (например, 0.02 = 2%)
            take_profit: Уровень тейк-профита (например, 0.05 = 5%)
            leverage: Плечо (например, 1.0 = без плеча, 2.0 = 2x плечо)
            margin_call_threshold: Порог для маржин-колла (например, 0.2 = 20% маржи)
            allow_short: Разрешены ли короткие позиции
        """
        self.initial_balance = initial_balance
        self.fee = fee
        self.slippage = slippage
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.leverage = leverage
        self.margin_call_threshold = margin_call_threshold
        self.allow_short = allow_short

        # Метрики бэктестирования
        self.metrics = {}

        # Результаты бэктестирования
        self.backtest_results = None

        # Логгер
        self.logger = logging.getLogger(__name__)

        logger.info(f"Инициализирован движок бэктестирования с начальным балансом {initial_balance}")

    def run_backtest(
            self,
            data: pd.DataFrame,
            model: Optional[Union[BaseModel, Callable]] = None,
            signal_column: Optional[str] = None,
            price_column: str = 'close',
            timestamp_column: Optional[str] = None,
            feature_columns: Optional[List[str]] = None,
            stop_loss_column: Optional[str] = None,
            take_profit_column: Optional[str] = None,
            position_size_column: Optional[str] = None,
            custom_logic: Optional[Callable] = None
    ) -> Dict:
        """
        Выполняет бэктестирование на основе исторических данных.

        Args:
            data: DataFrame с историческими данными
            model: Модель для генерации сигналов (опционально)
            signal_column: Колонка с сигналами (опционально)
            price_column: Колонка с ценами
            timestamp_column: Колонка с временными метками
            feature_columns: Список колонок с признаками для модели
            stop_loss_column: Колонка с уровнями стоп-лосс (опционально)
            take_profit_column: Колонка с уровнями тейк-профит (опционально)
            position_size_column: Колонка с размерами позиций (опционально)
            custom_logic: Пользовательская функция для генерации сигналов (опционально)

        Returns:
            Словарь с результатами бэктестирования
        """
        # Проверяем наличие необходимых колонок
        if price_column not in data.columns:
            raise ValueError(f"Колонка с ценами '{price_column}' отсутствует в данных")

        # Создаем копию данных для бэктестирования
        backtest_data = data.copy()

        # Если заданы временные метки, устанавливаем их в качестве индекса
        if timestamp_column and timestamp_column in backtest_data.columns:
            backtest_data.set_index(timestamp_column, inplace=True)

        # Генерируем сигналы, если задана модель
        if model is not None and not signal_column:
            logger.info(f"Генерация сигналов с помощью модели")

            if hasattr(model, 'predict_from_dataframe'):
                # Если модель поддерживает прогнозирование из DataFrame
                backtest_data = model.predict_from_dataframe(
                    backtest_data,
                    feature_columns=feature_columns
                )
                signal_column = 'hybrid_signal'  # Для гибридной модели
            else:
                # Если модель поддерживает только predict
                if feature_columns is None:
                    raise ValueError("Для использования модели необходимо задать список колонок с признаками")

                # Подготавливаем признаки
                X = backtest_data[feature_columns].values

                # Генерируем сигналы
                signals = model.predict(X)

                # Добавляем сигналы в DataFrame
                backtest_data['model_signal'] = 0
                backtest_data.loc[backtest_data.index[len(backtest_data) - len(signals):], 'model_signal'] = signals

                signal_column = 'model_signal'

        # Используем пользовательскую функцию для генерации сигналов
        elif custom_logic is not None and not signal_column:
            logger.info(f"Генерация сигналов с помощью пользовательской функции")

            # Вызываем пользовательскую функцию
            backtest_data = custom_logic(backtest_data)

            # Проверяем, что функция вернула DataFrame с сигналами
            if 'signal' not in backtest_data.columns:
                raise ValueError("Пользовательская функция должна добавить колонку 'signal' в DataFrame")

            signal_column = 'signal'

        # Проверяем наличие колонки с сигналами
        if not signal_column or signal_column not in backtest_data.columns:
            raise ValueError(f"Колонка с сигналами '{signal_column}' отсутствует в данных")

        # Инициализируем переменные для отслеживания состояния
        balance = self.initial_balance
        position = 0.0
        entry_price = 0.0
        trades = []

        # Добавляем колонки для отслеживания баланса и позиции
        backtest_data['balance'] = self.initial_balance
        backtest_data['position'] = 0.0
        backtest_data['position_value'] = 0.0
        backtest_data['equity'] = self.initial_balance

        # Проходим по данным и симулируем торговлю
        for i in range(1, len(backtest_data)):
            prev_idx = i - 1

            # Получаем текущую цену и сигнал
            current_price = backtest_data.iloc[i][price_column]
            signal = backtest_data.iloc[i][signal_column]

            # Текущий баланс и стоимость позиции
            current_balance = backtest_data.iloc[prev_idx]['balance']
            current_position_value = position * current_price

            # Получаем индивидуальные уровни стоп-лосс и тейк-профит, если заданы соответствующие колонки
            current_stop_loss = None
            if stop_loss_column and stop_loss_column in backtest_data.columns:
                current_stop_loss = backtest_data.iloc[i][stop_loss_column]
            else:
                current_stop_loss = self.stop_loss

            current_take_profit = None
            if take_profit_column and take_profit_column in backtest_data.columns:
                current_take_profit = backtest_data.iloc[i][take_profit_column]
            else:
                current_take_profit = self.take_profit

            # Получаем индивидуальный размер позиции, если задана соответствующая колонка
            current_position_size = None
            if position_size_column and position_size_column in backtest_data.columns:
                current_position_size = backtest_data.iloc[i][position_size_column]
            else:
                current_position_size = self.position_size

            # Проверяем стоп-лосс и тейк-профит, если есть длинная позиция
            if position > 0 and (current_stop_loss is not None or current_take_profit is not None):
                price_change_pct = (current_price - entry_price) / entry_price

                # Срабатывание стоп-лосса
                if current_stop_loss is not None and price_change_pct <= -current_stop_loss:
                    # Учитываем проскальзывание
                    effective_price = current_price * (1 - self.slippage)

                    # Закрываем позицию
                    sale_value = position * effective_price * (1 - self.fee)
                    current_balance += sale_value

                    # Записываем сделку
                    trades.append({
                        'type': 'stop_loss',
                        'entry_time': backtest_data.index[prev_idx - 1] if hasattr(backtest_data.index[0],
                                                                                   'strftime') else str(
                            backtest_data.index[prev_idx - 1]),
                        'exit_time': backtest_data.index[i] if hasattr(backtest_data.index[0], 'strftime') else str(
                            backtest_data.index[i]),
                        'entry_price': entry_price,
                        'exit_price': effective_price,
                        'position': position,
                        'profit_loss': sale_value - (position * entry_price),
                        'profit_loss_pct': price_change_pct * 100
                    })

                    # Обнуляем позицию
                    position = 0.0
                    entry_price = 0.0

                # Срабатывание тейк-профита
                elif current_take_profit is not None and price_change_pct >= current_take_profit:
                    # Учитываем проскальзывание
                    effective_price = current_price * (1 - self.slippage)

                    # Закрываем позицию
                    sale_value = position * effective_price * (1 - self.fee)
                    current_balance += sale_value

                    # Записываем сделку
                    trades.append({
                        'type': 'take_profit',
                        'entry_time': backtest_data.index[prev_idx - 1] if hasattr(backtest_data.index[0],
                                                                                   'strftime') else str(
                            backtest_data.index[prev_idx - 1]),
                        'exit_time': backtest_data.index[i] if hasattr(backtest_data.index[0], 'strftime') else str(
                            backtest_data.index[i]),
                        'entry_price': entry_price,
                        'exit_price': effective_price,
                        'position': position,
                        'profit_loss': sale_value - (position * entry_price),
                        'profit_loss_pct': price_change_pct * 100
                    })

                    # Обнуляем позицию
                    position = 0.0
                    entry_price = 0.0

            # Проверяем стоп-лосс и тейк-профит, если есть короткая позиция
            elif position < 0 and self.allow_short and (
                    current_stop_loss is not None or current_take_profit is not None):
                price_change_pct = (entry_price - current_price) / entry_price

                # Срабатывание стоп-лосса для короткой позиции
                if current_stop_loss is not None and price_change_pct <= -current_stop_loss:
                    # Учитываем проскальзывание
                    effective_price = current_price * (1 + self.slippage)

                    # Закрываем позицию
                    buy_value = abs(position) * effective_price * (1 + self.fee)
                    current_balance -= buy_value

                    # Записываем сделку
                    trades.append({
                        'type': 'stop_loss_short',
                        'entry_time': backtest_data.index[prev_idx - 1] if hasattr(backtest_data.index[0],
                                                                                   'strftime') else str(
                            backtest_data.index[prev_idx - 1]),
                        'exit_time': backtest_data.index[i] if hasattr(backtest_data.index[0], 'strftime') else str(
                            backtest_data.index[i]),
                        'entry_price': entry_price,
                        'exit_price': effective_price,
                        'position': position,
                        'profit_loss': (entry_price * abs(position)) - buy_value,
                        'profit_loss_pct': price_change_pct * 100
                    })

                    # Обнуляем позицию
                    position = 0.0
                    entry_price = 0.0

                # Срабатывание тейк-профита для короткой позиции
                elif current_take_profit is not None and price_change_pct >= current_take_profit:
                    # Учитываем проскальзывание
                    effective_price = current_price * (1 + self.slippage)

                    # Закрываем позицию
                    buy_value = abs(position) * effective_price * (1 + self.fee)
                    current_balance -= buy_value

                    # Записываем сделку
                    trades.append({
                        'type': 'take_profit_short',
                        'entry_time': backtest_data.index[prev_idx - 1] if hasattr(backtest_data.index[0],
                                                                                   'strftime') else str(
                            backtest_data.index[prev_idx - 1]),
                        'exit_time': backtest_data.index[i] if hasattr(backtest_data.index[0], 'strftime') else str(
                            backtest_data.index[i]),
                        'entry_price': entry_price,
                        'exit_price': effective_price,
                        'position': position,
                        'profit_loss': (entry_price * abs(position)) - buy_value,
                        'profit_loss_pct': price_change_pct * 100
                    })

                    # Обнуляем позицию
                    position = 0.0
                    entry_price = 0.0

            # Проверка на маржин-колл для плеча
            if self.leverage > 1.0 and position != 0:
                # Рассчитываем текущую маржу
                if position > 0:
                    # Для длинной позиции
                    entry_value = position * entry_price
                    current_value = position * current_price
                    unrealized_pnl = current_value - entry_value
                    margin = (current_balance + unrealized_pnl) / (entry_value * self.leverage)
                else:
                    # Для короткой позиции
                    entry_value = abs(position) * entry_price
                    current_value = abs(position) * current_price
                    unrealized_pnl = entry_value - current_value
                    margin = (current_balance + unrealized_pnl) / (entry_value * self.leverage)

                # Проверяем маржин-колл
                if margin <= self.margin_call_threshold:
                    logger.warning(f"Маржин-колл на шаге {i} с маржой {margin:.4f}")

                    # Ликвидируем позицию
                    if position > 0:
                        # Учитываем проскальзывание
                        effective_price = current_price * (1 - self.slippage)

                        # Закрываем позицию
                        sale_value = position * effective_price * (1 - self.fee)
                        current_balance += sale_value
                    else:
                        # Учитываем проскальзывание
                        effective_price = current_price * (1 + self.slippage)

                        # Закрываем позицию
                        buy_value = abs(position) * effective_price * (1 + self.fee)
                        current_balance -= buy_value

                    # Записываем сделку
                    trades.append({
                        'type': 'margin_call',
                        'entry_time': backtest_data.index[prev_idx - 1] if hasattr(backtest_data.index[0],
                                                                                   'strftime') else str(
                            backtest_data.index[prev_idx - 1]),
                        'exit_time': backtest_data.index[i] if hasattr(backtest_data.index[0], 'strftime') else str(
                            backtest_data.index[i]),
                        'entry_price': entry_price,
                        'exit_price': effective_price,
                        'position': position,
                        'profit_loss': sale_value - (position * entry_price) if position > 0 else (entry_price * abs(position)) - buy_value,
                        'profit_loss_pct': ((effective_price - entry_price) / entry_price * 100) if position > 0 else ((entry_price - effective_price) / entry_price * 100)
                    })

                    # Обнуляем позицию
                    position = 0.0
                    entry_price = 0.0

            # Сигнал на покупку
            if signal == 1 and position <= 0:
                # Если есть короткая позиция, закрываем ее сначала
                if position < 0:
                    # Учитываем проскальзывание
                    effective_price = current_price * (1 + self.slippage)

                    # Закрываем короткую позицию
                    buy_value = abs(position) * effective_price * (1 + self.fee)
                    current_balance -= buy_value

                    # Рассчитываем прибыль/убыток
                    price_change_pct = (entry_price - effective_price) / entry_price
                    profit_loss = (entry_price * abs(position)) - buy_value

                    # Записываем сделку
                    trades.append({
                        'type': 'close_short',
                        'entry_time': backtest_data.index[prev_idx - 1] if hasattr(backtest_data.index[0],
                                                                                   'strftime') else str(
                            backtest_data.index[prev_idx - 1]),
                        'exit_time': backtest_data.index[i] if hasattr(backtest_data.index[0], 'strftime') else str(
                            backtest_data.index[i]),
                        'entry_price': entry_price,
                        'exit_price': effective_price,
                        'position': position,
                        'profit_loss': profit_loss,
                        'profit_loss_pct': price_change_pct * 100
                    })

                    # Обнуляем позицию
                    position = 0.0

                # Открываем длинную позицию
                # Учитываем проскальзывание
                effective_price = current_price * (1 + self.slippage)

                # Рассчитываем размер позиции
                position_value = current_balance * current_position_size

                # При использовании плеча
                if self.leverage > 1.0:
                    position_value *= self.leverage

                # Учитываем комиссию
                effective_position_value = position_value / (1 + self.fee)
                position = effective_position_value / effective_price

                # Обновляем баланс
                current_balance -= position_value / self.leverage

                # Запоминаем цену входа
                entry_price = effective_price

            # Сигнал на продажу
            elif signal == 2 and (position >= 0 or self.allow_short):
                # Если есть длинная позиция, закрываем ее сначала
                if position > 0:
                    # Учитываем проскальзывание
                    effective_price = current_price * (1 - self.slippage)

                    # Закрываем длинную позицию
                    sale_value = position * effective_price * (1 - self.fee)
                    current_balance += sale_value

                    # Рассчитываем прибыль/убыток
                    price_change_pct = (effective_price - entry_price) / entry_price
                    profit_loss = sale_value - (position * entry_price)

                    # Записываем сделку
                    trades.append({
                        'type': 'close_long',
                        'entry_time': backtest_data.index[prev_idx - 1] if hasattr(backtest_data.index[0],
                                                                                   'strftime') else str(
                            backtest_data.index[prev_idx - 1]),
                        'exit_time': backtest_data.index[i] if hasattr(backtest_data.index[0], 'strftime') else str(
                            backtest_data.index[i]),
                        'entry_price': entry_price,
                        'exit_price': effective_price,
                        'position': position,
                        'profit_loss': profit_loss,
                        'profit_loss_pct': price_change_pct * 100
                    })

                    # Обнуляем позицию
                    position = 0.0

                # Открываем короткую позицию, если разрешено
                if self.allow_short:
                    # Учитываем проскальзывание
                    effective_price = current_price * (1 - self.slippage)

                    # Рассчитываем размер позиции
                    position_value = current_balance * current_position_size

                    # При использовании плеча
                    if self.leverage > 1.0:
                        position_value *= self.leverage

                    # Учитываем комиссию
                    effective_position_value = position_value / (1 + self.fee)
                    position = -effective_position_value / effective_price

                    # Обновляем баланс (для короткой позиции баланс не меняется сразу)

                    # Запоминаем цену входа
                    entry_price = effective_price

            # Обновляем данные
            backtest_data.loc[backtest_data.index[i], 'balance'] = current_balance
            backtest_data.loc[backtest_data.index[i], 'position'] = position
            backtest_data.loc[backtest_data.index[i], 'position_value'] = position * current_price
            backtest_data.loc[backtest_data.index[i], 'equity'] = current_balance + (position * current_price)

        # Закрываем позицию в конце периода, если она открыта
        if position != 0:
            # Получаем последнюю цену
            last_price = backtest_data.iloc[-1][price_column]

            if position > 0:
                # Учитываем проскальзывание
                effective_price = last_price * (1 - self.slippage)

                # Закрываем длинную позицию
                sale_value = position * effective_price * (1 - self.fee)
                final_balance = backtest_data.iloc[-1]['balance'] + sale_value

                # Рассчитываем прибыль/убыток
                price_change_pct = (effective_price - entry_price) / entry_price
                profit_loss = sale_value - (position * entry_price)
            else:
                # Учитываем проскальзывание
                effective_price = last_price * (1 + self.slippage)

                # Закрываем короткую позицию
                buy_value = abs(position) * effective_price * (1 + self.fee)
                final_balance = backtest_data.iloc[-1]['balance'] - buy_value

                # Рассчитываем прибыль/убыток
                price_change_pct = (entry_price - effective_price) / entry_price
                profit_loss = (entry_price * abs(position)) - buy_value

            # Записываем сделку
            trades.append({
                'type': 'end_of_period',
                'entry_time': backtest_data.index[-2] if hasattr(backtest_data.index[0], 'strftime') else str(
                    backtest_data.index[-2]),
                'exit_time': backtest_data.index[-1] if hasattr(backtest_data.index[0], 'strftime') else str(
                    backtest_data.index[-1]),
                'entry_price': entry_price,
                'exit_price': effective_price,
                'position': position,
                'profit_loss': profit_loss,
                'profit_loss_pct': price_change_pct * 100
            })

            # Обновляем последнюю строку
            backtest_data.loc[backtest_data.index[-1], 'balance'] = final_balance
            backtest_data.loc[backtest_data.index[-1], 'position'] = 0.0
            backtest_data.loc[backtest_data.index[-1], 'position_value'] = 0.0
            backtest_data.loc[backtest_data.index[-1], 'equity'] = final_balance

        # Рассчитываем метрики бэктестирования
        metrics = self._calculate_metrics(backtest_data, trades)

        # Создаем результаты бэктестирования
        backtest_results = {
            'data': backtest_data,
            'trades': trades,
            'metrics': metrics
        }

        # Сохраняем результаты
        self.backtest_results = backtest_results
        self.metrics = metrics

        # Логируем результаты
        self._log_backtest_results(metrics)

        return backtest_results

    def _calculate_metrics(self, backtest_data: pd.DataFrame, trades: List[Dict]) -> Dict:
        """
        Рассчитывает метрики бэктестирования.

        Args:
            backtest_data: DataFrame с результатами бэктестирования
            trades: Список сделок

        Returns:
            Словарь с метриками
        """
        # Базовые метрики
        initial_equity = self.initial_balance
        final_equity = backtest_data.iloc[-1]['equity']
        total_return = final_equity / initial_equity - 1

        # Рассчитываем максимальную просадку
        backtest_data['drawdown'] = 1 - backtest_data['equity'] / backtest_data['equity'].cummax()
        max_drawdown = backtest_data['drawdown'].max()

        # Метрики для сделок
        if trades:
            # Прибыльные и убыточные сделки
            profitable_trades = [t for t in trades if t['profit_loss'] > 0]
            losing_trades = [t for t in trades if t['profit_loss'] <= 0]

            # Общая статистика
            total_trades = len(trades)
            profitable_trades_count = len(profitable_trades)
            win_rate = profitable_trades_count / total_trades if total_trades > 0 else 0

            # Средняя прибыль и убыток
            avg_profit = sum(t['profit_loss'] for t in profitable_trades) / len(
                profitable_trades) if profitable_trades else 0
            avg_loss = sum(t['profit_loss'] for t in losing_trades) / len(losing_trades) if losing_trades else 0

            # Отношение среднего выигрыша к среднему проигрышу
            profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')

            # Ожидаемая прибыль
            expected_profit = (win_rate * avg_profit) + ((1 - win_rate) * avg_loss)

            # Фактор восстановления
            recovery_factor = total_return / max_drawdown if max_drawdown > 0 else float('inf')

            # Годовая доходность (предполагаем 252 торговых дня в году)
            if hasattr(backtest_data.index[0], 'strftime'):
                # Если индекс - datetime
                start_date = backtest_data.index[0]
                end_date = backtest_data.index[-1]
                days = (end_date - start_date).days
            else:
                # Если индекс - не datetime, предполагаем, что каждая строка - это день
                days = len(backtest_data)

            annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0

            # Коэффициент Шарпа (если есть данные о доходности)
            returns = backtest_data['equity'].pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(
                returns) > 0 and returns.std() > 0 else 0

            # Средняя длительность сделок
            if hasattr(backtest_data.index[0], 'strftime'):
                # Если индекс - datetime
                avg_trade_duration = 0
                for trade in trades:
                    entry_time = datetime.strptime(trade['entry_time'], '%Y-%m-%d %H:%M:%S') if isinstance(
                        trade['entry_time'], str) else trade['entry_time']
                    exit_time = datetime.strptime(trade['exit_time'], '%Y-%m-%d %H:%M:%S') if isinstance(
                        trade['exit_time'], str) else trade['exit_time']
                    duration = (exit_time - entry_time).total_seconds() / 3600  # в часах
                    avg_trade_duration += duration

                avg_trade_duration /= len(trades) if trades else 1
            else:
                # Если индекс - не datetime, предполагаем, что каждая строка - это время
                avg_trade_duration = 0

            # Максимальное количество последовательных потерь
            consecutive_losses = 0
            max_consecutive_losses = 0

            for trade in trades:
                if trade['profit_loss'] <= 0:
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                else:
                    consecutive_losses = 0

            # Создаем метрики
            metrics = {
                'initial_equity': initial_equity,
                'final_equity': final_equity,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'annual_return': annual_return,
                'annual_return_pct': annual_return * 100,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown * 100,
                'total_trades': total_trades,
                'profitable_trades': profitable_trades_count,
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'win_rate_pct': win_rate * 100,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_loss_ratio': profit_loss_ratio,
                'expected_profit': expected_profit,
                'recovery_factor': recovery_factor,
                'sharpe_ratio': sharpe_ratio,
                'avg_trade_duration': avg_trade_duration,
                'max_consecutive_losses': max_consecutive_losses
            }
        else:
            # Если сделок не было
            metrics = {
                'initial_equity': initial_equity,
                'final_equity': final_equity,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'annual_return': 0,
                'annual_return_pct': 0,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown * 100,
                'total_trades': 0,
                'profitable_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'win_rate_pct': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_loss_ratio': 0,
                'expected_profit': 0,
                'recovery_factor': 0,
                'sharpe_ratio': 0,
                'avg_trade_duration': 0,
                'max_consecutive_losses': 0
            }

        return metrics

    def _log_backtest_results(self, metrics: Dict) -> None:
        """
        Логирует результаты бэктестирования.

        Args:
            metrics: Словарь с метриками
        """
        logger.info(f"Результаты бэктестирования:")
        logger.info(f"Начальный капитал: {metrics['initial_equity']:.2f}")
        logger.info(f"Конечный капитал: {metrics['final_equity']:.2f}")
        logger.info(f"Общая доходность: {metrics['total_return_pct']:.2f}%")
        logger.info(f"Годовая доходность: {metrics['annual_return_pct']:.2f}%")
        logger.info(f"Максимальная просадка: {metrics['max_drawdown_pct']:.2f}%")
        logger.info(f"Всего сделок: {metrics['total_trades']}")

        if metrics['total_trades'] > 0:
            logger.info(f"Прибыльных сделок: {metrics['profitable_trades']} ({metrics['win_rate_pct']:.2f}%)")
            logger.info(f"Убыточных сделок: {metrics['losing_trades']} ({100 - metrics['win_rate_pct']:.2f}%)")
            logger.info(f"Отношение прибыли к убытку: {metrics['profit_loss_ratio']:.2f}")
            logger.info(f"Фактор восстановления: {metrics['recovery_factor']:.2f}")
            logger.info(f"Коэффициент Шарпа: {metrics['sharpe_ratio']:.2f}")

    def plot_equity_curve(self, filepath: Optional[str] = None) -> None:
        """
        Строит график кривой капитала.

        Args:
            filepath: Путь для сохранения графика (опционально)
        """
        if self.backtest_results is None:
            logger.warning("Нет результатов бэктестирования для построения графика")
            return

        plt.figure(figsize=(12, 6))

        backtest_data = self.backtest_results['data']

        # Построение графика капитала
        plt.plot(backtest_data['equity'], label='Капитал', color='blue')

        # Построение просадок
        drawdown = backtest_data['drawdown']
        plt.fill_between(backtest_data.index, 0, drawdown * 100, alpha=0.3, color='red', label='Просадка %')

        # Настройка графика
        plt.title('Кривая капитала и просадки')
        plt.xlabel('Время')
        plt.ylabel('Капитал')

        # Создаем вторую ось Y для просадок
        ax2 = plt.gca().twinx()
        ax2.set_ylabel('Просадка %')
        ax2.set_ylim(0, max(drawdown * 100) * 1.1)

        # Форматирование оси X для временных меток
        if hasattr(backtest_data.index[0], 'strftime'):
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
            plt.gcf().autofmt_xdate()

        plt.grid(True, alpha=0.3)
        plt.legend()

        # Сохранение графика, если задан путь
        if filepath:
            plt.savefig(filepath)
            logger.info(f"График сохранен в {filepath}")

        plt.tight_layout()
        plt.show()

    def plot_trades(self, price_column: str = 'close', filepath: Optional[str] = None) -> None:
        """
        Строит график цены с отмеченными сделками.

        Args:
            price_column: Колонка с ценами
            filepath: Путь для сохранения графика (опционально)
        """
        if self.backtest_results is None:
            logger.warning("Нет результатов бэктестирования для построения графика")
            return

        backtest_data = self.backtest_results['data']
        trades = self.backtest_results['trades']

        plt.figure(figsize=(12, 6))

        # Построение графика цены
        plt.plot(backtest_data[price_column], label='Цена', color='blue')

        # Отметка сделок на графике
        for trade in trades:
            # Преобразуем время входа и выхода в индекс, если они представлены в виде строк
            if isinstance(trade['entry_time'], str) and hasattr(backtest_data.index[0], 'strftime'):
                entry_time = datetime.strptime(trade['entry_time'], '%Y-%m-%d %H:%M:%S')
            else:
                entry_time = trade['entry_time']

            if isinstance(trade['exit_time'], str) and hasattr(backtest_data.index[0], 'strftime'):
                exit_time = datetime.strptime(trade['exit_time'], '%Y-%m-%d %H:%M:%S')
            else:
                exit_time = trade['exit_time']

            # Определяем индексы для входа и выхода
            entry_idx = backtest_data.index.get_indexer([entry_time], method='nearest')[0]
            exit_idx = backtest_data.index.get_indexer([exit_time], method='nearest')[0]

            # Получаем цены входа и выхода
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']

            # Определяем цвет маркера в зависимости от типа сделки и ее результата
            if trade['profit_loss'] > 0:
                color = 'green'
            else:
                color = 'red'

            # Отмечаем точки входа и выхода
            plt.scatter(backtest_data.index[entry_idx], entry_price, color=color, marker='^', s=100)
            plt.scatter(backtest_data.index[exit_idx], exit_price, color=color, marker='v', s=100)

            # Соединяем точки входа и выхода
            plt.plot([backtest_data.index[entry_idx], backtest_data.index[exit_idx]],
                     [entry_price, exit_price], color=color, linestyle='--', alpha=0.5)

        # Настройка графика
        plt.title('График цены с отмеченными сделками')
        plt.xlabel('Время')
        plt.ylabel('Цена')

        # Форматирование оси X для временных меток
        if hasattr(backtest_data.index[0], 'strftime'):
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
            plt.gcf().autofmt_xdate()

        plt.grid(True, alpha=0.3)

        # Добавляем легенду для типов сделок
        plt.scatter([], [], color='green', marker='^', s=100, label='Покупка (прибыль)')
        plt.scatter([], [], color='red', marker='^', s=100, label='Покупка (убыток)')
        plt.scatter([], [], color='green', marker='v', s=100, label='Продажа (прибыль)')
        plt.scatter([], [], color='red', marker='v', s=100, label='Продажа (убыток)')

        plt.legend()

        # Сохранение графика, если задан путь
        if filepath:
            plt.savefig(filepath)
            logger.info(f"График сохранен в {filepath}")

        plt.tight_layout()
        plt.show()

    def plot_monthly_returns(self, filepath: Optional[str] = None) -> None:
        """
        Строит график месячной доходности.

        Args:
            filepath: Путь для сохранения графика (опционально)
        """
        if self.backtest_results is None:
            logger.warning("Нет результатов бэктестирования для построения графика")
            return

        backtest_data = self.backtest_results['data']

        # Проверяем, что индекс - временная метка
        if not hasattr(backtest_data.index[0], 'strftime'):
            logger.warning(
                "Индекс данных не является временной меткой, невозможно построить график месячной доходности")
            return

        # Рассчитываем месячную доходность
        monthly_returns = backtest_data['equity'].resample('M').last().pct_change().dropna()

        plt.figure(figsize=(12, 6))

        # Построение графика месячной доходности
        monthly_returns_pct = monthly_returns * 100
        colors = ['green' if r >= 0 else 'red' for r in monthly_returns_pct]

        plt.bar(monthly_returns_pct.index, monthly_returns_pct, color=colors)

        # Добавляем линию средней месячной доходности
        plt.axhline(y=monthly_returns_pct.mean(), color='blue', linestyle='--',
                    label=f'Средняя ({monthly_returns_pct.mean():.2f}%)')

        # Настройка графика
        plt.title('Месячная доходность')
        plt.xlabel('Месяц')
        plt.ylabel('Доходность (%)')

        # Форматирование оси X
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate()

        plt.grid(True, alpha=0.3)
        plt.legend()

        # Сохранение графика, если задан путь
        if filepath:
            plt.savefig(filepath)
            logger.info(f"График сохранен в {filepath}")

        plt.tight_layout()
        plt.show()

    def plot_drawdown_distribution(self, filepath: Optional[str] = None) -> None:
        """
        Строит распределение просадок.

        Args:
            filepath: Путь для сохранения графика (опционально)
        """
        if self.backtest_results is None:
            logger.warning("Нет результатов бэктестирования для построения графика")
            return

        backtest_data = self.backtest_results['data']

        plt.figure(figsize=(12, 6))

        # Построение гистограммы просадок
        drawdown = backtest_data['drawdown'] * 100
        plt.hist(drawdown, bins=50, alpha=0.7, color='red')

        # Настройка графика
        plt.title('Распределение просадок')
        plt.xlabel('Просадка (%)')
        plt.ylabel('Частота')

        plt.grid(True, alpha=0.3)

        # Добавляем вертикальную линию для максимальной просадки
        max_drawdown = drawdown.max()
        plt.axvline(x=max_drawdown, color='black', linestyle='--', label=f'Макс. просадка ({max_drawdown:.2f}%)')

        plt.legend()

        # Сохранение графика, если задан путь
        if filepath:
            plt.savefig(filepath)
            logger.info(f"График сохранен в {filepath}")

        plt.tight_layout()
        plt.show()

    def plot_trade_distribution(self, filepath: Optional[str] = None) -> None:
        """
        Строит распределение прибыли по сделкам.

        Args:
            filepath: Путь для сохранения графика (опционально)
        """
        if self.backtest_results is None:
            logger.warning("Нет результатов бэктестирования для построения графика")
            return

        trades = self.backtest_results['trades']

        if not trades:
            logger.warning("Нет сделок для построения графика")
            return

        plt.figure(figsize=(12, 6))

        # Построение гистограммы прибыли по сделкам
        profit_loss_pct = [trade['profit_loss_pct'] for trade in trades]

        # Определяем цвета для прибыльных и убыточных сделок
        colors = ['green' if pct >= 0 else 'red' for pct in profit_loss_pct]

        plt.hist(profit_loss_pct, bins=20, alpha=0.7, color=colors)

        # Настройка графика
        plt.title('Распределение прибыли по сделкам')
        plt.xlabel('Прибыль (%)')
        plt.ylabel('Количество сделок')

        plt.grid(True, alpha=0.3)

        # Добавляем вертикальную линию для средней прибыли
        avg_profit = sum(profit_loss_pct) / len(profit_loss_pct)
        plt.axvline(x=avg_profit, color='blue', linestyle='--', label=f'Средняя прибыль ({avg_profit:.2f}%)')

        plt.legend()

        # Сохранение графика, если задан путь
        if filepath:
            plt.savefig(filepath)
            logger.info(f"График сохранен в {filepath}")

        plt.tight_layout()
        plt.show()

    def save_report(self, filepath: str) -> None:
        """
        Сохраняет отчет о результатах бэктестирования в JSON файл.

        Args:
            filepath: Путь для сохранения отчета
        """
        if self.backtest_results is None:
            logger.warning("Нет результатов бэктестирования для сохранения отчета")
            return

        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Создаем отчет
        report = {
            'metrics': self.metrics,
            'trades': self.backtest_results['trades']
        }

        # Сохраняем отчет в JSON файл
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=4)

        logger.info(f"Отчет сохранен в {filepath}")

    def compare_with_baseline(
            self,
            baseline_data: pd.DataFrame,
            price_column: str = 'close',
            plot: bool = True,
            filepath: Optional[str] = None
    ) -> Dict:
        """
        Сравнивает результаты бэктестирования с базовой стратегией.

        Args:
            baseline_data: DataFrame с данными для базовой стратегии
            price_column: Колонка с ценами
            plot: Строить ли график
            filepath: Путь для сохранения графика (опционально)

        Returns:
            Словарь со сравнительными метриками
        """
        if self.backtest_results is None:
            logger.warning("Нет результатов бэктестирования для сравнения")
            return {}

        # Рассчитываем доходность базовой стратегии (buy and hold)
        first_price = baseline_data[price_column].iloc[0]
        last_price = baseline_data[price_column].iloc[-1]
        baseline_return = last_price / first_price - 1

        # Рассчитываем годовую доходность базовой стратегии
        if hasattr(baseline_data.index[0], 'strftime'):
            # Если индекс - datetime
            start_date = baseline_data.index[0]
            end_date = baseline_data.index[-1]
            days = (end_date - start_date).days
        else:
            # Если индекс - не datetime, предполагаем, что каждая строка - это день
            days = len(baseline_data)

        baseline_annual_return = (1 + baseline_return) ** (252 / days) - 1 if days > 0 else 0

        # Рассчитываем максимальную просадку базовой стратегии
        baseline_data['cummax'] = baseline_data[price_column].cummax()
        baseline_data['drawdown'] = 1 - baseline_data[price_column] / baseline_data['cummax']
        baseline_max_drawdown = baseline_data['drawdown'].max()

        # Создаем сравнительные метрики
        comparison = {
            'strategy_return': self.metrics['total_return'],
            'strategy_return_pct': self.metrics['total_return_pct'],
            'strategy_annual_return': self.metrics['annual_return'],
            'strategy_annual_return_pct': self.metrics['annual_return_pct'],
            'strategy_max_drawdown': self.metrics['max_drawdown'],
            'strategy_max_drawdown_pct': self.metrics['max_drawdown_pct'],
            'strategy_sharpe_ratio': self.metrics['sharpe_ratio'],
            'baseline_return': baseline_return,
            'baseline_return_pct': baseline_return * 100,
            'baseline_annual_return': baseline_annual_return,
            'baseline_annual_return_pct': baseline_annual_return * 100,
            'baseline_max_drawdown': baseline_max_drawdown,
            'baseline_max_drawdown_pct': baseline_max_drawdown * 100,
            'outperformance': self.metrics['total_return'] - baseline_return,
            'outperformance_pct': (self.metrics['total_return'] - baseline_return) * 100,
            'annual_outperformance': self.metrics['annual_return'] - baseline_annual_return,
            'annual_outperformance_pct': (self.metrics['annual_return'] - baseline_annual_return) * 100
        }

        # Логируем сравнительные метрики
        logger.info(f"Сравнение с базовой стратегией:")
        logger.info(
            f"Доходность стратегии: {comparison['strategy_return_pct']:.2f}% (годовая: {comparison['strategy_annual_return_pct']:.2f}%)")
        logger.info(
            f"Доходность базовой стратегии: {comparison['baseline_return_pct']:.2f}% (годовая: {comparison['baseline_annual_return_pct']:.2f}%)")
        logger.info(
            f"Превышение доходности: {comparison['outperformance_pct']:.2f}% (годовое: {comparison['annual_outperformance_pct']:.2f}%)")
        logger.info(f"Макс. просадка стратегии: {comparison['strategy_max_drawdown_pct']:.2f}%")
        logger.info(f"Макс. просадка базовой стратегии: {comparison['baseline_max_drawdown_pct']:.2f}%")

        # Строим график сравнения
        if plot:
            plt.figure(figsize=(12, 6))

            # Нормализуем стоимость портфеля стратегии и базовой стратегии к начальному значению
            strategy_equity = self.backtest_results['data']['equity'] / self.initial_balance
            baseline_equity = baseline_data[price_column] / first_price

            # Построение графика
            plt.plot(strategy_equity.index, strategy_equity, label='Стратегия', color='blue')
            plt.plot(baseline_data.index, baseline_equity, label='Buy & Hold', color='green')

            # Настройка графика
            plt.title('Сравнение стратегии с базовой стратегией')
            plt.xlabel('Время')
            plt.ylabel('Относительная стоимость портфеля')

            # Форматирование оси X для временных меток
            if hasattr(baseline_data.index[0], 'strftime'):
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                plt.gcf().autofmt_xdate()

            plt.grid(True, alpha=0.3)
            plt.legend()

            # Сохранение графика, если задан путь
            if filepath:
                plt.savefig(filepath)
                logger.info(f"График сохранен в {filepath}")

            plt.tight_layout()
            plt.show()

        return comparison