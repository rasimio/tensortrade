"""
Модуль для генерации отчетов и визуализации результатов бэктестирования.
"""
import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta

from src.backtest.metrics import calculate_all_metrics

logger = logging.getLogger(__name__)


class BacktestReport:
    """
    Класс для генерации отчетов и визуализации результатов бэктестирования.
    """

    def __init__(
            self,
            data: pd.DataFrame,
            trades: List[Dict],
            metrics: Dict,
            strategy_name: str = "Strategy",
            output_dir: str = "reports"
    ):
        """
        Инициализирует объект отчета.

        Args:
            data: DataFrame с результатами бэктестирования
            trades: Список сделок
            metrics: Словарь с метриками
            strategy_name: Название стратегии
            output_dir: Директория для сохранения отчетов
        """
        self.data = data
        self.trades = trades
        self.metrics = metrics
        self.strategy_name = strategy_name
        self.output_dir = output_dir

        # Создаем директорию для отчетов, если она не существует
        os.makedirs(output_dir, exist_ok=True)

        # Создаем директорию для графиков
        self.plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    def generate_summary(self) -> pd.DataFrame:
        """
        Генерирует сводку основных метрик.

        Returns:
            DataFrame с основными метриками
        """
        # Выбираем основные метрики для сводки
        summary_metrics = [
            'total_return_pct', 'annual_return_pct', 'max_drawdown_pct',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'win_rate_pct',
            'profit_factor', 'total_trades', 'profitable_trades', 'losing_trades'
        ]

        # Форматированные названия метрик
        metric_names = {
            'total_return_pct': 'Total Return (%)',
            'annual_return_pct': 'Annual Return (%)',
            'max_drawdown_pct': 'Max Drawdown (%)',
            'sharpe_ratio': 'Sharpe Ratio',
            'sortino_ratio': 'Sortino Ratio',
            'calmar_ratio': 'Calmar Ratio',
            'win_rate_pct': 'Win Rate (%)',
            'profit_factor': 'Profit Factor',
            'total_trades': 'Total Trades',
            'profitable_trades': 'Profitable Trades',
            'losing_trades': 'Losing Trades'
        }

        # Создаем DataFrame с метриками
        summary = pd.DataFrame(columns=['Metric', 'Value'])

        for metric in summary_metrics:
            if metric in self.metrics:
                summary = pd.concat([
                    summary,
                    pd.DataFrame({'Metric': [metric_names.get(metric, metric)], 'Value': [self.metrics[metric]]})
                ], ignore_index=True)

        return summary

    def generate_monthly_returns(self) -> pd.DataFrame:
        """
        Генерирует таблицу месячных доходностей.

        Returns:
            DataFrame с месячными доходностями
        """
        # Проверяем, что индекс - временная метка
        if not hasattr(self.data.index[0], 'strftime'):
            logger.warning("Индекс данных не является временной меткой, невозможно рассчитать месячную доходность")
            return pd.DataFrame()

        # Рассчитываем месячную доходность
        monthly_returns = self.data['equity'].resample('M').last().pct_change().dropna() * 100

        # Создаем DataFrame с месячными доходностями
        monthly_df = pd.DataFrame(monthly_returns)
        monthly_df.columns = ['Returns (%)']

        # Форматируем индекс
        monthly_df.index = monthly_df.index.strftime('%Y-%m')

        return monthly_df

    def generate_yearly_returns(self) -> pd.DataFrame:
        """
        Генерирует таблицу годовых доходностей.

        Returns:
            DataFrame с годовыми доходностями
        """
        # Проверяем, что индекс - временная метка
        if not hasattr(self.data.index[0], 'strftime'):
            logger.warning("Индекс данных не является временная метка, невозможно рассчитать годовую доходность")
            return pd.DataFrame()

        # Рассчитываем годовую доходность
        yearly_returns = self.data['equity'].resample('Y').last().pct_change().dropna() * 100

        # Создаем DataFrame с годовыми доходностями
        yearly_df = pd.DataFrame(yearly_returns)
        yearly_df.columns = ['Returns (%)']

        # Форматируем индекс
        yearly_df.index = yearly_df.index.strftime('%Y')

        return yearly_df

    def generate_trade_statistics(self) -> pd.DataFrame:
        """
        Генерирует статистику по сделкам.

        Returns:
            DataFrame со статистикой по сделкам
        """
        if not self.trades:
            logger.warning("Нет сделок для генерации статистики")
            return pd.DataFrame()

        # Создаем DataFrame из сделок
        trades_df = pd.DataFrame(self.trades)

        # Преобразуем временные метки
        if 'entry_time' in trades_df.columns:
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])

        if 'exit_time' in trades_df.columns:
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

        # Добавляем длительность сделок
        if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
            trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / (
                        60 * 60 * 24)  # в днях

        # Выбираем колонки для отображения
        display_columns = [
            'type', 'entry_time', 'exit_time', 'entry_price', 'exit_price',
            'position', 'profit_loss', 'profit_loss_pct'
        ]

        # Фильтруем только существующие колонки
        display_columns = [col for col in display_columns if col in trades_df.columns]

        # Если есть колонка с длительностью, добавляем ее
        if 'duration' in trades_df.columns:
            display_columns.append('duration')

        return trades_df[display_columns]

    def generate_drawdown_periods(self, threshold: float = 0.05) -> pd.DataFrame:
        """
        Генерирует информацию о периодах просадок, превышающих заданный порог.

        Args:
            threshold: Порог просадки (например, 0.05 = 5%)

        Returns:
            DataFrame с информацией о периодах просадок
        """
        # Проверяем, что индекс - временная метка
        if not hasattr(self.data.index[0], 'strftime'):
            logger.warning("Индекс данных не является временной меткой, невозможно рассчитать периоды просадок")
            return pd.DataFrame()

        # Рассчитываем просадку
        drawdown = 1 - self.data['equity'] / self.data['equity'].cummax()

        # Находим периоды просадок, превышающих порог
        is_drawdown = drawdown > threshold

        # Находим начало и конец каждого периода просадки
        # Используем смещение для определения изменений состояния
        state_change = is_drawdown.astype(int).diff().fillna(0).astype(int)

        # Находим индексы начала просадок (смена 0 -> 1)
        drawdown_start_indices = np.where(state_change == 1)[0]

        # Находим индексы конца просадок (смена 1 -> 0)
        drawdown_end_indices = np.where(state_change == -1)[0]

        # Если последняя просадка еще не завершилась, добавляем конец данных
        if len(drawdown_start_indices) > len(drawdown_end_indices):
            drawdown_end_indices = np.append(drawdown_end_indices, len(drawdown) - 1)

        # Создаем список периодов просадок
        drawdown_periods = []

        for i in range(len(drawdown_start_indices)):
            start_idx = drawdown_start_indices[i]
            end_idx = drawdown_end_indices[i]

            # Получаем максимальную просадку в этом периоде
            max_drawdown_in_period = drawdown.iloc[start_idx:end_idx + 1].max()

            # Получаем индекс максимальной просадки
            max_drawdown_idx = drawdown.iloc[start_idx:end_idx + 1].idxmax()

            # Добавляем информацию о периоде просадки
            drawdown_periods.append({
                'start_date': self.data.index[start_idx],
                'end_date': self.data.index[end_idx],
                'duration_days': (self.data.index[end_idx] - self.data.index[start_idx]).days,
                'max_drawdown': max_drawdown_in_period,
                'max_drawdown_date': max_drawdown_idx
            })

        # Создаем DataFrame с периодами просадок
        drawdown_df = pd.DataFrame(drawdown_periods)

        # Форматируем проценты
        drawdown_df['max_drawdown'] = drawdown_df['max_drawdown'] * 100

        # Переименовываем колонки
        drawdown_df.columns = [
            'Start Date', 'End Date', 'Duration (days)', 'Max Drawdown (%)', 'Max Drawdown Date'
        ]

        return drawdown_df

    def plot_equity_curve(self, benchmark_data: Optional[pd.DataFrame] = None, price_column: str = 'close',
                          save: bool = True) -> plt.Figure:
        """
        Строит график кривой капитала.

        Args:
            benchmark_data: DataFrame с данными бенчмарка (опционально)
            price_column: Колонка с ценами в бенчмарке
            save: Сохранять ли график

        Returns:
            Объект Figure
        """
        plt.figure(figsize=(12, 6))

        # Построение графика капитала
        plt.plot(self.data.index, self.data['equity'], label=f'{self.strategy_name} Equity', color='blue')

        # Если задан бенчмарк, добавляем его на график
        if benchmark_data is not None:
            # Нормализуем бенчмарк относительно начального капитала
            first_price = benchmark_data[price_column].iloc[0]
            normalized_benchmark = benchmark_data[price_column] / first_price * self.data['equity'].iloc[0]

            plt.plot(benchmark_data.index, normalized_benchmark, label='Benchmark', color='green', linestyle='--')

        # Построение просадок
        drawdown = 1 - self.data['equity'] / self.data['equity'].cummax()
        plt.fill_between(self.data.index, 0, drawdown * 100, alpha=0.3, color='red', label='Drawdown (%)')

        # Настройка графика
        plt.title(f'Equity Curve - {self.strategy_name}')
        plt.xlabel('Date')
        plt.ylabel('Equity')

        # Создаем вторую ось Y для просадок
        ax2 = plt.gca().twinx()
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_ylim(0, max(drawdown * 100) * 1.1)

        # Форматирование оси X для временных меток
        if hasattr(self.data.index[0], 'strftime'):
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.gcf().autofmt_xdate()

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # Сохранение графика, если нужно
        if save:
            filename = os.path.join(self.plots_dir, 'equity_curve.png')
            plt.savefig(filename)
            logger.info(f"График сохранен в {filename}")

        return plt.gcf()

    def plot_monthly_returns(self, save: bool = True) -> plt.Figure:
        """
        Строит график месячной доходности.

        Args:
            save: Сохранять ли график

        Returns:
            Объект Figure
        """
        # Проверяем, что индекс - временная метка
        if not hasattr(self.data.index[0], 'strftime'):
            logger.warning(
                "Индекс данных не является временной меткой, невозможно построить график месячной доходности")
            return plt.figure()

        # Рассчитываем месячную доходность
        monthly_returns = self.data['equity'].resample('M').last().pct_change().dropna() * 100

        plt.figure(figsize=(12, 6))

        # Построение графика месячной доходности
        colors = ['green' if r >= 0 else 'red' for r in monthly_returns]

        plt.bar(monthly_returns.index, monthly_returns, color=colors)

        # Добавляем линию средней месячной доходности
        plt.axhline(y=monthly_returns.mean(), color='blue', linestyle='--',
                    label=f'Mean ({monthly_returns.mean():.2f}%)')

        # Настройка графика
        plt.title(f'Monthly Returns - {self.strategy_name}')
        plt.xlabel('Month')
        plt.ylabel('Return (%)')

        # Форматирование оси X
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate()

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # Сохранение графика, если нужно
        if save:
            filename = os.path.join(self.plots_dir, 'monthly_returns.png')
            plt.savefig(filename)
            logger.info(f"График сохранен в {filename}")

        return plt.gcf()

    def plot_drawdowns(self, save: bool = True) -> plt.Figure:
        """
        Строит график просадок.

        Args:
            save: Сохранять ли график

        Returns:
            Объект Figure
        """
        plt.figure(figsize=(12, 6))

        # Рассчитываем просадку
        drawdown = 1 - self.data['equity'] / self.data['equity'].cummax()

        # Построение графика просадок
        plt.plot(self.data.index, drawdown * 100, color='red', label='Drawdown (%)')

        # Добавляем горизонтальные линии для различных уровней просадок
        plt.axhline(y=5, color='yellow', linestyle='--', alpha=0.5, label='5%')
        plt.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='10%')
        plt.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='20%')

        # Настройка графика
        plt.title(f'Drawdowns - {self.strategy_name}')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')

        # Форматирование оси X для временных меток
        if hasattr(self.data.index[0], 'strftime'):
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.gcf().autofmt_xdate()

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # Сохранение графика, если нужно
        if save:
            filename = os.path.join(self.plots_dir, 'drawdowns.png')
            plt.savefig(filename)
            logger.info(f"График сохранен в {filename}")

        return plt.gcf()

    def plot_trade_profits(self, save: bool = True) -> plt.Figure:
        """
        Строит график прибыли/убытка по сделкам.

        Args:
            save: Сохранять ли график

        Returns:
            Объект Figure
        """
        if not self.trades:
            logger.warning("Нет сделок для построения графика")
            return plt.figure()

        plt.figure(figsize=(12, 6))

        # Получаем прибыль/убыток по каждой сделке
        trade_profits = [trade.get('profit_loss_pct', 0) for trade in self.trades]

        # Создаем индекс для сделок
        trade_indices = list(range(1, len(trade_profits) + 1))

        # Определяем цвета для прибыльных и убыточных сделок
        colors = ['green' if profit >= 0 else 'red' for profit in trade_profits]

        # Построение графика прибыли/убытка по сделкам
        plt.bar(trade_indices, trade_profits, color=colors)

        # Добавляем линию нулевой прибыли
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        # Добавляем линию средней прибыли/убытка
        avg_profit = sum(trade_profits) / len(trade_profits)
        plt.axhline(y=avg_profit, color='blue', linestyle='--', label=f'Mean ({avg_profit:.2f}%)')

        # Настройка графика
        plt.title(f'Trade Profits - {self.strategy_name}')
        plt.xlabel('Trade #')
        plt.ylabel('Profit/Loss (%)')

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # Сохранение графика, если нужно
        if save:
            filename = os.path.join(self.plots_dir, 'trade_profits.png')
            plt.savefig(filename)
            logger.info(f"График сохранен в {filename}")

        return plt.gcf()

    def plot_profit_distribution(self, save: bool = True) -> plt.Figure:
        """
        Строит распределение прибыли по сделкам.

        Args:
            save: Сохранять ли график

        Returns:
            Объект Figure
        """
        if not self.trades:
            logger.warning("Нет сделок для построения графика")
            return plt.figure()

        plt.figure(figsize=(12, 6))

        # Получаем прибыль/убыток по каждой сделке
        trade_profits = [trade.get('profit_loss_pct', 0) for trade in self.trades]

        # Построение гистограммы прибыли по сделкам
        sns.histplot(trade_profits, bins=20, kde=True)

        # Добавляем вертикальную линию для нулевой прибыли
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)

        # Добавляем вертикальную линию для средней прибыли/убытка
        avg_profit = sum(trade_profits) / len(trade_profits)
        plt.axvline(x=avg_profit, color='blue', linestyle='--', label=f'Mean ({avg_profit:.2f}%)')

        # Настройка графика
        plt.title(f'Profit Distribution - {self.strategy_name}')
        plt.xlabel('Profit/Loss (%)')
        plt.ylabel('Frequency')

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # Сохранение графика, если нужно
        if save:
            filename = os.path.join(self.plots_dir, 'profit_distribution.png')
            plt.savefig(filename)
            logger.info(f"График сохранен в {filename}")

        return plt.gcf()

    def plot_trade_durations(self, save: bool = True) -> plt.Figure:
        """
        Строит распределение продолжительности сделок.

        Args:
            save: Сохранять ли график

        Returns:
            Объект Figure
        """
        if not self.trades:
            logger.warning("Нет сделок для построения графика")
            return plt.figure()

        # Проверяем наличие информации о времени входа и выхода
        if 'entry_time' not in self.trades[0] or 'exit_time' not in self.trades[0]:
            logger.warning("Нет информации о времени входа и выхода для построения графика")
            return plt.figure()

        plt.figure(figsize=(12, 6))

        # Рассчитываем продолжительность каждой сделки
        durations = []

        for trade in self.trades:
            entry_time = pd.to_datetime(trade['entry_time'])
            exit_time = pd.to_datetime(trade['exit_time'])

            duration = (exit_time - entry_time).total_seconds() / (60 * 60 * 24)  # в днях
            durations.append(duration)

        # Построение гистограммы продолжительности сделок
        sns.histplot(durations, bins=20, kde=True)

        # Добавляем вертикальную линию для средней продолжительности
        avg_duration = sum(durations) / len(durations)
        plt.axvline(x=avg_duration, color='blue', linestyle='--', label=f'Mean ({avg_duration:.2f} days)')

        # Настройка графика
        plt.title(f'Trade Duration Distribution - {self.strategy_name}')
        plt.xlabel('Duration (days)')
        plt.ylabel('Frequency')

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # Сохранение графика, если нужно
        if save:
            filename = os.path.join(self.plots_dir, 'trade_durations.png')
            plt.savefig(filename)
            logger.info(f"График сохранен в {filename}")

        return plt.gcf()

    def plot_win_loss_ratio(self, save: bool = True) -> plt.Figure:
        """
        Строит круговую диаграмму соотношения выигрышных и проигрышных сделок.

        Args:
            save: Сохранять ли график

        Returns:
            Объект Figure
        """
        if not self.trades:
            logger.warning("Нет сделок для построения графика")
            return plt.figure()

        plt.figure(figsize=(10, 6))

        # Подсчитываем выигрышные и проигрышные сделки
        winning_trades = len([t for t in self.trades if t.get('profit_loss', 0) > 0])
        losing_trades = len([t for t in self.trades if t.get('profit_loss', 0) <= 0])

        # Создаем данные для круговой диаграммы
        labels = ['Winning Trades', 'Losing Trades']
        sizes = [winning_trades, losing_trades]
        colors = ['green', 'red']
        explode = (0.1, 0)  # выделяем выигрышные сделки

        # Построение круговой диаграммы
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')  # для круговой диаграммы

        # Настройка графика
        plt.title(f'Win/Loss Ratio - {self.strategy_name}')

        # Сохранение графика, если нужно
        if save:
            filename = os.path.join(self.plots_dir, 'win_loss_ratio.png')
            plt.savefig(filename)
            logger.info(f"График сохранен в {filename}")

        return plt.gcf()

    def plot_rolling_returns(self, window: int = 252, save: bool = True) -> plt.Figure:
        """
        Строит график скользящей доходности.

        Args:
            window: Размер окна (в днях)
            save: Сохранять ли график

        Returns:
            Объект Figure
        """
        if len(self.data) < window:
            logger.warning(f"Недостаточно данных для расчета скользящей доходности с окном {window}")
            return plt.figure()

        plt.figure(figsize=(12, 6))

        # Рассчитываем скользящую доходность
        equity_series = self.data['equity']
        rolling_returns = equity_series.pct_change(window).dropna() * 100

        # Построение графика скользящей доходности
        plt.plot(rolling_returns.index, rolling_returns, label=f'{window}-day Rolling Return (%)', color='blue')

        # Добавляем линию нулевой доходности
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        # Настройка графика
        plt.title(f'{window}-day Rolling Returns - {self.strategy_name}')
        plt.xlabel('Date')
        plt.ylabel('Return (%)')

        # Форматирование оси X для временных меток
        if hasattr(self.data.index[0], 'strftime'):
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.gcf().autofmt_xdate()

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # Сохранение графика, если нужно
        if save:
            filename = os.path.join(self.plots_dir, f'rolling_returns_{window}d.png')
            plt.savefig(filename)
            logger.info(f"График сохранен в {filename}")

        return plt.gcf()

    def plot_rolling_sharpe(self, window: int = 252, risk_free_rate: float = 0.0, save: bool = True) -> plt.Figure:
        """
        Строит график скользящего коэффициента Шарпа.

        Args:
            window: Размер окна (в днях)
            risk_free_rate: Безрисковая ставка (годовая)
            save: Сохранять ли график

        Returns:
            Объект Figure
        """
        if len(self.data) < window:
            logger.warning(f"Недостаточно данных для расчета скользящего коэффициента Шарпа с окном {window}")
            return plt.figure()

        plt.figure(figsize=(12, 6))

        # Рассчитываем дневную доходность
        equity_series = self.data['equity']
        returns = equity_series.pct_change().dropna()

        # Переводим годовую безрисковую ставку в дневную
        daily_risk_free = (1.0 + risk_free_rate) ** (1.0 / window) - 1.0

        # Рассчитываем избыточную доходность
        excess_returns = returns - daily_risk_free

        # Рассчитываем скользящее среднее и стандартное отклонение
        rolling_mean = excess_returns.rolling(window=window).mean()
        rolling_std = excess_returns.rolling(window=window).std()

        # Рассчитываем скользящий коэффициент Шарпа
        rolling_sharpe = rolling_mean / rolling_std * np.sqrt(window)

        # Построение графика скользящего коэффициента Шарпа
        plt.plot(rolling_sharpe.index, rolling_sharpe, label=f'{window}-day Rolling Sharpe', color='purple')

        # Добавляем линию нулевого коэффициента Шарпа
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        # Настройка графика
        plt.title(f'{window}-day Rolling Sharpe Ratio - {self.strategy_name}')
        plt.xlabel('Date')
        plt.ylabel('Sharpe Ratio')

        # Форматирование оси X для временных меток
        if hasattr(self.data.index[0], 'strftime'):
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.gcf().autofmt_xdate()

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # Сохранение графика, если нужно
        if save:
            filename = os.path.join(self.plots_dir, f'rolling_sharpe_{window}d.png')
            plt.savefig(filename)
            logger.info(f"График сохранен в {filename}")

        return plt.gcf()

    def generate_html_report(self, output_file: str = "report.html") -> str:
        """
        Генерирует HTML-отчет с результатами бэктестирования.

        Args:
            output_file: Имя файла для отчета

        Returns:
            Путь к сгенерированному отчету
        """
        import base64
        from io import BytesIO

        # Создаем директорию для отчета, если она не существует
        report_dir = os.path.join(self.output_dir, "html")
        os.makedirs(report_dir, exist_index=True)

        # Путь к отчету
        report_path = os.path.join(report_dir, output_file)

        # Генерируем графики и получаем их как base64
        def fig_to_base64(fig):
            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            plt.close(fig)
            return img_base64

        equity_curve_base64 = fig_to_base64(self.plot_equity_curve(save=False))
        monthly_returns_base64 = fig_to_base64(self.plot_monthly_returns(save=False))
        drawdowns_base64 = fig_to_base64(self.plot_drawdowns(save=False))
        trade_profits_base64 = fig_to_base64(self.plot_trade_profits(save=False))
        profit_distribution_base64 = fig_to_base64(self.plot_profit_distribution(save=False))
        win_loss_ratio_base64 = fig_to_base64(self.plot_win_loss_ratio(save=False))

        # Генерируем таблицы
        summary_df = self.generate_summary()
        monthly_returns_df = self.generate_monthly_returns()
        yearly_returns_df = self.generate_yearly_returns()
        trade_statistics_df = self.generate_trade_statistics()
        drawdown_periods_df = self.generate_drawdown_periods()

        # Создаем HTML-отчет
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report - {self.strategy_name}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    padding: 0;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .section {{
                    margin-bottom: 30px;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .img-container {{
                    margin-bottom: 20px;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                    gap: 20px;
                }}
                .metric-card {{
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    background-color: #f9f9f9;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2980b9;
                }}
                .positive {{
                    color: green;
                }}
                .negative {{
                    color: red;
                }}
            </style>
        </head>
        <body>
            <h1>Backtest Report - {self.strategy_name}</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <div class="section">
                <h2>Summary</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h3>Total Return</h3>
                        <div class="metric-value {'positive' if self.metrics['total_return'] >= 0 else 'negative'}">
                            {self.metrics['total_return_pct']:.2f}%
                        </div>
                    </div>
                    <div class="metric-card">
                        <h3>Annual Return</h3>
                        <div class="metric-value {'positive' if self.metrics['annual_return'] >= 0 else 'negative'}">
                            {self.metrics['annual_return_pct']:.2f}%
                        </div>
                    </div>
                    <div class="metric-card">
                        <h3>Max Drawdown</h3>
                        <div class="metric-value negative">
                            {self.metrics['max_drawdown_pct']:.2f}%
                        </div>
                    </div>
                    <div class="metric-card">
                        <h3>Sharpe Ratio</h3>
                        <div class="metric-value {'positive' if self.metrics['sharpe_ratio'] >= 0 else 'negative'}">
                            {self.metrics['sharpe_ratio']:.2f}
                        </div>
                    </div>
                    <div class="metric-card">
                        <h3>Win Rate</h3>
                        <div class="metric-value">
                            {self.metrics['win_rate_pct']:.2f}%
                        </div>
                    </div>
                    <div class="metric-card">
                        <h3>Profit Factor</h3>
                        <div class="metric-value">
                            {self.metrics['profit_factor']:.2f}
                        </div>
                    </div>
                    <div class="metric-card">
                        <h3>Total Trades</h3>
                        <div class="metric-value">
                            {self.metrics['total_trades']}
                        </div>
                    </div>
                    <div class="metric-card">
                        <h3>Recovery Factor</h3>
                        <div class="metric-value">
                            {self.metrics['recovery_factor']:.2f}
                        </div>
                    </div>
                </div>

                <h3>All Metrics</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    {summary_df.to_html(index=False, classes='table table-striped', border=0)}
                </table>
            </div>

            <div class="section">
                <h2>Equity Curve</h2>
                <div class="img-container">
                    <img src="data:image/png;base64,{equity_curve_base64}" alt="Equity Curve">
                </div>
            </div>

            <div class="section">
                <h2>Monthly Returns</h2>
                <div class="img-container">
                    <img src="data:image/png;base64,{monthly_returns_base64}" alt="Monthly Returns">
                </div>

                <h3>Monthly Returns Table</h3>
                <table>
                    <tr>
                        <th>Month</th>
                        <th>Return (%)</th>
                    </tr>
                    {monthly_returns_df.to_html(classes='table table-striped', border=0)}
                </table>

                <h3>Yearly Returns Table</h3>
                <table>
                    <tr>
                        <th>Year</th>
                        <th>Return (%)</th>
                    </tr>
                    {yearly_returns_df.to_html(classes='table table-striped', border=0)}
                </table>
            </div>

            <div class="section">
                <h2>Drawdowns</h2>
                <div class="img-container">
                    <img src="data:image/png;base64,{drawdowns_base64}" alt="Drawdowns">
                </div>

                <h3>Drawdown Periods</h3>
                <table>
                    <tr>
                        <th>Start Date</th>
                        <th>End Date</th>
                        <th>Duration (days)</th>
                        <th>Max Drawdown (%)</th>
                        <th>Max Drawdown Date</th>
                    </tr>
                    {drawdown_periods_df.to_html(classes='table table-striped', border=0)}
                </table>
            </div>

            <div class="section">
                <h2>Trade Analysis</h2>
                <div class="img-container">
                    <img src="data:image/png;base64,{trade_profits_base64}" alt="Trade Profits">
                </div>

                <div class="img-container">
                    <img src="data:image/png;base64,{profit_distribution_base64}" alt="Profit Distribution">
                </div>

                <div class="img-container">
                    <img src="data:image/png;base64,{win_loss_ratio_base64}" alt="Win/Loss Ratio">
                </div>

                <h3>Trade Statistics</h3>
                <table>
                    <tr>
                        <th>Type</th>
                        <th>Entry Time</th>
                        <th>Exit Time</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>Position</th>
                        <th>Profit/Loss</th>
                        <th>Profit/Loss (%)</th>
                    </tr>
                    {trade_statistics_df.to_html(classes='table table-striped', border=0)}
                </table>
            </div>
        </body>
        </html>
        """

        # Сохраняем HTML-отчет
        with open(report_path, 'w') as f:
            f.write(html)

        logger.info(f"HTML-отчет сохранен в {report_path}")

        return report_path

    def save_metrics_to_json(self, output_file: str = "metrics.json") -> str:
        """
        Сохраняет метрики в JSON-файл.

        Args:
            output_file: Имя файла для метрик

        Returns:
            Путь к сохраненному файлу
        """
        # Создаем директорию для JSON, если она не существует
        json_dir = os.path.join(self.output_dir, "json")
        os.makedirs(json_dir, exist_ok=True)

        # Путь к файлу
        json_path = os.path.join(json_dir, output_file)

        # Сохраняем метрики в JSON-файл
        with open(json_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)

        logger.info(f"Метрики сохранены в {json_path}")

        return json_path

    def save_trades_to_csv(self, output_file: str = "trades.csv") -> str:
        """
        Сохраняет сделки в CSV-файл.

        Args:
            output_file: Имя файла для сделок

        Returns:
            Путь к сохраненному файлу
        """
        if not self.trades:
            logger.warning("Нет сделок для сохранения")
            return ""

        # Создаем директорию для CSV, если она не существует
        csv_dir = os.path.join(self.output_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)

        # Путь к файлу
        csv_path = os.path.join(csv_dir, output_file)

        # Создаем DataFrame из сделок
        trades_df = pd.DataFrame(self.trades)

        # Сохраняем сделки в CSV-файл
        trades_df.to_csv(csv_path, index=False)

        logger.info(f"Сделки сохранены в {csv_path}")

        return csv_path

    def generate_full_report(self) -> Dict[str, str]:
        """
        Генерирует полный отчет, включая HTML, JSON и CSV.

        Returns:
            Словарь с путями к сгенерированным файлам
        """
        # Генерируем все графики
        self.plot_equity_curve()
        self.plot_monthly_returns()
        self.plot_drawdowns()
        self.plot_trade_profits()
        self.plot_profit_distribution()
        self.plot_trade_durations()
        self.plot_win_loss_ratio()
        self.plot_rolling_returns()
        self.plot_rolling_sharpe()

        # Генерируем HTML-отчет
        html_path = self.generate_html_report()

        # Сохраняем метрики и сделки
        json_path = self.save_metrics_to_json()
        csv_path = self.save_trades_to_csv()

        # Возвращаем пути к файлам
        return {
            'html_report': html_path,
            'metrics_json': json_path,
            'trades_csv': csv_path,
            'plots_dir': self.plots_dir
        }


def compare_backtest_results(
        results1: Dict,
        results2: Dict,
        name1: str = "Strategy 1",
        name2: str = "Strategy 2",
        output_dir: str = "reports/comparison"
) -> Dict:
    """
    Сравнивает результаты двух бэктестов и генерирует отчет.

    Args:
        results1: Результаты первого бэктеста
        results2: Результаты второго бэктеста
        name1: Название первой стратегии
        name2: Название второй стратегии
        output_dir: Директория для сохранения отчетов

    Returns:
        Словарь со сравнительными метриками
    """
    # Создаем директорию для отчетов, если она не существует
    os.makedirs(output_dir, exist_ok=True)

    # Извлекаем данные и метрики
    data1 = results1['data']
    data2 = results2['data']
    metrics1 = results1['metrics']
    metrics2 = results2['metrics']

    # Сравниваем метрики
    comparison = {}

    # Метрики для сравнения
    metrics_to_compare = [
        'total_return', 'annual_return', 'max_drawdown', 'sharpe_ratio',
        'sortino_ratio', 'calmar_ratio', 'win_rate', 'profit_factor',
        'total_trades', 'profitable_trades', 'losing_trades'
    ]

    for metric in metrics_to_compare:
        if metric in metrics1 and metric in metrics2:
            comparison[f'{metric}_{name1}'] = metrics1[metric]
            comparison[f'{metric}_{name2}'] = metrics2[metric]
            comparison[f'{metric}_diff'] = metrics1[metric] - metrics2[metric]

            # Для процентных метрик
            if metric in ['total_return', 'annual_return', 'max_drawdown', 'win_rate']:
                comparison[f'{metric}_pct_{name1}'] = metrics1[f'{metric}_pct']
                comparison[f'{metric}_pct_{name2}'] = metrics2[f'{metric}_pct']
                comparison[f'{metric}_pct_diff'] = metrics1[f'{metric}_pct'] - metrics2[f'{metric}_pct']

    # Строим сравнительные графики
    plt.figure(figsize=(12, 6))

    # Нормализуем кривые капитала для сравнения
    equity1 = data1['equity'] / data1['equity'].iloc[0]
    equity2 = data2['equity'] / data2['equity'].iloc[0]

    # Построение графика сравнения кривых капитала
    plt.plot(equity1.index, equity1, label=name1, color='blue')
    plt.plot(equity2.index, equity2, label=name2, color='green')

    # Настройка графика
    plt.title(f'Equity Curve Comparison: {name1} vs {name2}')
    plt.xlabel('Date')
    plt.ylabel('Normalized Equity')

    # Форматирование оси X для временных меток
    if hasattr(data1.index[0], 'strftime'):
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate()

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Сохраняем график
    plt.savefig(os.path.join(output_dir, 'equity_comparison.png'))

    # Сохраняем сравнительные метрики в JSON
    with open(os.path.join(output_dir, 'comparison_metrics.json'), 'w') as f:
        json.dump(comparison, f, indent=4)

    # Строим сравнительную таблицу
    comparison_df = pd.DataFrame({
        'Metric': [
            'Total Return (%)', 'Annual Return (%)', 'Max Drawdown (%)',
            'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
            'Win Rate (%)', 'Profit Factor', 'Total Trades',
            'Profitable Trades', 'Losing Trades'
        ],
        name1: [
            metrics1.get('total_return_pct', 0),
            metrics1.get('annual_return_pct', 0),
            metrics1.get('max_drawdown_pct', 0),
            metrics1.get('sharpe_ratio', 0),
            metrics1.get('sortino_ratio', 0),
            metrics1.get('calmar_ratio', 0),
            metrics1.get('win_rate_pct', 0),
            metrics1.get('profit_factor', 0),
            metrics1.get('total_trades', 0),
            metrics1.get('profitable_trades', 0),
            metrics1.get('losing_trades', 0)
        ],
        name2: [
            metrics2.get('total_return_pct', 0),
            metrics2.get('annual_return_pct', 0),
            metrics2.get('max_drawdown_pct', 0),
            metrics2.get('sharpe_ratio', 0),
            metrics2.get('sortino_ratio', 0),
            metrics2.get('calmar_ratio', 0),
            metrics2.get('win_rate_pct', 0),
            metrics2.get('profit_factor', 0),
            metrics2.get('total_trades', 0),
            metrics2.get('profitable_trades', 0),
            metrics2.get('losing_trades', 0)
        ],
        'Difference': [
            metrics1.get('total_return_pct', 0) - metrics2.get('total_return_pct', 0),
            metrics1.get('annual_return_pct', 0) - metrics2.get('annual_return_pct', 0),
            metrics1.get('max_drawdown_pct', 0) - metrics2.get('max_drawdown_pct', 0),
            metrics1.get('sharpe_ratio', 0) - metrics2.get('sharpe_ratio', 0),
            metrics1.get('sortino_ratio', 0) - metrics2.get('sortino_ratio', 0),
            metrics1.get('calmar_ratio', 0) - metrics2.get('calmar_ratio', 0),
            metrics1.get('win_rate_pct', 0) - metrics2.get('win_rate_pct', 0),
            metrics1.get('profit_factor', 0) - metrics2.get('profit_factor', 0),
            metrics1.get('total_trades', 0) - metrics2.get('total_trades', 0),
            metrics1.get('profitable_trades', 0) - metrics2.get('profitable_trades', 0),
            metrics1.get('losing_trades', 0) - metrics2.get('losing_trades', 0)
        ]
    })

    # Сохраняем сравнительную таблицу в CSV
    comparison_df.to_csv(os.path.join(output_dir, 'comparison_table.csv'), index=False)

    return comparison