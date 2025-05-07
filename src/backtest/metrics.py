"""
Модуль с метриками для оценки эффективности торговых стратегий.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def calculate_total_return(equity: pd.Series) -> float:
    """
    Рассчитывает общую доходность.

    Args:
        equity: Серия с стоимостью портфеля

    Returns:
        Общая доходность
    """
    if len(equity) < 2:
        return 0.0

    return equity.iloc[-1] / equity.iloc[0] - 1.0


def calculate_annual_return(equity: pd.Series, trading_days: int = 252) -> float:
    """
    Рассчитывает годовую доходность.

    Args:
        equity: Серия с стоимостью портфеля
        trading_days: Количество торговых дней в году

    Returns:
        Годовая доходность
    """
    if len(equity) < 2:
        return 0.0

    total_return = calculate_total_return(equity)
    days = len(equity)

    return (1.0 + total_return) ** (trading_days / days) - 1.0


def calculate_drawdown(equity: pd.Series) -> pd.Series:
    """
    Рассчитывает просадку.

    Args:
        equity: Серия с стоимостью портфеля

    Returns:
        Серия с просадкой
    """
    return 1.0 - equity / equity.cummax()


def calculate_max_drawdown(equity: pd.Series) -> float:
    """
    Рассчитывает максимальную просадку.

    Args:
        equity: Серия с стоимостью портфеля

    Returns:
        Максимальная просадка
    """
    drawdown = calculate_drawdown(equity)
    return drawdown.max()


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, trading_days: int = 252) -> float:
    """
    Рассчитывает коэффициент Шарпа.

    Args:
        returns: Серия с доходностью
        risk_free_rate: Безрисковая ставка (годовая)
        trading_days: Количество торговых дней в году

    Returns:
        Коэффициент Шарпа
    """
    if len(returns) < 2:
        return 0.0

    # Переводим годовую безрисковую ставку в дневную
    daily_risk_free = (1.0 + risk_free_rate) ** (1.0 / trading_days) - 1.0

    excess_returns = returns - daily_risk_free
    return excess_returns.mean() / excess_returns.std() * np.sqrt(trading_days)


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, trading_days: int = 252) -> float:
    """
    Рассчитывает коэффициент Сортино (учитывает только отрицательную волатильность).

    Args:
        returns: Серия с доходностью
        risk_free_rate: Безрисковая ставка (годовая)
        trading_days: Количество торговых дней в году

    Returns:
        Коэффициент Сортино
    """
    if len(returns) < 2:
        return 0.0

    # Переводим годовую безрисковую ставку в дневную
    daily_risk_free = (1.0 + risk_free_rate) ** (1.0 / trading_days) - 1.0

    excess_returns = returns - daily_risk_free

    # Расчет стандартного отклонения только отрицательных доходностей
    negative_returns = excess_returns[excess_returns < 0]

    if len(negative_returns) == 0 or negative_returns.std() == 0:
        return float('inf')

    return excess_returns.mean() / negative_returns.std() * np.sqrt(trading_days)


def calculate_calmar_ratio(equity: pd.Series, trading_days: int = 252) -> float:
    """
    Рассчитывает коэффициент Калмара (годовая доходность / максимальная просадка).

    Args:
        equity: Серия с стоимостью портфеля
        trading_days: Количество торговых дней в году

    Returns:
        Коэффициент Калмара
    """
    if len(equity) < 2:
        return 0.0

    annual_return = calculate_annual_return(equity, trading_days)
    max_drawdown = calculate_max_drawdown(equity)

    if max_drawdown == 0:
        return float('inf')

    return annual_return / max_drawdown


def calculate_gain_to_pain_ratio(returns: pd.Series) -> float:
    """
    Рассчитывает отношение прибыли к боли (сумма положительных доходностей / модуль суммы отрицательных доходностей).

    Args:
        returns: Серия с доходностью

    Returns:
        Отношение прибыли к боли
    """
    if len(returns) < 2:
        return 0.0

    positive_returns = returns[returns > 0].sum()
    negative_returns = abs(returns[returns < 0].sum())

    if negative_returns == 0:
        return float('inf')

    return positive_returns / negative_returns


def calculate_win_rate(trades: List[Dict]) -> float:
    """
    Рассчитывает процент выигрышных сделок.

    Args:
        trades: Список сделок

    Returns:
        Процент выигрышных сделок
    """
    if not trades:
        return 0.0

    winning_trades = [t for t in trades if t.get('profit_loss', 0) > 0]
    return len(winning_trades) / len(trades)


def calculate_profit_factor(trades: List[Dict]) -> float:
    """
    Рассчитывает фактор прибыли (общая прибыль / общий убыток).

    Args:
        trades: Список сделок

    Returns:
        Фактор прибыли
    """
    if not trades:
        return 0.0

    winning_trades = [t.get('profit_loss', 0) for t in trades if t.get('profit_loss', 0) > 0]
    losing_trades = [abs(t.get('profit_loss', 0)) for t in trades if t.get('profit_loss', 0) < 0]

    total_profit = sum(winning_trades)
    total_loss = sum(losing_trades)

    if total_loss == 0:
        return float('inf')

    return total_profit / total_loss


def calculate_expectancy(trades: List[Dict]) -> float:
    """
    Рассчитывает математическое ожидание прибыли на сделку.

    Args:
        trades: Список сделок

    Returns:
        Математическое ожидание прибыли на сделку
    """
    if not trades:
        return 0.0

    total_profit_loss = sum(t.get('profit_loss', 0) for t in trades)
    return total_profit_loss / len(trades)


def calculate_average_trade_duration(trades: List[Dict]) -> float:
    """
    Рассчитывает среднюю продолжительность сделки (в днях).

    Args:
        trades: Список сделок

    Returns:
        Средняя продолжительность сделки
    """
    if not trades:
        return 0.0

    # Проверяем, есть ли информация о времени входа и выхода
    if 'entry_time' not in trades[0] or 'exit_time' not in trades[0]:
        return 0.0

    durations = []

    for trade in trades:
        entry_time = pd.to_datetime(trade['entry_time'])
        exit_time = pd.to_datetime(trade['exit_time'])

        duration = (exit_time - entry_time).total_seconds() / (60 * 60 * 24)  # в днях
        durations.append(duration)

    return sum(durations) / len(durations)


def calculate_max_consecutive_losses(trades: List[Dict]) -> int:
    """
    Рассчитывает максимальное количество последовательных убыточных сделок.

    Args:
        trades: Список сделок

    Returns:
        Максимальное количество последовательных убыточных сделок
    """
    if not trades:
        return 0

    consecutive_losses = 0
    max_consecutive_losses = 0

    for trade in trades:
        if trade.get('profit_loss', 0) <= 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            consecutive_losses = 0

    return max_consecutive_losses


def calculate_average_profit_loss(trades: List[Dict]) -> Tuple[float, float]:
    """
    Рассчитывает среднюю прибыль и средний убыток по сделкам.

    Args:
        trades: Список сделок

    Returns:
        Кортеж (средняя прибыль, средний убыток)
    """
    if not trades:
        return (0.0, 0.0)

    winning_trades = [t.get('profit_loss', 0) for t in trades if t.get('profit_loss', 0) > 0]
    losing_trades = [t.get('profit_loss', 0) for t in trades if t.get('profit_loss', 0) <= 0]

    avg_profit = sum(winning_trades) / len(winning_trades) if winning_trades else 0.0
    avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0.0

    return (avg_profit, avg_loss)


def calculate_ulcer_index(equity: pd.Series) -> float:
    """
    Рассчитывает Ulcer Index (мера риска, учитывающая глубину и продолжительность просадок).

    Args:
        equity: Серия с стоимостью портфеля

    Returns:
        Ulcer Index
    """
    if len(equity) < 2:
        return 0.0

    drawdown = calculate_drawdown(equity)
    return np.sqrt(np.mean(drawdown ** 2))


def calculate_maximum_adverse_excursion(trades: List[Dict]) -> Tuple[float, float, float]:
    """
    Рассчитывает максимальное неблагоприятное отклонение (MAE) для сделок.

    Args:
        trades: Список сделок

    Returns:
        Кортеж (средний MAE, максимальный MAE, минимальный MAE)
    """
    if not trades or 'mae' not in trades[0]:
        return (0.0, 0.0, 0.0)

    mae_values = [t.get('mae', 0.0) for t in trades]

    avg_mae = sum(mae_values) / len(mae_values)
    max_mae = max(mae_values)
    min_mae = min(mae_values)

    return (avg_mae, max_mae, min_mae)


def calculate_maximum_favorable_excursion(trades: List[Dict]) -> Tuple[float, float, float]:
    """
    Рассчитывает максимальное благоприятное отклонение (MFE) для сделок.

    Args:
        trades: Список сделок

    Returns:
        Кортеж (средний MFE, максимальный MFE, минимальный MFE)
    """
    if not trades or 'mfe' not in trades[0]:
        return (0.0, 0.0, 0.0)

    mfe_values = [t.get('mfe', 0.0) for t in trades]

    avg_mfe = sum(mfe_values) / len(mfe_values)
    max_mfe = max(mfe_values)
    min_mfe = min(mfe_values)

    return (avg_mfe, max_mfe, min_mfe)


def calculate_all_metrics(equity: pd.Series, trades: List[Dict], risk_free_rate: float = 0.0) -> Dict:
    """
    Рассчитывает все доступные метрики.

    Args:
        equity: Серия с стоимостью портфеля
        trades: Список сделок
        risk_free_rate: Безрисковая ставка (годовая)

    Returns:
        Словарь с метриками
    """
    # Расчет доходности
    returns = equity.pct_change().dropna()

    # Расчет всех метрик
    total_return = calculate_total_return(equity)
    annual_return = calculate_annual_return(equity)
    max_drawdown = calculate_max_drawdown(equity)
    sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate)
    sortino_ratio = calculate_sortino_ratio(returns, risk_free_rate)
    calmar_ratio = calculate_calmar_ratio(equity)
    gain_to_pain_ratio = calculate_gain_to_pain_ratio(returns)
    ulcer_index = calculate_ulcer_index(equity)

    # Метрики для сделок
    win_rate = calculate_win_rate(trades)
    profit_factor = calculate_profit_factor(trades)
    expectancy = calculate_expectancy(trades)
    avg_trade_duration = calculate_average_trade_duration(trades)
    max_consecutive_losses = calculate_max_consecutive_losses(trades)
    avg_profit, avg_loss = calculate_average_profit_loss(trades)

    # Сбор всех метрик в словарь
    metrics = {
        'total_return': total_return,
        'total_return_pct': total_return * 100,
        'annual_return': annual_return,
        'annual_return_pct': annual_return * 100,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown * 100,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'gain_to_pain_ratio': gain_to_pain_ratio,
        'ulcer_index': ulcer_index,
        'win_rate': win_rate,
        'win_rate_pct': win_rate * 100,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'avg_trade_duration': avg_trade_duration,
        'max_consecutive_losses': max_consecutive_losses,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'profit_loss_ratio': abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf'),
        'total_trades': len(trades),
        'profitable_trades': len([t for t in trades if t.get('profit_loss', 0) > 0]),
        'losing_trades': len([t for t in trades if t.get('profit_loss', 0) <= 0]),
        'recovery_factor': total_return / max_drawdown if max_drawdown > 0 else float('inf')
    }

    return metrics


def compare_strategies(metrics1: Dict, metrics2: Dict, name1: str = 'Strategy 1', name2: str = 'Strategy 2') -> Dict:
    """
    Сравнивает две стратегии по метрикам.

    Args:
        metrics1: Метрики первой стратегии
        metrics2: Метрики второй стратегии
        name1: Название первой стратегии
        name2: Название второй стратегии

    Returns:
        Словарь со сравнительными метриками
    """
    # Метрики для сравнения
    comparison_metrics = [
        'total_return', 'annual_return', 'max_drawdown', 'sharpe_ratio',
        'sortino_ratio', 'calmar_ratio', 'gain_to_pain_ratio', 'win_rate',
        'profit_factor', 'expectancy'
    ]

    comparison = {}

    for metric in comparison_metrics:
        value1 = metrics1.get(metric, 0.0)
        value2 = metrics2.get(metric, 0.0)

        comparison[f'{metric}_{name1}'] = value1
        comparison[f'{metric}_{name2}'] = value2
        comparison[f'{metric}_diff'] = value1 - value2
        comparison[f'{metric}_ratio'] = value1 / value2 if value2 != 0 else float('inf')

    return comparison


def calculate_monthly_returns(equity: pd.Series) -> pd.Series:
    """
    Рассчитывает месячную доходность.

    Args:
        equity: Серия с стоимостью портфеля

    Returns:
        Серия с месячной доходностью
    """
    # Проверяем, что индекс - временная метка
    if not hasattr(equity.index[0], 'strftime'):
        logger.warning("Индекс данных не является временной меткой, невозможно рассчитать месячную доходность")
        return pd.Series()

    # Рассчитываем месячную доходность
    monthly_returns = equity.resample('M').last().pct_change().dropna()

    return monthly_returns


def calculate_annual_returns(equity: pd.Series) -> pd.Series:
    """
    Рассчитывает годовую доходность.

    Args:
        equity: Серия с стоимостью портфеля

    Returns:
        Серия с годовой доходностью
    """
    # Проверяем, что индекс - временная метка
    if not hasattr(equity.index[0], 'strftime'):
        logger.warning("Индекс данных не является временной меткой, невозможно рассчитать годовую доходность")
        return pd.Series()

    # Рассчитываем годовую доходность
    annual_returns = equity.resample('Y').last().pct_change().dropna()

    return annual_returns


def calculate_rolling_returns(equity: pd.Series, window: int = 252) -> pd.Series:
    """
    Рассчитывает скользящую доходность за указанный период.

    Args:
        equity: Серия с стоимостью портфеля
        window: Размер окна (в днях)

    Returns:
        Серия со скользящей доходностью
    """
    if len(equity) < window:
        logger.warning(f"Недостаточно данных для расчета скользящей доходности с окном {window}")
        return pd.Series()

    # Рассчитываем скользящую доходность
    rolling_returns = equity.pct_change(window).dropna()

    return rolling_returns


def calculate_rolling_sharpe(returns: pd.Series, window: int = 252, risk_free_rate: float = 0.0) -> pd.Series:
    """
    Рассчитывает скользящий коэффициент Шарпа.

    Args:
        returns: Серия с доходностью
        window: Размер окна (в днях)
        risk_free_rate: Безрисковая ставка (годовая)

    Returns:
        Серия со скользящим коэффициентом Шарпа
    """
    if len(returns) < window:
        logger.warning(f"Недостаточно данных для расчета скользящего коэффициента Шарпа с окном {window}")
        return pd.Series()

    # Переводим годовую безрисковую ставку в дневную
    daily_risk_free = (1.0 + risk_free_rate) ** (1.0 / window) - 1.0

    # Рассчитываем избыточную доходность
    excess_returns = returns - daily_risk_free

    # Рассчитываем скользящее среднее и стандартное отклонение
    rolling_mean = excess_returns.rolling(window=window).mean()
    rolling_std = excess_returns.rolling(window=window).std()

    # Рассчитываем скользящий коэффициент Шарпа
    rolling_sharpe = rolling_mean / rolling_std * np.sqrt(window)

    return rolling_sharpe


def calculate_beta(returns: pd.Series, benchmark_returns: pd.Series, window: Optional[int] = None) -> Union[
    float, pd.Series]:
    """
    Рассчитывает бету (коэффициент относительной волатильности).

    Args:
        returns: Серия с доходностью стратегии
        benchmark_returns: Серия с доходностью бенчмарка
        window: Размер окна для скользящей беты (опционально)

    Returns:
        Бета или серия со скользящей бетой
    """
    if window is None:
        # Расчет беты для всего периода
        covariance = returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()

        if benchmark_variance == 0:
            return 0.0

        return covariance / benchmark_variance
    else:
        # Расчет скользящей беты
        if len(returns) < window or len(benchmark_returns) < window:
            logger.warning(f"Недостаточно данных для расчета скользящей беты с окном {window}")
            return pd.Series()

        # Создаем DataFrame с доходностями
        df = pd.DataFrame({'returns': returns, 'benchmark': benchmark_returns})

        # Рассчитываем скользящую ковариацию и дисперсию
        rolling_cov = df['returns'].rolling(window=window).cov(df['benchmark'])
        rolling_var = df['benchmark'].rolling(window=window).var()

        # Рассчитываем скользящую бету
        rolling_beta = rolling_cov / rolling_var

        return rolling_beta


def calculate_alpha(returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.0,
                    window: Optional[int] = None) -> Union[float, pd.Series]:
    """
    Рассчитывает альфу (избыточную доходность с учетом беты).

    Args:
        returns: Серия с доходностью стратегии
        benchmark_returns: Серия с доходностью бенчмарка
        risk_free_rate: Безрисковая ставка (годовая)
        window: Размер окна для скользящей альфы (опционально)

    Returns:
        Альфа или серия со скользящей альфой
    """
    # Переводим годовую безрисковую ставку в дневную
    daily_risk_free = (1.0 + risk_free_rate) ** (1.0 / 252) - 1.0

    if window is None:
        # Расчет альфы для всего периода
        beta = calculate_beta(returns, benchmark_returns)

        # Расчет альфы по формуле CAPM
        alpha = returns.mean() - daily_risk_free - beta * (benchmark_returns.mean() - daily_risk_free)

        # Годовая альфа
        annual_alpha = (1 + alpha) ** 252 - 1

        return annual_alpha
    else:
        # Расчет скользящей альфы
        if len(returns) < window or len(benchmark_returns) < window:
            logger.warning(f"Недостаточно данных для расчета скользящей альфы с окном {window}")
            return pd.Series()

        # Рассчитываем скользящую бету
        rolling_beta = calculate_beta(returns, benchmark_returns, window)

        # Рассчитываем скользящее среднее
        rolling_returns_mean = returns.rolling(window=window).mean()
        rolling_benchmark_mean = benchmark_returns.rolling(window=window).mean()

        # Рассчитываем скользящую альфу
        rolling_alpha = rolling_returns_mean - daily_risk_free - rolling_beta * (
                    rolling_benchmark_mean - daily_risk_free)

        # Годовая альфа
        annual_rolling_alpha = (1 + rolling_alpha) ** 252 - 1

        return annual_rolling_alpha