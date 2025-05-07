"""
Модуль для модели на основе технических индикаторов.
"""
import logging
import os
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report

from src.models.base_model import BaseModel
from src.data.features import TechnicalAnalysis

logger = logging.getLogger(__name__)


class TechnicalModel(BaseModel):
    """
    Модель для принятия торговых решений на основе технических индикаторов.
    Использует методы машинного обучения для определения оптимальных моментов входа/выхода.
    """

    def __init__(self, name: str, model_params: Dict = None):
        """
        Инициализирует модель на основе технических индикаторов.

        Args:
            name: Имя модели
            model_params: Параметры модели
                - classifier: Тип классификатора ('rf', 'gb', 'svm', 'lr', 'mlp')
                - signal_threshold: Порог для генерации сигнала
                - strategy: Стратегия для генерации сигналов
                - feature_importance: Использовать ли анализ важности признаков
                - n_estimators: Количество деревьев (для RF, GB)
                - max_depth: Максимальная глубина деревьев (для RF, GB)
                - C: Параметр регуляризации (для SVM, LR)
                - kernel: Тип ядра (для SVM)
                - hidden_layer_sizes: Размеры скрытых слоев (для MLP)
                - class_weight: Веса классов
        """
        # Параметры по умолчанию
        default_params = {
            "classifier": "rf",
            "signal_threshold": 0.5,
            "strategy": "combined",
            "feature_importance": True,
            "n_estimators": 100,
            "max_depth": 10,
            "C": 1.0,
            "kernel": "rbf",
            "hidden_layer_sizes": (100, 50),
            "class_weight": "balanced",
            "random_state": 42
        }

        # Объединяем параметры по умолчанию с переданными
        if model_params:
            default_params.update(model_params)

        super().__init__(name, default_params)

        self.technical_analyzer = TechnicalAnalysis()

    def build(self) -> None:
        """
        Строит классификатор с заданными параметрами.
        """
        classifier_type = self.model_params.get("classifier", "rf").lower()

        if classifier_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=self.model_params.get("n_estimators", 100),
                max_depth=self.model_params.get("max_depth", 10),
                class_weight=self.model_params.get("class_weight", "balanced"),
                random_state=self.model_params.get("random_state", 42),
                n_jobs=-1
            )

        elif classifier_type == "gb":
            self.model = GradientBoostingClassifier(
                n_estimators=self.model_params.get("n_estimators", 100),
                max_depth=self.model_params.get("max_depth", 10),
                random_state=self.model_params.get("random_state", 42)
            )

        elif classifier_type == "svm":
            self.model = SVC(
                C=self.model_params.get("C", 1.0),
                kernel=self.model_params.get("kernel", "rbf"),
                class_weight=self.model_params.get("class_weight", "balanced"),
                random_state=self.model_params.get("random_state", 42),
                probability=True
            )

        elif classifier_type == "lr":
            self.model = LogisticRegression(
                C=self.model_params.get("C", 1.0),
                class_weight=self.model_params.get("class_weight", "balanced"),
                random_state=self.model_params.get("random_state", 42),
                max_iter=1000,
                n_jobs=-1
            )

        elif classifier_type == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=self.model_params.get("hidden_layer_sizes", (100, 50)),
                activation=self.model_params.get("activation", "relu"),
                solver=self.model_params.get("solver", "adam"),
                alpha=self.model_params.get("alpha", 0.0001),
                learning_rate=self.model_params.get("learning_rate", "adaptive"),
                max_iter=1000,
                random_state=self.model_params.get("random_state", 42)
            )

        else:
            raise ValueError(f"Неподдерживаемый тип классификатора: {classifier_type}")

        logger.info(f"Построена модель {self.name} с классификатором {classifier_type}")

    def prepare_data_from_dataframe(
            self,
            df: pd.DataFrame,
            **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготавливает данные для модели из DataFrame.
        Генерирует сигналы и преобразует их в метки классов.

        Args:
            df: DataFrame с данными и техническими индикаторами
            **kwargs: Дополнительные аргументы для подготовки данных

        Returns:
            Кортеж с признаками (X) и целевыми значениями (y)
        """
        # Генерируем торговые сигналы
        strategy = kwargs.get("strategy", self.model_params.get("strategy", "combined"))
        signal_threshold = kwargs.get("signal_threshold", self.model_params.get("signal_threshold", 0.5))

        data_with_signals = self.technical_analyzer.generate_trading_signals(
            df=df,
            strategy=strategy,
            signal_threshold=signal_threshold
        )

        # Выбираем признаки для модели
        exclude_columns = ['signal', 'buy_signal', 'sell_signal', 'hold_signal']
        # Также исключаем 'open', 'high', 'low', 'close', 'volume' из признаков
        exclude_columns.extend(['open', 'high', 'low', 'close', 'volume'])

        # Предполагаем, что целевые колонки уже созданы ('buy_signal', 'sell_signal', 'hold_signal')
        feature_columns = [col for col in data_with_signals.columns if col not in exclude_columns]

        # Создаем мультиклассовую целевую переменную
        # 0 - держать, 1 - покупать, 2 - продавать
        y = np.zeros(len(data_with_signals))
        y[data_with_signals['buy_signal'] == 1] = 1
        y[data_with_signals['sell_signal'] == 1] = 2

        # Выбираем признаки
        X = data_with_signals[feature_columns].values

        # Сохраняем имена признаков
        self.feature_names = feature_columns

        # Удаляем строки с NaN
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]

        logger.info(f"Подготовлены данные: {X.shape[0]} образцов, {X.shape[1]} признаков")

        return X, y

    def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            **kwargs
    ) -> Dict:
        """
        Обучает модель на заданных данных.

        Args:
            X_train: Обучающие данные (признаки)
            y_train: Обучающие целевые значения
            X_val: Валидационные данные (признаки, опционально)
            y_val: Валидационные целевые значения (опционально)
            **kwargs: Дополнительные аргументы для обучения

        Returns:
            Словарь с историей обучения и метриками
        """
        # Проверяем, построена ли модель
        if self.model is None:
            self.build()

        # Обучаем модель
        logger.info(f"Начинаем обучение модели {self.name}")

        self.model.fit(X_train, y_train)

        # Вычисляем метрики на обучающем наборе
        y_train_pred = self.model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, y_train_pred)

        # Если есть валидационный набор, вычисляем метрики на нем
        val_metrics = None
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, y_val_pred)

        # Анализ важности признаков (если доступно)
        if self.model_params.get("feature_importance", True) and hasattr(self.model, "feature_importances_"):
            feature_importances = self.model.feature_importances_

            # Создаем словарь с важностью признаков
            importance_dict = dict(zip(self.feature_names, feature_importances))

            # Сортируем по убыванию важности
            sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

            # Сохраняем в метаданные
            self.metadata["feature_importances"] = sorted_importances

            # Логируем топ-10 признаков
            logger.info(f"Топ-10 важных признаков:")
            for feature, importance in sorted_importances[:10]:
                logger.info(f"{feature}: {importance:.4f}")

        # Устанавливаем флаг обученной модели
        self.is_trained = True

        # Сохраняем метрики в метаданные
        self.metadata["train_metrics"] = train_metrics
        if val_metrics:
            self.metadata["val_metrics"] = val_metrics

        logger.info(f"Обучение модели {self.name} завершено")
        logger.info(
            f"Метрики на обучающем наборе: accuracy={train_metrics['accuracy']:.4f}, f1={train_metrics['f1']:.4f}")

        if val_metrics:
            logger.info(
                f"Метрики на валидационном наборе: accuracy={val_metrics['accuracy']:.4f}, f1={val_metrics['f1']:.4f}")

        return {
            "train_metrics": train_metrics,
            "val_metrics": val_metrics
        }

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Делает прогнозы (классы) на основе входных данных.

        Args:
            X: Входные данные для прогнозирования
            **kwargs: Дополнительные аргументы для прогнозирования

        Returns:
            Массив прогнозов (0: держать, 1: покупать, 2: продавать)
        """
        if not self.is_trained:
            logger.warning(f"Модель {self.name} не обучена, невозможно сделать прогноз")
            return np.array([])

        # Делаем прогноз
        predictions = self.model.predict(X)

        return predictions

    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Делает прогнозы (вероятности классов) на основе входных данных.

        Args:
            X: Входные данные для прогнозирования
            **kwargs: Дополнительные аргументы для прогнозирования

        Returns:
            Массив вероятностей классов
        """
        if not self.is_trained:
            logger.warning(f"Модель {self.name} не обучена, невозможно сделать прогноз")
            return np.array([])

        # Проверяем, поддерживает ли модель predict_proba
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            return probabilities
        else:
            logger.warning(f"Модель {self.name} не поддерживает прогнозирование вероятностей")
            return np.array([])

    def generate_signals_from_dataframe(
            self,
            df: pd.DataFrame,
            **kwargs
    ) -> pd.DataFrame:
        """
        Генерирует торговые сигналы для DataFrame.

        Args:
            df: DataFrame с данными и техническими индикаторами
            **kwargs: Дополнительные аргументы для генерации сигналов

        Returns:
            DataFrame с добавленными сигналами
        """
        if not self.is_trained:
            logger.warning(f"Модель {self.name} не обучена, невозможно сгенерировать сигналы")
            return df

        # Создаем копию DataFrame
        data = df.copy()

        # Выбираем признаки
        exclude_columns = ['signal', 'buy_signal', 'sell_signal', 'hold_signal']
        exclude_columns.extend(['open', 'high', 'low', 'close', 'volume'])

        feature_columns = self.feature_names or [col for col in data.columns if col not in exclude_columns]

        # Проверяем, есть ли все необходимые признаки
        missing_features = [col for col in feature_columns if col not in data.columns]
        if missing_features:
            logger.warning(f"Отсутствуют признаки: {missing_features}")
            return data

        # Получаем признаки
        X = data[feature_columns].values

        # Удаляем строки с NaN
        valid_indices = ~np.isnan(X).any(axis=1)

        # Если все строки содержат NaN, возвращаем исходный DataFrame
        if not valid_indices.any():
            logger.warning("Все строки содержат NaN, невозможно сгенерировать сигналы")
            return data

        # Подготавливаем массив для хранения сигналов (изначально все 0 - держать)
        signals = np.zeros(len(data))

        # Делаем прогноз только для валидных индексов
        valid_X = X[valid_indices]
        valid_predictions = self.model.predict(valid_X)

        # Заполняем массив сигналов
        signals[valid_indices] = valid_predictions

        # Добавляем прогнозы в DataFrame
        data['predicted_signal'] = signals

        # Разбиваем прогнозы на отдельные столбцы
        data['predicted_hold'] = (data['predicted_signal'] == 0).astype(int)
        data['predicted_buy'] = (data['predicted_signal'] == 1).astype(int)
        data['predicted_sell'] = (data['predicted_signal'] == 2).astype(int)

        # Если модель поддерживает прогнозирование вероятностей, добавляем их
        if hasattr(self.model, 'predict_proba'):
            # Создаем массив для хранения вероятностей
            probabilities = np.zeros((len(data), 3))

            # Делаем прогноз только для валидных индексов
            valid_probas = self.model.predict_proba(valid_X)

            # Заполняем массив вероятностей
            probabilities[valid_indices] = valid_probas

            # Добавляем вероятности в DataFrame
            data['hold_probability'] = probabilities[:, 0]
            data['buy_probability'] = probabilities[:, 1]
            data['sell_probability'] = probabilities[:, 2]

        logger.info(f"Сгенерированы сигналы для {valid_indices.sum()} из {len(data)} строк")

        return data

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Вычисляет метрики классификации.

        Args:
            y_true: Истинные метки классов
            y_pred: Предсказанные метки классов

        Returns:
            Словарь с метриками
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average='macro'),
            "recall_macro": recall_score(y_true, y_pred, average='macro'),
            "f1_macro": f1_score(y_true, y_pred, average='macro'),
            "precision_weighted": precision_score(y_true, y_pred, average='weighted'),
            "recall_weighted": recall_score(y_true, y_pred, average='weighted'),
            "f1_weighted": f1_score(y_true, y_pred, average='weighted'),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }

        # Добавляем метрики для каждого класса
        for cls in np.unique(np.concatenate([y_true, y_pred])):
            cls_name = {0: "hold", 1: "buy", 2: "sell"}.get(cls, str(cls))

            metrics[f"precision_{cls_name}"] = precision_score(y_true, y_pred, average=None, labels=[cls])[0]
            metrics[f"recall_{cls_name}"] = recall_score(y_true, y_pred, average=None, labels=[cls])[0]
            metrics[f"f1_{cls_name}"] = f1_score(y_true, y_pred, average=None, labels=[cls])[0]

        # Для удобства использования в других методах
        metrics["precision"] = metrics["precision_weighted"]
        metrics["recall"] = metrics["recall_weighted"]
        metrics["f1"] = metrics["f1_weighted"]

        return metrics

    def backtest(
            self,
            df: pd.DataFrame,
            initial_balance: float = 10000.0,
            position_size: float = 1.0,
            fee: float = 0.001,
            stop_loss: Optional[float] = None,
            take_profit: Optional[float] = None
    ) -> Dict:
        """
        Выполняет бэктестинг модели на исторических данных.

        Args:
            df: DataFrame с историческими данными
            initial_balance: Начальный баланс
            position_size: Размер позиции (доля от баланса)
            fee: Комиссия за сделку (например, 0.001 = 0.1%)
            stop_loss: Уровень стоп-лосса (например, 0.02 = 2%)
            take_profit: Уровень тейк-профита (например, 0.05 = 5%)

        Returns:
            Словарь с результатами бэктестинга
        """
        # Генерируем сигналы
        data = self.generate_signals_from_dataframe(df)

        # Создаем копию для бэктестинга
        backtest_data = data.copy()

        # Добавляем колонки для отслеживания баланса и позиции
        backtest_data['balance'] = initial_balance
        backtest_data['position'] = 0.0
        backtest_data['position_value'] = 0.0
        backtest_data['equity'] = initial_balance

        # Информация о текущей позиции
        current_position = 0.0
        entry_price = 0.0

        # Список для хранения сделок
        trades = []

        # Проходим по данным и симулируем торговлю
        for i in range(1, len(backtest_data)):
            prev_idx = i - 1

            # Получаем текущую цену и сигнал
            current_price = backtest_data.iloc[i]['close']
            signal = backtest_data.iloc[i]['predicted_signal']

            # Текущий баланс и стоимость позиции
            current_balance = backtest_data.iloc[prev_idx]['balance']
            current_position_value = current_position * current_price

            # Проверяем стоп-лосс и тейк-профит, если есть позиция
            if current_position > 0 and (stop_loss is not None or take_profit is not None):
                price_change_pct = (current_price - entry_price) / entry_price

                # Срабатывание стоп-лосса
                if stop_loss is not None and price_change_pct <= -stop_loss:
                    # Закрываем позицию
                    sale_value = current_position_value * (1 - fee)
                    current_balance += sale_value

                    # Записываем сделку
                    trades.append({
                        'type': 'stop_loss',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': current_position,
                        'profit_loss': sale_value - (current_position * entry_price),
                        'profit_loss_pct': price_change_pct * 100
                    })

                    # Обнуляем позицию
                    current_position = 0.0
                    entry_price = 0.0

                # Срабатывание тейк-профита
                elif take_profit is not None and price_change_pct >= take_profit:
                    # Закрываем позицию
                    sale_value = current_position_value * (1 - fee)
                    current_balance += sale_value

                    # Записываем сделку
                    trades.append({
                        'type': 'take_profit',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': current_position,
                        'profit_loss': sale_value - (current_position * entry_price),
                        'profit_loss_pct': price_change_pct * 100
                    })

                    # Обнуляем позицию
                    current_position = 0.0
                    entry_price = 0.0

            # Сигнал на покупку
            if signal == 1 and current_position == 0:
                # Рассчитываем размер позиции
                position_value = current_balance * position_size

                # Учитываем комиссию
                effective_position_value = position_value / (1 + fee)
                current_position = effective_position_value / current_price

                # Обновляем баланс
                current_balance -= position_value

                # Запоминаем цену входа
                entry_price = current_price

            # Сигнал на продажу
            elif signal == 2 and current_position > 0:
                # Закрываем позицию
                sale_value = current_position_value * (1 - fee)
                current_balance += sale_value

                # Рассчитываем прибыль/убыток
                entry_value = current_position * entry_price
                profit_loss = sale_value - entry_value
                profit_loss_pct = (current_price - entry_price) / entry_price * 100

                # Записываем сделку
                trades.append({
                    'type': 'sell_signal',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position': current_position,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': profit_loss_pct
                })

                # Обнуляем позицию
                current_position = 0.0
                entry_price = 0.0

            # Обновляем данные
            backtest_data.loc[backtest_data.index[i], 'balance'] = current_balance
            backtest_data.loc[backtest_data.index[i], 'position'] = current_position
            backtest_data.loc[backtest_data.index[i], 'position_value'] = current_position * current_price
            backtest_data.loc[backtest_data.index[i], 'equity'] = current_balance + (current_position * current_price)

        # Рассчитываем метрики бэктестинга
        initial_equity = initial_balance
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

            # Создаем результаты бэктестинга
            backtest_results = {
                'initial_equity': initial_equity,
                'final_equity': final_equity,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
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
                'trades': trades,
                'equity_curve': backtest_data['equity'].values
            }
        else:
            # Если сделок не было
            backtest_results = {
                'initial_equity': initial_equity,
                'final_equity': final_equity,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
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
                'trades': [],
                'equity_curve': backtest_data['equity'].values
            }

        # Логируем результаты
        logger.info(f"Результаты бэктестинга модели {self.name}:")
        logger.info(f"Начальный капитал: {initial_equity:.2f}")
        logger.info(f"Конечный капитал: {final_equity:.2f}")
        logger.info(f"Общая доходность: {backtest_results['total_return_pct']:.2f}%")
        logger.info(f"Максимальная просадка: {backtest_results['max_drawdown_pct']:.2f}%")
        logger.info(f"Всего сделок: {backtest_results['total_trades']}")

        if backtest_results['total_trades'] > 0:
            logger.info(
                f"Прибыльных сделок: {backtest_results['profitable_trades']} ({backtest_results['win_rate_pct']:.2f}%)")
            logger.info(
                f"Убыточных сделок: {backtest_results['losing_trades']} ({100 - backtest_results['win_rate_pct']:.2f}%)")
            logger.info(f"Отношение прибыли к убытку: {backtest_results['profit_loss_ratio']:.2f}")
            logger.info(f"Ожидаемая прибыль на сделку: {backtest_results['expected_profit']:.2f}")
            logger.info(f"Фактор восстановления: {backtest_results['recovery_factor']:.2f}")

        return backtest_results

    def optimize_parameters(
            self,
            df: pd.DataFrame,
            param_grid: Dict[str, List],
            cv: int = 3,
            scoring: str = 'f1_weighted',
            n_jobs: int = -1
    ) -> Dict:
        """
        Оптимизирует параметры модели с использованием поиска по сетке.

        Args:
            df: DataFrame с данными
            param_grid: Словарь с параметрами для поиска
            cv: Число фолдов для кросс-валидации
            scoring: Метрика для оптимизации
            n_jobs: Число процессов для параллельного выполнения

        Returns:
            Словарь с лучшими параметрами и результатами
        """
        from sklearn.model_selection import GridSearchCV

        # Подготавливаем данные
        X, y = self.prepare_data_from_dataframe(df)

        # Строим базовую модель
        self.build()

        # Создаем поиск по сетке
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )

        # Выполняем поиск
        logger.info(f"Начинаем оптимизацию параметров модели {self.name}")
        grid_search.fit(X, y)

        # Логируем результаты
        logger.info(f"Оптимизация параметров завершена")
        logger.info(f"Лучшие параметры: {grid_search.best_params_}")
        logger.info(f"Лучшая оценка: {grid_search.best_score_:.4f}")

        # Обновляем параметры модели
        self.model_params.update(grid_search.best_params_)

        # Перестраиваем модель с лучшими параметрами
        self.build()

        # Возвращаем результаты
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }