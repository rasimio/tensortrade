"""
Модуль для гибридной модели, объединяющей RL, LSTM и технический анализ.
"""
import logging
import os
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd

from src.models.base_model import BaseModel
from src.models.lstm import LSTMModel
from src.models.reinforcement import RLModel
from src.models.technical import TechnicalModel

logger = logging.getLogger(__name__)


class HybridModel(BaseModel):
    """
    Гибридная модель, объединяющая подходы RL, LSTM и технический анализ
    для достижения максимальной доходности при торговле.
    """

    def __init__(self, name: str, model_params: Dict = None):
        """
        Инициализирует гибридную модель.

        Args:
            name: Имя модели
            model_params: Параметры модели
                - lstm_params: Параметры LSTM модели
                - rl_params: Параметры RL модели
                - technical_params: Параметры модели технического анализа
                - alpha: Вес для LSTM модели
                - beta: Вес для RL модели
                - gamma: Вес для технической модели
                - decision_threshold: Порог для принятия решения
                - use_combined_features: Использовать ли совместные признаки
                - sequence_length: Длина последовательности для временных признаков
        """
        # Параметры по умолчанию
        default_params = {
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
                "policy": "MlpPolicy",
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "batch_size": 64,
                "window_size": 60,
                "env_params": {
                    "fee": 0.001,
                    "initial_balance": 10000.0,
                    "max_position": 1,
                    "reward_scaling": 1.0
                }
            },
            "technical_params": {
                "classifier": "rf",
                "signal_threshold": 0.5,
                "strategy": "combined",
                "feature_importance": True,
                "n_estimators": 100,
                "max_depth": 10
            },
            "alpha": 0.3,  # Вес для LSTM
            "beta": 0.4,  # Вес для RL
            "gamma": 0.3,  # Вес для технического анализа
            "decision_threshold": 0.5,
            "use_combined_features": True,
            "sequence_length": 60
        }

        # Объединяем параметры по умолчанию с переданными
        if model_params:
            # Для вложенных словарей
            if model_params.get("lstm_params") and default_params.get("lstm_params"):
                default_params["lstm_params"].update(model_params.get("lstm_params", {}))
                # Удаляем, чтобы не переопределить полностью при следующем update
                if "lstm_params" in model_params:
                    del model_params["lstm_params"]

            if model_params.get("rl_params") and default_params.get("rl_params"):
                # Обработка env_params для RL
                if model_params.get("rl_params", {}).get("env_params") and default_params.get("rl_params", {}).get(
                        "env_params"):
                    default_params["rl_params"]["env_params"].update(
                        model_params.get("rl_params", {}).get("env_params", {}))
                    # Удаляем, чтобы не переопределить полностью при следующем update
                    if "env_params" in model_params.get("rl_params", {}):
                        del model_params["rl_params"]["env_params"]

                default_params["rl_params"].update(model_params.get("rl_params", {}))
                # Удаляем, чтобы не переопределить полностью при следующем update
                if "rl_params" in model_params:
                    del model_params["rl_params"]

            if model_params.get("technical_params") and default_params.get("technical_params"):
                default_params["technical_params"].update(model_params.get("technical_params", {}))
                # Удаляем, чтобы не переопределить полностью при следующем update
                if "technical_params" in model_params:
                    del model_params["technical_params"]

            default_params.update(model_params)

        super().__init__(name, default_params)

        # Создаем подмодели
        self.lstm_model = None
        self.rl_model = None
        self.technical_model = None

    def build(self) -> None:
        """
        Создает и инициализирует все подмодели.
        """
        try:
            # Создаем LSTM модель
            lstm_params = self.model_params.get("lstm_params", {})
            self.lstm_model = LSTMModel(f"{self.name}_lstm", lstm_params)
            # Строим модель, чтобы инициализировать её внутренние компоненты
            self.lstm_model.build()
            logger.info(f"LSTM модель {self.lstm_model.name} создана")
        except Exception as e:
            logger.error(f"Ошибка при создании LSTM модели: {e}")
            self.lstm_model = None

        try:
            # Создаем RL модель
            rl_params = self.model_params.get("rl_params", {})
            self.rl_model = RLModel(f"{self.name}_rl", rl_params)
            logger.info(f"RL модель {self.rl_model.name} создана")
        except Exception as e:
            logger.error(f"Ошибка при создании RL модели: {e}")
            self.rl_model = None

        try:
            # Создаем модель технического анализа
            technical_params = self.model_params.get("technical_params", {})
            self.technical_model = TechnicalModel(f"{self.name}_technical", technical_params)
            # Строим модель, чтобы инициализировать её внутренние компоненты
            self.technical_model.build()
            logger.info(f"Техническая модель {self.technical_model.name} создана")
        except Exception as e:
            logger.error(f"Ошибка при создании технической модели: {e}")
            self.technical_model = None

        # Установим флаг, что модель построена
        self.is_built = True

        logger.info(f"Создана гибридная модель {self.name}")

    def train(
            self,
            X_train: Optional[np.ndarray] = None,
            y_train: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            data: Optional[pd.DataFrame] = None,
            feature_columns: Optional[List[str]] = None,
            **kwargs
    ) -> Dict:
        """
        Обучает гибридную модель на заданных данных.

        Args:
            X_train: Обучающие данные (признаки)
            y_train: Обучающие целевые значения
            X_val: Валидационные данные (признаки)
            y_val: Валидационные целевые значения
            data: DataFrame с данными для обучения
            feature_columns: Список колонок с признаками
            **kwargs: Дополнительные аргументы для обучения

        Returns:
            Словарь с историей обучения и метриками
        """
        if self.lstm_model is None or self.rl_model is None or self.technical_model is None:
            self.build()

        # Проверяем, переданы ли необходимые данные
        if data is None and (X_train is None or y_train is None):
            raise ValueError(
                "Для обучения гибридной модели необходимо передать DataFrame с данными или X_train и y_train")

        # Если передан DataFrame, используем его для подготовки данных
        if data is not None:
            logger.info(f"Подготовка данных из DataFrame для обучения гибридной модели")

            # Если не переданы колонки с признаками, используем все числовые колонки
            if feature_columns is None:
                feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()

                # Исключаем целевые колонки и колонки с сигналами
                exclude_columns = ['buy_signal', 'sell_signal', 'hold_signal', 'signal',
                                   'predicted_signal', 'predicted_buy', 'predicted_sell', 'predicted_hold']
                feature_columns = [col for col in feature_columns if col not in exclude_columns]

            # Сохраняем имена признаков
            self.feature_names = feature_columns

            # Подготавливаем данные для LSTM
            if kwargs.get("train_lstm", True):
                logger.info(f"Подготовка данных для обучения LSTM модели")

                # Подготавливаем последовательности для LSTM
                sequence_length = self.model_params.get("sequence_length", 60)

                # Создаем целевую переменную (будущая цена)
                target_column = kwargs.get("lstm_target", "close")
                prediction_horizon = kwargs.get("lstm_prediction_horizon", 1)

                # Используем цены закрытия с задержкой как целевую переменную
                # Нормализуем целевую переменную, используя проценты изменения вместо абсолютных цен
                price_data = data[target_column].values
                future_prices_pct = np.zeros_like(price_data[:-prediction_horizon])
                for i in range(len(price_data) - prediction_horizon):
                    # Рассчитываем процент изменения цены
                    future_prices_pct[i] = (price_data[i + prediction_horizon] / price_data[i]) - 1.0

                # Разделяем данные на обучающую и валидационную выборки
                train_size = int(len(data) * 0.8)

                # Обучающие данные
                train_data = data.iloc[:train_size].copy()
                train_features = train_data[feature_columns].values
                train_target = future_prices_pct[:train_size]

                # Создаем последовательности для LSTM
                X_lstm_train, y_lstm_train = [], []
                for i in range(len(train_features) - sequence_length):
                    X_lstm_train.append(train_features[i:i + sequence_length])
                    y_lstm_train.append(train_target[i + sequence_length - 1])

                X_lstm_train = np.array(X_lstm_train)
                y_lstm_train = np.array(y_lstm_train)

                # Валидационные данные
                if train_size < len(data) - sequence_length:
                    val_data = data.iloc[train_size:].copy()
                    val_features = val_data[feature_columns].values
                    val_target = future_prices_pct[train_size:]

                    X_lstm_val, y_lstm_val = [], []
                    for i in range(len(val_features) - sequence_length):
                        X_lstm_val.append(val_features[i:i + sequence_length])
                        y_lstm_val.append(val_target[i + sequence_length - 1])

                    X_lstm_val = np.array(X_lstm_val)
                    y_lstm_val = np.array(y_lstm_val)
                else:
                    X_lstm_val, y_lstm_val = None, None

            # Подготавливаем данные для модели технического анализа
            if kwargs.get("train_technical", True):
                logger.info(f"Подготовка данных для обучения модели технического анализа")

                # Если данные для технического анализа не переданы отдельно, генерируем их
                X_tech, y_tech = self.technical_model.prepare_data_from_dataframe(data)

                # Разделяем данные на обучающую и валидационную выборки
                train_size_tech = int(len(X_tech) * 0.8)

                X_tech_train = X_tech[:train_size_tech]
                y_tech_train = y_tech[:train_size_tech]

                X_tech_val = X_tech[train_size_tech:]
                y_tech_val = y_tech[train_size_tech:]

        else:
            # Используем переданные данные
            X_lstm_train, y_lstm_train = X_train, y_train
            X_lstm_val, y_lstm_val = X_val, y_val

            X_tech_train, y_tech_train = X_train, y_train
            X_tech_val, y_tech_val = X_val, y_val

        # Обучаем LSTM модель
        lstm_metrics = None
        if kwargs.get("train_lstm", True):
            logger.info(f"Обучение LSTM модели")

            # Устанавливаем имена признаков
            self.lstm_model.set_feature_names(self.feature_names)

            # Обучаем модель
            lstm_metrics = self.lstm_model.train(
                X_lstm_train, y_lstm_train,
                X_lstm_val, y_lstm_val,
                **{k.replace("lstm_", ""): v for k, v in kwargs.items() if k.startswith("lstm_")}
            )

        # Обучаем RL модель
        rl_metrics = None
        if kwargs.get("train_rl", True) and data is not None:
            logger.info(f"Обучение RL модели")

            # Устанавливаем имена признаков
            self.rl_model.set_feature_names(self.feature_names)

            # Обучаем модель
            rl_metrics = self.rl_model.train(
                None, None, None, None,
                data=data,
                feature_columns=feature_columns,
                total_timesteps=kwargs.get("rl_total_timesteps", 100000),
                **{k.replace("rl_", ""): v for k, v in kwargs.items() if k.startswith("rl_")}
            )

        # Обучаем модель технического анализа
        technical_metrics = None
        if kwargs.get("train_technical", True):
            logger.info(f"Обучение модели технического анализа")

            # Устанавливаем имена признаков
            self.technical_model.set_feature_names(self.feature_names)

            # Обучаем модель
            technical_metrics = self.technical_model.train(
                X_tech_train, y_tech_train,
                X_tech_val, y_tech_val,
                **{k.replace("technical_", ""): v for k, v in kwargs.items() if k.startswith("technical_")}
            )

        # Устанавливаем флаг обученной модели
        self.is_trained = (
                (not kwargs.get("train_lstm", True) or self.lstm_model.is_trained) and
                (not kwargs.get("train_rl", True) or self.rl_model.is_trained) and
                (not kwargs.get("train_technical", True) or self.technical_model.is_trained)
        )

        # Сохраняем результаты в метаданные
        self.metadata["lstm_metrics"] = lstm_metrics
        self.metadata["rl_metrics"] = rl_metrics
        self.metadata["technical_metrics"] = technical_metrics

        # Логируем результаты
        logger.info(f"Обучение гибридной модели {self.name} завершено")

        return {
            "lstm_metrics": lstm_metrics,
            "rl_metrics": rl_metrics,
            "technical_metrics": technical_metrics,
            "is_trained": self.is_trained
        }

    def predict(
            self,
            X: np.ndarray,
            return_probabilities: bool = False,
            **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Делает прогнозы на основе входных данных.

        Args:
            X: Входные данные для прогнозирования
            return_probabilities: Возвращать ли вероятности классов
            **kwargs: Дополнительные аргументы для прогнозирования

        Returns:
            Массив прогнозов или кортеж с прогнозами и вероятностями
        """
        if not self.is_trained:
            logger.warning(f"Модель {self.name} не обучена полностью, прогноз может быть неточным")

        # Проверяем инициализацию всех подмоделей
        if not hasattr(self, 'lstm_model') or self.lstm_model is None:
            logger.warning("LSTM модель не инициализирована")
            lstm_predictions = None
        elif not hasattr(self.lstm_model, 'is_trained') or not self.lstm_model.is_trained:
            logger.warning("LSTM модель не обучена")
            lstm_predictions = None
        else:
            # Используем LSTM модель
            try:
                # Для LSTM нужна 3D форма данных (samples, sequence_length, features)
                if len(X.shape) == 2:
                    sequence_length = self.model_params.get("sequence_length", 60)
                    if X.shape[0] >= sequence_length:
                        # Преобразуем в последовательность, если достаточно данных
                        lstm_input = np.array([X[-sequence_length:]])
                    else:
                        # Если данных недостаточно, используем все доступные
                        lstm_input = np.array([X])
                else:
                    lstm_input = X

                # Получаем прогноз
                lstm_predictions = self.lstm_model.predict(lstm_input)

                # Преобразуем прогноз процентного изменения цены в абсолютное значение
                current_price = X[-1, 0] if len(X.shape) == 2 else X[0, -1, 0]
                lstm_predictions = self.lstm_model.predict(lstm_input, current_price=current_price)

                # Преобразуем прогноз цены в направление движения
                if lstm_predictions.shape[0] > 0:
                    lstm_predictions = (lstm_predictions > X[-1, 0]).astype(int)  # 1 - рост, 0 - падение

            except Exception as e:
                logger.error(f"Ошибка при получении прогноза от LSTM модели: {e}")
                lstm_predictions = None

        # Аналогичные проверки для RL и технической моделей
        if not hasattr(self, 'rl_model') or self.rl_model is None:
            logger.warning("RL модель не инициализирована")
            rl_predictions = None
        elif not hasattr(self.rl_model, 'is_trained') or not self.rl_model.is_trained:
            logger.warning("RL модель не обучена")
            rl_predictions = None
        else:
            try:
                # Код для получения прогноза от RL модели
                # ... (ваш существующий код)
                if len(X.shape) == 2:
                    sequence_length = self.model_params.get("sequence_length", 60)
                    if X.shape[0] >= sequence_length:
                        # Преобразуем в последовательность, если достаточно данных
                        rl_input = np.array([X[-sequence_length:]])
                    else:
                        # Если данных недостаточно, используем все доступные
                        rl_input = np.array([X])
                else:
                    rl_input = X

                # Получаем прогноз
                rl_predictions = self.rl_model.predict(rl_input)

            except Exception as e:
                logger.error(f"Ошибка при получении прогноза от RL модели: {e}")
                rl_predictions = None

        if not hasattr(self, 'technical_model') or self.technical_model is None:
            logger.warning("Техническая модель не инициализирована")
            technical_predictions = None
            technical_probas = None
        elif not hasattr(self.technical_model, 'is_trained') or not self.technical_model.is_trained:
            logger.warning("Техническая модель не обучена")
            technical_predictions = None
            technical_probas = None
        else:
            try:
                # Получаем прогноз
                technical_predictions = self.technical_model.predict(X)

                # Получаем вероятности
                technical_probas = None
                if hasattr(self.technical_model, 'predict_proba'):
                    technical_probas = self.technical_model.predict_proba(X)

            except Exception as e:
                logger.error(f"Ошибка при получении прогноза от модели технического анализа: {e}")
                technical_predictions = None
                technical_probas = None

        # Далее продолжаем с существующим кодом объединения прогнозов...
        # ... (ваш существующий код)

        # Объединяем прогнозы
        alpha = self.model_params.get("alpha", 0.3)  # Вес для LSTM
        beta = self.model_params.get("beta", 0.4)  # Вес для RL
        gamma = self.model_params.get("gamma", 0.3)  # Вес для технического анализа

        # Проверяем, какие прогнозы доступны, и нормализуем веса
        available_models_count = 0
        if lstm_predictions is not None:
            available_models_count += 1
        if rl_predictions is not None:
            available_models_count += 1
        if technical_predictions is not None:
            available_models_count += 1

        if available_models_count == 0:
            logger.warning(f"Ни одна подмодель не сделала прогноз")
            return np.array([0])  # Возвращаем сигнал "держать" по умолчанию

        # Нормализуем веса
        norm_alpha = alpha / (alpha + beta + gamma) if lstm_predictions is not None else 0
        norm_beta = beta / (alpha + beta + gamma) if rl_predictions is not None else 0
        norm_gamma = gamma / (alpha + beta + gamma) if technical_predictions is not None else 0

        # Вычисляем взвешенные вероятности
        buy_probability = 0.0
        sell_probability = 0.0
        hold_probability = 0.0

        # Преобразуем прогнозы в вероятности
        if lstm_predictions is not None:
            # LSTM прогнозирует движение цены, преобразуем в сигналы
            if lstm_predictions.item() == 1:  # Цена будет расти
                buy_probability += norm_alpha
            else:  # Цена будет падать
                sell_probability += norm_alpha

        if rl_predictions is not None:
            # RL возвращает действия: 0 - держать, 1 - покупать, 2 - продавать
            if rl_predictions.item() == 0:
                hold_probability += norm_beta
            elif rl_predictions.item() == 1:
                buy_probability += norm_beta
            elif rl_predictions.item() == 2:
                sell_probability += norm_beta

        if technical_predictions is not None:
            # Технический анализ возвращает классы: 0 - держать, 1 - покупать, 2 - продавать
            if technical_predictions.item() == 0:
                hold_probability += norm_gamma
            elif technical_predictions.item() == 1:
                buy_probability += norm_gamma
            elif technical_predictions.item() == 2:
                sell_probability += norm_gamma

        # Принимаем решение на основе вероятностей
        decision_threshold = self.model_params.get("decision_threshold", 0.5)

        if buy_probability > decision_threshold and buy_probability > sell_probability:
            final_prediction = 1  # Покупать
        elif sell_probability > decision_threshold and sell_probability > buy_probability:
            final_prediction = 2  # Продавать
        else:
            final_prediction = 0  # Держать

        if return_probabilities:
            probabilities = {
                "buy_probability": buy_probability,
                "sell_probability": sell_probability,
                "hold_probability": hold_probability,
                "lstm_prediction": lstm_predictions.item() if lstm_predictions is not None else None,
                "rl_prediction": rl_predictions.item() if rl_predictions is not None else None,
                "technical_prediction": technical_predictions.item() if technical_predictions is not None else None
            }
            return np.array([final_prediction]), probabilities

        return np.array([final_prediction])

    def predict_from_dataframe(
            self,
            df: pd.DataFrame,
            feature_columns: Optional[List[str]] = None,
            return_probabilities: bool = True
    ) -> pd.DataFrame:
        """
        Делает прогнозы на основе DataFrame с данными.

        Args:
            df: DataFrame с данными
            feature_columns: Список колонок с признаками
            return_probabilities: Возвращать ли вероятности классов

        Returns:
            DataFrame с добавленными прогнозами и вероятностями
        """
        if not self.is_trained:
            logger.warning(f"Модель {self.name} не обучена полностью, прогноз может быть неточным")

        # Если не переданы колонки с признаками, используем сохраненные
        if feature_columns is None:
            feature_columns = self.feature_names

        if feature_columns is None:
            # Если нет сохраненных признаков, используем все числовые колонки
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()

            # Исключаем целевые колонки и колонки с сигналами
            exclude_columns = ['buy_signal', 'sell_signal', 'hold_signal', 'signal',
                               'predicted_signal', 'predicted_buy', 'predicted_sell', 'predicted_hold']
            feature_columns = [col for col in feature_columns if col not in exclude_columns]

        # Проверяем, все ли признаки есть в DataFrame
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            logger.warning(f"Отсутствуют признаки: {missing_features}")
            # Используем только доступные признаки
            feature_columns = [col for col in feature_columns if col in df.columns]

        # Создаем копию DataFrame
        result_df = df.copy()

        # Добавляем колонки для прогнозов
        result_df['hybrid_signal'] = 0
        result_df['buy_probability'] = 0.0
        result_df['sell_probability'] = 0.0
        result_df['hold_probability'] = 0.0

        # Окно для прогнозирования
        sequence_length = self.model_params.get("sequence_length", 60)

        # Проходим по данным и делаем прогнозы
        for i in range(sequence_length, len(result_df)):
            # Получаем последовательность данных
            sequence = result_df.iloc[i - sequence_length:i][feature_columns].values

            # Делаем прогноз
            if return_probabilities:
                prediction, probabilities = self.predict(sequence, return_probabilities=True)

                # Добавляем прогноз и вероятности
                result_df.loc[result_df.index[i], 'hybrid_signal'] = prediction.item()
                result_df.loc[result_df.index[i], 'buy_probability'] = probabilities.get("buy_probability", 0.0)
                result_df.loc[result_df.index[i], 'sell_probability'] = probabilities.get("sell_probability", 0.0)
                result_df.loc[result_df.index[i], 'hold_probability'] = probabilities.get("hold_probability", 0.0)

                # Дополнительно сохраняем прогнозы от каждой подмодели
                if "lstm_prediction" in probabilities and probabilities["lstm_prediction"] is not None:
                    result_df.loc[result_df.index[i], 'lstm_prediction'] = probabilities["lstm_prediction"]

                if "rl_prediction" in probabilities and probabilities["rl_prediction"] is not None:
                    result_df.loc[result_df.index[i], 'rl_prediction'] = probabilities["rl_prediction"]

                if "technical_prediction" in probabilities and probabilities["technical_prediction"] is not None:
                    result_df.loc[result_df.index[i], 'technical_prediction'] = probabilities["technical_prediction"]
            else:
                prediction = self.predict(sequence, return_probabilities=False)

                # Добавляем прогноз
                result_df.loc[result_df.index[i], 'hybrid_signal'] = prediction.item()

        # Добавляем столбцы с дискретными сигналами
        result_df['buy_signal'] = (result_df['hybrid_signal'] == 1).astype(int)
        result_df['sell_signal'] = (result_df['hybrid_signal'] == 2).astype(int)
        result_df['hold_signal'] = (result_df['hybrid_signal'] == 0).astype(int)

        return result_df

    def backtest(
            self,
            df: pd.DataFrame,
            feature_columns: Optional[List[str]] = None,
            initial_balance: float = 10000.0,
            position_size: float = 1.0,
            fee: float = 0.001,
            stop_loss: Optional[float] = None,
            take_profit: Optional[float] = None
    ) -> Dict:
        """
        Выполняет бэктестинг гибридной модели на исторических данных.

        Args:
            df: DataFrame с историческими данными
            feature_columns: Список колонок с признаками
            initial_balance: Начальный баланс
            position_size: Размер позиции (доля от баланса)
            fee: Комиссия за сделку (например, 0.001 = 0.1%)
            stop_loss: Уровень стоп-лосса (например, 0.02 = 2%)
            take_profit: Уровень тейк-профита (например, 0.05 = 5%)

        Returns:
            Словарь с результатами бэктестинга
        """
        # Генерируем сигналы
        data_with_signals = self.predict_from_dataframe(df, feature_columns, return_probabilities=True)

        # Создаем копию для бэктестинга
        backtest_data = data_with_signals.copy()

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
            signal = backtest_data.iloc[i]['hybrid_signal']

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

        # Закрываем позицию в конце периода, если она открыта
        if current_position > 0:
            # Получаем последнюю цену
            last_price = backtest_data.iloc[-1]['close']

            # Закрываем позицию
            sale_value = current_position * last_price * (1 - fee)
            final_balance = backtest_data.iloc[-1]['balance'] + sale_value

            # Рассчитываем прибыль/убыток
            entry_value = current_position * entry_price
            profit_loss = sale_value - entry_value
            profit_loss_pct = (last_price - entry_price) / entry_price * 100

            # Записываем сделку
            trades.append({
                'type': 'end_of_period',
                'entry_price': entry_price,
                'exit_price': last_price,
                'position': current_position,
                'profit_loss': profit_loss,
                'profit_loss_pct': profit_loss_pct
            })

            # Обновляем последнюю строку
            backtest_data.loc[backtest_data.index[-1], 'balance'] = final_balance
            backtest_data.loc[backtest_data.index[-1], 'position'] = 0.0
            backtest_data.loc[backtest_data.index[-1], 'position_value'] = 0.0
            backtest_data.loc[backtest_data.index[-1], 'equity'] = final_balance

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

            # Годовая доходность (предполагаем 252 торговых дня в году)
            days = (backtest_data.index[-1] - backtest_data.index[0]).days
            annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0

            # Коэффициент Шарпа (если есть данные о доходности)
            returns = backtest_data['equity'].pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(
                returns) > 0 and returns.std() > 0 else 0

            # Создаем результаты бэктестинга
            backtest_results = {
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
                'trades': [],
                'equity_curve': backtest_data['equity'].values
            }

        # Логируем результаты
        logger.info(f"Результаты бэктестинга гибридной модели {self.name}:")
        logger.info(f"Начальный капитал: {initial_equity:.2f}")
        logger.info(f"Конечный капитал: {final_equity:.2f}")
        logger.info(f"Общая доходность: {backtest_results['total_return_pct']:.2f}%")
        logger.info(f"Годовая доходность: {backtest_results['annual_return_pct']:.2f}%")
        logger.info(f"Максимальная просадка: {backtest_results['max_drawdown_pct']:.2f}%")
        logger.info(f"Всего сделок: {backtest_results['total_trades']}")

        if backtest_results['total_trades'] > 0:
            logger.info(
                f"Прибыльных сделок: {backtest_results['profitable_trades']} ({backtest_results['win_rate_pct']:.2f}%)")
            logger.info(
                f"Убыточных сделок: {backtest_results['losing_trades']} ({100 - backtest_results['win_rate_pct']:.2f}%)")
            logger.info(f"Отношение прибыли к убытку: {backtest_results['profit_loss_ratio']:.2f}")
            logger.info(f"Фактор восстановления: {backtest_results['recovery_factor']:.2f}")
            logger.info(f"Коэффициент Шарпа: {backtest_results['sharpe_ratio']:.2f}")

        return backtest_results

    def optimize_weights(
            self,
            df: pd.DataFrame,
            feature_columns: Optional[List[str]] = None,
            alpha_range: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
            beta_range: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
            gamma_range: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
            threshold_range: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
            metric: str = 'sharpe_ratio'
    ) -> Dict:
        """
        Оптимизирует веса моделей в ансамбле для достижения лучших результатов.

        Args:
            df: DataFrame с данными
            feature_columns: Список колонок с признаками
            alpha_range: Диапазон весов для LSTM модели
            beta_range: Диапазон весов для RL модели
            gamma_range: Диапазон весов для модели технического анализа
            threshold_range: Диапазон порогов для принятия решения
            metric: Метрика для оптимизации

        Returns:
            Словарь с лучшими параметрами и результатами
        """
        if not self.is_trained:
            logger.warning(f"Модель {self.name} не обучена полностью, оптимизация весов может быть неточной")

        # Проверяем доступность подмоделей
        available_models = {}
        if self.lstm_model.is_trained:
            available_models["lstm"] = True
        if self.rl_model.is_trained:
            available_models["rl"] = True
        if self.technical_model.is_trained:
            available_models["technical"] = True

        if len(available_models) < 2:
            logger.warning("Недостаточно обученных моделей для оптимизации весов")
            return {}

        # Подготовка диапазонов весов
        if "lstm" not in available_models:
            alpha_range = [0.0]
        if "rl" not in available_models:
            beta_range = [0.0]
        if "technical" not in available_models:
            gamma_range = [0.0]

        # Проверяем валидность метрики
        valid_metrics = ['sharpe_ratio', 'total_return', 'annual_return', 'recovery_factor', 'win_rate']
        if metric not in valid_metrics:
            logger.warning(f"Неподдерживаемая метрика: {metric}, используем sharpe_ratio")
            metric = 'sharpe_ratio'

        # Логируем начало оптимизации
        logger.info(f"Начинаем оптимизацию весов для гибридной модели {self.name} по метрике {metric}")

        # Создаем сетку параметров
        best_params = None
        best_metric_value = float('-inf')
        best_backtest_results = None

        # Проходим по всем комбинациям параметров
        total_combinations = len(alpha_range) * len(beta_range) * len(gamma_range) * len(threshold_range)
        current_combination = 0

        for alpha in alpha_range:
            for beta in beta_range:
                for gamma in gamma_range:
                    # Пропускаем невалидные комбинации весов
                    if alpha + beta + gamma == 0:
                        continue

                    # Нормализуем веса
                    norm_alpha = alpha / (alpha + beta + gamma)
                    norm_beta = beta / (alpha + beta + gamma)
                    norm_gamma = gamma / (alpha + beta + gamma)

                    for threshold in threshold_range:
                        current_combination += 1
                        logger.info(f"Тестирование комбинации {current_combination}/{total_combinations}: "
                                    f"alpha={norm_alpha:.2f}, beta={norm_beta:.2f}, gamma={norm_gamma:.2f}, threshold={threshold:.2f}")

                        # Обновляем параметры модели
                        self.model_params["alpha"] = norm_alpha
                        self.model_params["beta"] = norm_beta
                        self.model_params["gamma"] = norm_gamma
                        self.model_params["decision_threshold"] = threshold

                        # Выполняем бэктестинг
                        backtest_results = self.backtest(df, feature_columns)

                        # Проверяем метрику
                        metric_value = backtest_results.get(metric, float('-inf'))

                        # Обновляем лучшие параметры, если нужно
                        if metric_value > best_metric_value:
                            best_metric_value = metric_value
                            best_params = {
                                "alpha": norm_alpha,
                                "beta": norm_beta,
                                "gamma": norm_gamma,
                                "decision_threshold": threshold
                            }
                            best_backtest_results = backtest_results

        # Обновляем параметры модели до лучших найденных
        if best_params:
            self.model_params.update(best_params)

            logger.info(f"Оптимизация весов завершена")
            logger.info(f"Лучшие параметры: alpha={best_params['alpha']:.2f}, "
                        f"beta={best_params['beta']:.2f}, gamma={best_params['gamma']:.2f}, "
                        f"threshold={best_params['decision_threshold']:.2f}")
            logger.info(f"Значение метрики {metric}: {best_metric_value:.4f}")

        return {
            "best_params": best_params,
            "best_metric_value": best_metric_value,
            "best_backtest_results": best_backtest_results
        }