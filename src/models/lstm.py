"""
Модуль для LSTM модели предсказания цен.
"""
import logging
import os
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class LSTMModel(BaseModel):
    """
    Модель на основе LSTM для прогнозирования цен.
    """

    def __init__(self, name: str, model_params: Dict = None):
        """
        Инициализирует LSTM модель.

        Args:
            name: Имя модели
            model_params: Параметры модели
                - lstm_units: Количество LSTM юнитов
                - dense_units: Количество юнитов в полносвязном слое
                - dropout_rate: Коэффициент отсева
                - learning_rate: Скорость обучения
                - batch_size: Размер батча
                - epochs: Количество эпох
                - sequence_length: Длина последовательности (окно)
                - optimizer: Оптимизатор ('adam', 'rmsprop', etc.)
                - loss: Функция потерь ('mse', 'mae', 'huber', etc.)
                - output_size: Размер выходного слоя (1 для регрессии, >1 для классификации)
                - stateful: Сохранять ли состояние между батчами
        """
        # Параметры по умолчанию
        default_params = {
            "lstm_units": 100,
            "dense_units": 64,
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "sequence_length": 60,
            "optimizer": "adam",
            "loss": "mse",
            "output_size": 1,
            "stateful": False
        }

        # Объединяем параметры по умолчанию с переданными
        if model_params:
            default_params.update(model_params)

        super().__init__(name, default_params)

        # Флаг для отслеживания, построена ли модель
        self.is_built = False

    def build(self) -> None:
        """
        Строит модель с заданными параметрами.
        """
        sequence_length = self.model_params.get("sequence_length", 60)
        features_count = len(
            self.feature_names) if self.feature_names is not None else 92  # Задаем значение по умолчанию

        input_shape = (sequence_length, features_count)

        model = Sequential()

        # Добавляем LSTM слои с учетом количества признаков
        model.add(LSTM(
            units=self.model_params.get("lstm_units", 100),
            return_sequences=True,
            input_shape=input_shape,
            stateful=self.model_params.get("stateful", False)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.model_params.get("dropout_rate", 0.2)))

        # Второй LSTM слой
        model.add(LSTM(
            units=self.model_params.get("lstm_units", 100) // 2,
            return_sequences=False,
            stateful=self.model_params.get("stateful", False)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.model_params.get("dropout_rate", 0.2)))

        # Полносвязный слой
        model.add(Dense(units=self.model_params.get("dense_units", 64), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(self.model_params.get("dropout_rate", 0.2)))

        # Выходной слой
        model.add(Dense(units=self.model_params.get("output_size", 1), activation='linear'))

        # Компилируем модель
        optimizer = Adam(learning_rate=self.model_params.get("learning_rate", 0.001))
        model.compile(
            optimizer=optimizer,
            loss=self.model_params.get("loss", "mse"),
            metrics=['mae', 'mse']
        )

        self.model = model
        self.is_built = True

        logger.info(f"Построена LSTM модель {self.name}")

    def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            callbacks: List = None,
            **kwargs
    ) -> Dict:
        """
        Обучает LSTM модель на заданных данных.

        Args:
            X_train: Обучающие данные (признаки)
            y_train: Обучающие целевые значения
            X_val: Валидационные данные (признаки, опционально)
            y_val: Валидационные целевые значения (опционально)
            callbacks: Список колбэков TensorFlow (опционально)
            **kwargs: Дополнительные аргументы для обучения

        Returns:
            Словарь с историей обучения и метриками
        """
        # Проверяем, построена ли модель
        if not self.is_built:
            self.build()

        # Если колбэки не заданы, создаем стандартные
        if callbacks is None:
            callbacks = self._get_default_callbacks()

        # Параметры обучения
        batch_size = kwargs.get("batch_size", self.model_params.get("batch_size", 32))
        epochs = kwargs.get("epochs", self.model_params.get("epochs", 100))

        # Проверяем размерность входных данных
        if len(X_train.shape) != 3:
            logger.warning(
                f"Неправильная размерность X_train: {X_train.shape}. Ожидается (samples, sequence_length, features)")
            logger.warning("Попытка преобразовать размерность...")

            if len(X_train.shape) == 2:
                # Если это 2D массив, предполагаем, что это (samples, features)
                # и преобразуем в (samples, 1, features)
                X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

                if X_val is not None:
                    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])

        # Проверяем размерность целевых значений
        if len(y_train.shape) > 2:
            logger.warning(
                f"Неправильная размерность y_train: {y_train.shape}. Ожидается (samples,) или (samples, output_size)")
            # Попытка преобразовать размерность
            y_train = y_train.reshape(y_train.shape[0], -1)

            if y_val is not None:
                y_val = y_val.reshape(y_val.shape[0], -1)

        # Масштабирование целевых значений
        # Нормализуем целевые значения для предотвращения слишком больших значений loss
        y_scaler = MinMaxScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

        if y_val is not None:
            y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
        else:
            y_val_scaled = None

        # Сохраняем scaler в метаданных для дальнейшего использования при прогнозировании
        self.metadata["y_scaler"] = y_scaler

        # Формируем данные для валидации
        validation_data = None
        if X_val is not None and y_val_scaled is not None:
            validation_data = (X_val, y_val_scaled)

        # Обучаем модель
        logger.info(f"Начинаем обучение модели {self.name}")

        history = self.model.fit(
            X_train, y_train_scaled,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )

        # Устанавливаем флаг обученной модели
        self.is_trained = True

        # Сохраняем историю обучения в метаданные
        self.metadata["train_history"] = {
            "loss": history.history["loss"],
            "val_loss": history.history.get("val_loss", []),
            "mae": history.history.get("mae", []),
            "val_mae": history.history.get("val_mae", [])
        }

        # Вычисляем и сохраняем метрики на обучающем наборе
        train_metrics = self.model.evaluate(X_train, y_train_scaled, verbose=0)
        if isinstance(train_metrics, list):
            train_metrics = {
                metric_name: value
                for metric_name, value in zip(self.model.metrics_names, train_metrics)
            }
        self.metadata["train_metrics"] = train_metrics

        # Вычисляем и сохраняем метрики на валидационном наборе, если он есть
        if validation_data:
            val_metrics = self.model.evaluate(X_val, y_val_scaled, verbose=0)
            if isinstance(val_metrics, list):
                val_metrics = {
                    metric_name: value
                    for metric_name, value in zip(self.model.metrics_names, val_metrics)
                }
            self.metadata["val_metrics"] = val_metrics

        logger.info(f"Обучение модели {self.name} завершено")

        return {
            "history": history.history,
            "train_metrics": train_metrics,
            "val_metrics": self.metadata.get("val_metrics", {})
        }

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Делает прогнозы на основе входных данных.

        Args:
            X: Входные данные для прогнозирования
            **kwargs: Дополнительные аргументы для прогнозирования

        Returns:
            Массив прогнозов
        """
        if not self.is_trained:
            logger.warning(f"Модель {self.name} не обучена, невозможно сделать прогноз")
            return np.array([])

        # Проверяем размерность входных данных
        if len(X.shape) != 3:
            logger.warning(f"Неправильная размерность X: {X.shape}. Ожидается (samples, sequence_length, features)")
            logger.warning("Попытка преобразовать размерность...")

            if len(X.shape) == 2:
                # Если это 2D массив, предполагаем, что это (samples, features)
                # и преобразуем в (samples, 1, features)
                X = X.reshape(X.shape[0], 1, X.shape[1])

        # Делаем прогноз (относительное изменение)
        scaled_predictions = self.model.predict(X, **kwargs)

        # Обратное масштабирование прогнозов, если у нас есть scaler
        if "y_scaler" in self.metadata:
            y_scaler = self.metadata["y_scaler"]
            pct_change_predictions = y_scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).flatten()
        else:
            pct_change_predictions = scaled_predictions

        # Если передан текущий курс, преобразуем относительное изменение в абсолютную цену
        current_price = kwargs.get('current_price', None)
        if current_price is not None:
            # Преобразуем процентное изменение обратно в абсолютную цену
            absolute_predictions = current_price * (1 + pct_change_predictions)
            return absolute_predictions

        # Иначе возвращаем процентное изменение
        return pct_change_predictions

    def predict_sequence(
            self,
            initial_sequence: np.ndarray,
            n_steps: int = 10,
            include_history: bool = False
    ) -> np.ndarray:
        """
        Предсказывает последовательность из n_steps шагов.

        Args:
            initial_sequence: Начальная последовательность для прогнозирования
            n_steps: Количество шагов для прогнозирования
            include_history: Включать ли начальную последовательность в результат

        Returns:
            Массив предсказанных значений
        """
        if not self.is_trained:
            logger.warning(f"Модель {self.name} не обучена, невозможно сделать прогноз")
            return np.array([])

        # Создаем копию начальной последовательности
        sequence = initial_sequence.copy()

        # Проверяем размерность входных данных
        if len(sequence.shape) != 3:
            logger.warning(
                f"Неправильная размерность sequence: {sequence.shape}. Ожидается (1, sequence_length, features)")
            logger.warning("Попытка преобразовать размерность...")

            if len(sequence.shape) == 2:
                # Если это 2D массив, предполагаем, что это (sequence_length, features)
                # и преобразуем в (1, sequence_length, features)
                sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])

            elif len(sequence.shape) == 1:
                # Если это 1D массив, предполагаем, что это (sequence_length,)
                # и преобразуем в (1, sequence_length, 1)
                sequence = sequence.reshape(1, sequence.shape[0], 1)

        # Получаем параметры последовательности
        seq_length = sequence.shape[1]
        n_features = sequence.shape[2]

        # Массив для хранения прогнозов
        predictions = []

        # Если нужно включить историю, добавляем последнее значение из начальной последовательности
        if include_history:
            # Берем последние значения из каждой временной точки в последовательности
            for i in range(seq_length):
                predictions.append(sequence[0, i, 0])  # предполагаем, что первый признак - это цена

        # Прогнозируем n_steps шагов
        current_sequence = sequence.copy()

        for _ in range(n_steps):
            # Делаем прогноз следующего шага
            next_step = self.model.predict(current_sequence, verbose=0)

            # Если прогноз многомерный, берем только первое значение (цену)
            if next_step.shape[1] > 1:
                next_step = next_step[:, 0:1]

            # Добавляем прогноз в список
            predictions.append(next_step[0, 0])

            # Обновляем последовательность, сдвигая окно на один шаг
            # и добавляя новый прогноз в конец
            if n_features == 1:
                # Если у нас только один признак (цена)
                new_point = np.array([[[next_step[0, 0]]]])
            else:
                # Если у нас несколько признаков, создаем точку с нулями для всех признаков,
                # кроме цены (которую мы предсказали)
                new_point = np.zeros((1, 1, n_features))
                new_point[0, 0, 0] = next_step[0, 0]  # предполагаем, что первый признак - это цена

            # Сдвигаем окно и добавляем новую точку
            current_sequence = np.concatenate([current_sequence[:, 1:, :], new_point], axis=1)

        return np.array(predictions)

    def _get_default_callbacks(self) -> List:
        """
        Создает стандартные колбэки для обучения.

        Returns:
            Список колбэков
        """
        callbacks = []

        # Early Stopping для предотвращения переобучения
        early_stopping = EarlyStopping(
            monitor='val_loss' if self.model_params.get("use_validation", True) else 'loss',
            patience=10,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)

        # Reduce Learning Rate on Plateau для адаптивной скорости обучения
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss' if self.model_params.get("use_validation", True) else 'loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
        callbacks.append(reduce_lr)

        # Model Checkpoint для сохранения лучшей модели
        if self.model_params.get("checkpoint_path"):
            os.makedirs(os.path.dirname(self.model_params["checkpoint_path"]), exist_ok=True)

            checkpoint = ModelCheckpoint(
                filepath=self.model_params["checkpoint_path"],
                monitor='val_loss' if self.model_params.get("use_validation", True) else 'loss',
                save_best_only=True,
                save_weights_only=False
            )
            callbacks.append(checkpoint)

        return callbacks

    def save_keras_model(self, path: str) -> None:
        """
        Сохраняет только Keras модель (не весь объект).

        Args:
            path: Путь для сохранения модели
        """
        if not self.is_trained:
            logger.warning(f"Модель {self.name} не обучена, нечего сохранять")
            return

        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Сохраняем Keras модель
        self.model.save(path)

        logger.info(f"Keras модель {self.name} сохранена в {path}")

    @classmethod
    def load_keras_model(cls, path: str, name: str = "loaded_model") -> 'LSTMModel':
        """
        Загружает Keras модель и создает экземпляр LSTMModel.

        Args:
            path: Путь к файлу модели
            name: Имя для новой модели

        Returns:
            Экземпляр LSTMModel с загруженной Keras моделью
        """
        # Загружаем Keras модель
        keras_model = load_model(path)

        # Создаем экземпляр LSTMModel
        model_instance = cls(name=name)
        model_instance.model = keras_model
        model_instance.is_built = True
        model_instance.is_trained = True

        logger.info(f"Keras модель загружена из {path} в модель {name}")

        return model_instance
