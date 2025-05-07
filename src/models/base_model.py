"""
Базовый класс для всех моделей в проекте.
"""
import os
import logging
import pickle
from typing import Dict, List, Optional, Union, Tuple, Any
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Абстрактный базовый класс для всех моделей.
    Определяет общий интерфейс для всех моделей в проекте.
    """

    def __init__(self, name: str, model_params: Dict = None):
        """
        Инициализирует базовую модель.

        Args:
            name: Имя модели
            model_params: Параметры модели
        """
        self.name = name
        self.model_params = model_params or {}
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.target_name = None
        self.metadata = {}

        logger.info(f"Initialized {self.__class__.__name__} with name '{name}'")

    @abstractmethod
    def build(self) -> None:
        """
        Строит модель с заданными параметрами.
        Должен быть реализован в подклассах.
        """
        pass

    @abstractmethod
    def train(
            self,
            X_train: np.ndarray,
            y_train: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            **kwargs
    ) -> Dict:
        """
        Обучает модель на заданных данных.
        Должен быть реализован в подклассах.

        Args:
            X_train: Обучающие данные (признаки)
            y_train: Обучающие целевые значения (опционально)
            X_val: Валидационные данные (признаки, опционально)
            y_val: Валидационные целевые значения (опционально)
            **kwargs: Дополнительные аргументы для обучения

        Returns:
            Словарь с историей обучения и метриками
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Делает прогнозы на основе входных данных.
        Должен быть реализован в подклассах.

        Args:
            X: Входные данные для прогнозирования
            **kwargs: Дополнительные аргументы для прогнозирования

        Returns:
            Массив прогнозов
        """
        pass

    def save(self, path: str) -> None:
        """
        Сохраняет модель в файл.

        Args:
            path: Путь для сохранения модели
        """
        if not self.is_trained:
            logger.warning(f"Модель {self.name} не обучена, нечего сохранять")
            return

        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Словарь для сохранения
        save_dict = {
            "model": self.model,
            "name": self.name,
            "model_params": self.model_params,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "is_trained": self.is_trained,
            "metadata": self.metadata,
            "model_type": self.__class__.__name__
        }

        # Сохраняем модель
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

        logger.info(f"Модель {self.name} сохранена в {path}")

    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """
        Загружает модель из файла.

        Args:
            path: Путь к файлу модели

        Returns:
            Экземпляр загруженной модели
        """
        # Загружаем модель из файла
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        # Создаем экземпляр модели
        model_instance = cls(
            name=save_dict.get("name", "loaded_model"),
            model_params=save_dict.get("model_params", {})
        )

        # Восстанавливаем атрибуты
        model_instance.model = save_dict.get("model")
        model_instance.feature_names = save_dict.get("feature_names")
        model_instance.target_name = save_dict.get("target_name")
        model_instance.is_trained = save_dict.get("is_trained", False)
        model_instance.metadata = save_dict.get("metadata", {})

        logger.info(f"Модель {model_instance.name} загружена из {path}")

        return model_instance

    def set_feature_names(self, feature_names: List[str]) -> None:
        """
        Устанавливает имена признаков.

        Args:
            feature_names: Список имен признаков
        """
        self.feature_names = feature_names
        logger.info(f"Установлены имена признаков для модели {self.name}")

    def set_target_name(self, target_name: str) -> None:
        """
        Устанавливает имя целевой переменной.

        Args:
            target_name: Имя целевой переменной
        """
        self.target_name = target_name
        logger.info(f"Установлено имя целевой переменной '{target_name}' для модели {self.name}")

    def add_metadata(self, key: str, value: Any) -> None:
        """
        Добавляет метаданные к модели.

        Args:
            key: Ключ метаданных
            value: Значение метаданных
        """
        self.metadata[key] = value
        logger.info(f"Добавлены метаданные '{key}' для модели {self.name}")

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Получает метаданные модели по ключу.

        Args:
            key: Ключ метаданных
            default: Значение по умолчанию, если ключ не найден

        Returns:
            Значение метаданных или значение по умолчанию
        """
        return self.metadata.get(key, default)

    def evaluate(
            self,
            X: np.ndarray,
            y: np.ndarray,
            metrics: List[str] = None
    ) -> Dict[str, float]:
        """
        Оценивает модель на заданных данных.

        Args:
            X: Тестовые данные (признаки)
            y: Истинные целевые значения
            metrics: Список метрик для вычисления

        Returns:
            Словарь с результатами оценки
        """
        from sklearn import metrics as sk_metrics

        if not self.is_trained:
            logger.warning(f"Модель {self.name} не обучена, невозможно оценить")
            return {}

        # Метрики по умолчанию
        if metrics is None:
            metrics = ['mse', 'rmse', 'mae', 'r2']

        # Делаем прогнозы
        y_pred = self.predict(X)

        # Вычисляем метрики
        evaluation = {}
        for metric in metrics:
            if metric.lower() == 'mse':
                evaluation['mse'] = sk_metrics.mean_squared_error(y, y_pred)
            elif metric.lower() == 'rmse':
                evaluation['rmse'] = np.sqrt(sk_metrics.mean_squared_error(y, y_pred))
            elif metric.lower() == 'mae':
                evaluation['mae'] = sk_metrics.mean_absolute_error(y, y_pred)
            elif metric.lower() == 'r2':
                evaluation['r2'] = sk_metrics.r2_score(y, y_pred)
            elif metric.lower() == 'mape':
                evaluation['mape'] = sk_metrics.mean_absolute_percentage_error(y, y_pred)
            elif metric.lower() == 'accuracy':
                evaluation['accuracy'] = sk_metrics.accuracy_score(y, np.round(y_pred))
            elif metric.lower() == 'f1':
                evaluation['f1'] = sk_metrics.f1_score(y, np.round(y_pred), average='weighted')
            elif metric.lower() == 'precision':
                evaluation['precision'] = sk_metrics.precision_score(y, np.round(y_pred), average='weighted')
            elif metric.lower() == 'recall':
                evaluation['recall'] = sk_metrics.recall_score(y, np.round(y_pred), average='weighted')

        logger.info(f"Метрики модели {self.name}: {evaluation}")

        return evaluation

    def summary(self) -> Dict:
        """
        Возвращает сводку информации о модели.

        Returns:
            Словарь с информацией о модели
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "is_trained": self.is_trained,
            "parameters": self.model_params,
            "feature_count": len(self.feature_names) if self.feature_names else None,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "metadata": self.metadata
        }


class ModelFactory:
    """
    Фабрика для создания моделей разных типов.
    """

    @staticmethod
    def create_model(model_type: str, name: str, model_params: Dict = None) -> BaseModel:
        """
        Создает экземпляр модели указанного типа.

        Args:
            model_type: Тип модели ('lstm', 'reinforcement', 'technical', 'ensemble', 'hybrid')
            name: Имя модели
            model_params: Параметры модели

        Returns:
            Экземпляр модели
        """
        # Импортируем модели по требованию для предотвращения циклических импортов
        if model_type.lower() == 'lstm':
            from src.models.lstm import LSTMModel
            return LSTMModel(name=name, model_params=model_params)
        elif model_type.lower() == 'reinforcement':
            from src.models.reinforcement import RLModel
            return RLModel(name=name, model_params=model_params)
        elif model_type.lower() == 'technical':
            from src.models.technical import TechnicalModel
            return TechnicalModel(name=name, model_params=model_params)
        elif model_type.lower() == 'ensemble':
            from src.models.ensemble import EnsembleModel
            return EnsembleModel(name=name, model_params=model_params)
        elif model_type.lower() == 'hybrid':
            from src.models.hybrid import HybridModel
            return HybridModel(name=name, model_params=model_params)
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")