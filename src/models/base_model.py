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
            "name": self.name,
            "model_params": self.model_params,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "is_trained": self.is_trained,
            "metadata": self.metadata,
            "model_type": self.__class__.__name__
        }

        # Для гибридной модели сохраняем состояние подмоделей
        if self.__class__.__name__ == 'HybridModel':
            hybrid_components = {}

            # Сохраняем состояние технической модели, если она существует и обучена
            if hasattr(self,
                       'technical_model') and self.technical_model is not None and self.technical_model.is_trained:
                try:
                    # Сохраняем техническую модель отдельно
                    technical_model_path = os.path.splitext(path)[0] + "_technical.pkl"
                    self.technical_model.save(technical_model_path)
                    hybrid_components['technical_model_path'] = technical_model_path
                except Exception as e:
                    logger.warning(f"Не удалось сохранить техническую модель: {e}")

            # Сохраняем состояние LSTM модели, если она существует и обучена
            if hasattr(self, 'lstm_model') and self.lstm_model is not None and self.lstm_model.is_trained:
                try:
                    # Сохраняем LSTM модель отдельно
                    lstm_model_path = os.path.splitext(path)[0] + "_lstm.pkl"
                    self.lstm_model.save(lstm_model_path)
                    hybrid_components['lstm_model_path'] = lstm_model_path
                except Exception as e:
                    logger.warning(f"Не удалось сохранить LSTM модель: {e}")

            # Для RL модели, которая часто не может быть сериализована с помощью pickle,
            # сохраняем только параметры и флаг обучения
            if hasattr(self, 'rl_model') and self.rl_model is not None and self.rl_model.is_trained:
                try:
                    # Сохраняем RL модель через собственный метод, если доступен
                    if hasattr(self.rl_model, 'save_stable_baselines_model'):
                        rl_model_path = os.path.splitext(path)[0] + "_rl"
                        self.rl_model.save_stable_baselines_model(rl_model_path)
                        hybrid_components['rl_model_path'] = rl_model_path
                    else:
                        # Сохраняем только параметры и флаг обучения
                        hybrid_components['rl_model_params'] = self.rl_model.model_params
                        hybrid_components['rl_model_trained'] = True
                except Exception as e:
                    logger.warning(f"Не удалось сохранить RL модель: {e}")
                    hybrid_components['rl_model_params'] = self.rl_model.model_params
                    hybrid_components['rl_model_trained'] = self.rl_model.is_trained

            save_dict["hybrid_components"] = hybrid_components
        else:
            # Пытаемся сохранить модель, если это не гибридная модель
            try:
                save_dict["model"] = self.model
            except Exception as e:
                logger.warning(f"Не удалось сериализовать модель {self.name}: {e}")
                save_dict["model"] = None

        # Сохраняем модель
        try:
            with open(path, 'wb') as f:
                pickle.dump(save_dict, f)
            logger.info(f"Модель {self.name} сохранена в {path}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении модели {self.name}: {e}")

    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """
        Загружает модель из файла.

        Args:
            path: Путь к файлу модели

        Returns:
            Экземпляр загруженной модели
        """
        try:
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

            # Проверяем, есть ли компоненты гибридной модели
            hybrid_components = save_dict.get("hybrid_components", None)

            # Если это гибридная модель и есть компоненты
            if cls.__name__ == 'HybridModel' and hybrid_components:
                # Сначала строим пустые подмодели
                model_instance.build()

                # Импортируем зависимости здесь, чтобы избежать циклических импортов
                from src.models.technical import TechnicalModel
                from src.models.lstm import LSTMModel
                from src.models.reinforcement import RLModel

                # Восстанавливаем техническую модель, если доступна
                technical_model_path = hybrid_components.get('technical_model_path')
                if technical_model_path and os.path.exists(technical_model_path):
                    model_instance.technical_model = TechnicalModel.load(technical_model_path)

                # Восстанавливаем LSTM модель, если доступна
                lstm_model_path = hybrid_components.get('lstm_model_path')
                if lstm_model_path and os.path.exists(lstm_model_path):
                    model_instance.lstm_model = LSTMModel.load(lstm_model_path)

                # Восстанавливаем RL модель, если доступна
                rl_model_path = hybrid_components.get('rl_model_path')
                if rl_model_path:
                    # Проверяем наличие файла модели
                    if os.path.exists(rl_model_path + ".zip"):
                        try:
                            # Создаем экземпляр RL модели
                            rl_model = RLModel(name=model_instance.name + "_rl",
                                               model_params=model_instance.model_params.get("rl_params", {}))
                            # Устанавливаем флаг обученной модели
                            rl_model.is_trained = True
                            model_instance.rl_model = rl_model
                        except Exception as e:
                            logger.warning(f"Ошибка при загрузке RL модели: {e}")

                # Устанавливаем флаг обученной модели для гибридной модели, если хотя бы одна из подмоделей обучена
                if (hasattr(model_instance,
                            'technical_model') and model_instance.technical_model and model_instance.technical_model.is_trained) or \
                        (hasattr(model_instance,
                                 'lstm_model') and model_instance.lstm_model and model_instance.lstm_model.is_trained) or \
                        (hasattr(model_instance,
                                 'rl_model') and model_instance.rl_model and model_instance.rl_model.is_trained):
                    model_instance.is_trained = True

            logger.info(f"Модель {model_instance.name} загружена из {path}")

            return model_instance

        except Exception as e:
            logger.error(f"Ошибка при загрузке модели из {path}: {e}")
            # Возвращаем пустую модель
            return cls(name="error_loading_model")

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
