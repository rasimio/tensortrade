"""
Модуль для ансамблевой модели, комбинирующей несколько подходов.
"""
import logging
import os
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd

from src.models.base_model import BaseModel, ModelFactory

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """
    Ансамблевая модель, комбинирующая несколько моделей для улучшения результатов прогнозирования.
    Поддерживает различные стратегии объединения: голосование, взвешенное голосование, стекинг.
    """

    def __init__(self, name: str, model_params: Dict = None):
        """
        Инициализирует ансамблевую модель.

        Args:
            name: Имя модели
            model_params: Параметры модели
                - ensemble_type: Тип ансамбля ('voting', 'weighted', 'stacking')
                - model_configs: Конфигурации для подмоделей
                - weights: Веса для подмоделей (для взвешенного голосования)
                - meta_model_config: Конфигурация мета-модели (для стекинга)
        """
        # Параметры по умолчанию
        default_params = {
            "ensemble_type": "voting",
            "model_configs": [],
            "weights": None,
            "meta_model_config": None
        }

        # Объединяем параметры по умолчанию с переданными
        if model_params:
            default_params.update(model_params)

        super().__init__(name, default_params)

        self.models = []
        self.meta_model = None

    def build(self) -> None:
        """
        Строит ансамблевую модель с заданными параметрами.
        Создает и инициализирует все подмодели.
        """
        ensemble_type = self.model_params.get("ensemble_type", "voting")
        model_configs = self.model_params.get("model_configs", [])

        if not model_configs:
            logger.warning(f"Не заданы конфигурации для подмоделей, невозможно построить ансамбль")
            return

        # Создаем подмодели
        self.models = []
        for i, config in enumerate(model_configs):
            if 'model_type' not in config:
                logger.warning(f"Не указан тип модели для конфигурации {i}, пропуск")
                continue

            if 'name' not in config:
                config['name'] = f"{self.name}_sub{i}"

            model = ModelFactory.create_model(
                model_type=config['model_type'],
                name=config['name'],
                model_params=config.get('model_params')
            )

            self.models.append(model)

        # Создаем мета-модель для стекинга
        if ensemble_type == "stacking" and self.model_params.get("meta_model_config"):
            meta_config = self.model_params.get("meta_model_config")

            if 'model_type' not in meta_config:
                logger.warning(f"Не указан тип мета-модели, использование голосования по умолчанию")
            else:
                self.meta_model = ModelFactory.create_model(
                    model_type=meta_config['model_type'],
                    name=f"{self.name}_meta",
                    model_params=meta_config.get('model_params')
                )

        logger.info(f"Построен ансамбль {self.name} типа {ensemble_type} с {len(self.models)} подмоделями")

    def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            **kwargs
    ) -> Dict:
        """
        Обучает ансамблевую модель на заданных данных.

        Args:
            X_train: Обучающие данные (признаки)
            y_train: Обучающие целевые значения
            X_val: Валидационные данные (признаки, опционально)
            y_val: Валидационные целевые значения (опционально)
            **kwargs: Дополнительные аргументы для обучения

        Returns:
            Словарь с историей обучения и метриками
        """
        if not self.models:
            self.build()

        if not self.models:
            logger.warning(f"Нет подмоделей для обучения")
            return {}

        # Обучаем все подмодели
        logger.info(f"Начинаем обучение подмоделей для ансамбля {self.name}")

        models_metrics = {}

        for model in self.models:
            logger.info(f"Обучение подмодели {model.name}")

            # Обучаем модель
            metrics = model.train(X_train, y_train, X_val, y_val, **kwargs)

            # Сохраняем метрики
            models_metrics[model.name] = metrics

        # Если используется стекинг, обучаем мета-модель
        ensemble_type = self.model_params.get("ensemble_type", "voting")

        if ensemble_type == "stacking" and self.meta_model:
            logger.info(f"Обучение мета-модели для стекинга")

            # Получаем прогнозы подмоделей на обучающих данных
            meta_X_train = self._get_models_predictions(X_train)

            # Получаем прогнозы подмоделей на валидационных данных
            meta_X_val = None
            if X_val is not None:
                meta_X_val = self._get_models_predictions(X_val)

            # Обучаем мета-модель
            meta_metrics = self.meta_model.train(meta_X_train, y_train, meta_X_val, y_val, **kwargs)

            # Сохраняем метрики
            models_metrics["meta_model"] = meta_metrics

        # Вычисляем метрики для ансамбля
        y_pred = self.predict(X_train)

        # Используем метрики классификации или регрессии в зависимости от задачи
        if len(np.unique(y_train)) <= 10:  # Предполагаем, что это задача классификации
            from sklearn.metrics import accuracy_score, f1_score

            train_metrics = {
                "accuracy": accuracy_score(y_train, y_pred),
                "f1": f1_score(y_train, y_pred, average='weighted')
            }
        else:  # Предполагаем, что это задача регрессии
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            train_metrics = {
                "mse": mean_squared_error(y_train, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_train, y_pred)),
                "mae": mean_absolute_error(y_train, y_pred),
                "r2": r2_score(y_train, y_pred)
            }

        # Вычисляем метрики для валидационных данных
        val_metrics = None
        if X_val is not None and y_val is not None:
            y_val_pred = self.predict(X_val)

            if len(np.unique(y_val)) <= 10:  # Задача классификации
                val_metrics = {
                    "accuracy": accuracy_score(y_val, y_val_pred),
                    "f1": f1_score(y_val, y_val_pred, average='weighted')
                }
            else:  # Задача регрессии
                val_metrics = {
                    "mse": mean_squared_error(y_val, y_val_pred),
                    "rmse": np.sqrt(mean_squared_error(y_val, y_val_pred)),
                    "mae": mean_absolute_error(y_val, y_val_pred),
                    "r2": r2_score(y_val, y_val_pred)
                }

        # Устанавливаем флаг обученной модели
        self.is_trained = True

        # Сохраняем результаты в метаданные
        self.metadata["models_metrics"] = models_metrics
        self.metadata["train_metrics"] = train_metrics
        if val_metrics:
            self.metadata["val_metrics"] = val_metrics

        # Логируем результаты
        logger.info(f"Обучение ансамбля {self.name} завершено")

        for metric_name, metric_value in train_metrics.items():
            logger.info(f"Метрика {metric_name} на обучающем наборе: {metric_value:.4f}")

        if val_metrics:
            for metric_name, metric_value in val_metrics.items():
                logger.info(f"Метрика {metric_name} на валидационном наборе: {metric_value:.4f}")

        return {
            "models_metrics": models_metrics,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics
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

        ensemble_type = self.model_params.get("ensemble_type", "voting")

        # Получаем прогнозы всех подмоделей
        models_predictions = []

        for model in self.models:
            # Проверяем, обучена ли подмодель
            if not model.is_trained:
                logger.warning(f"Подмодель {model.name} не обучена, пропуск")
                continue

            # Получаем прогноз
            pred = model.predict(X, **kwargs)
            models_predictions.append(pred)

        if not models_predictions:
            logger.warning(f"Ни одна подмодель не сделала прогноз")
            return np.array([])

        # Объединяем прогнозы в зависимости от типа ансамбля
        if ensemble_type == "voting":
            # Простое голосование (для классификации)
            if len(models_predictions[0].shape) == 1 or models_predictions[0].shape[1] == 1:
                # Голосование для регрессии - среднее значение
                predictions = np.mean(models_predictions, axis=0)
            else:
                # Голосование для классификации - наиболее частый класс
                stacked_predictions = np.stack(models_predictions)
                predictions = np.apply_along_axis(
                    lambda x: np.argmax(np.bincount(x)),
                    axis=0,
                    arr=stacked_predictions
                )

        elif ensemble_type == "weighted":
            # Взвешенное голосование
            weights = self.model_params.get("weights")

            if weights is None or len(weights) != len(models_predictions):
                # Если веса не заданы или их количество не соответствует количеству моделей,
                # используем равные веса
                weights = [1.0 / len(models_predictions)] * len(models_predictions)

            # Нормализуем веса, чтобы они давали в сумме 1
            weights = np.array(weights) / sum(weights)

            # Взвешенное среднее
            predictions = np.average(models_predictions, axis=0, weights=weights)

        elif ensemble_type == "stacking" and self.meta_model and self.meta_model.is_trained:
            # Стекинг - используем мета-модель для объединения прогнозов
            meta_X = self._get_models_predictions(X)
            predictions = self.meta_model.predict(meta_X, **kwargs)

        else:
            # По умолчанию - простое среднее
            predictions = np.mean(models_predictions, axis=0)

        return predictions

    def _get_models_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Получает прогнозы всех подмоделей и объединяет их в один массив.

        Args:
            X: Входные данные для прогнозирования

        Returns:
            Массив прогнозов подмоделей (для стекинга)
        """
        # Получаем прогнозы всех подмоделей
        models_predictions = []

        for model in self.models:
            # Проверяем, обучена ли подмодель
            if not model.is_trained:
                logger.warning(f"Подмодель {model.name} не обучена, пропуск")
                continue

            # Получаем прогноз
            pred = model.predict(X)

            # Приводим к 2D виду, если нужно
            if len(pred.shape) == 1:
                pred = pred.reshape(-1, 1)

            models_predictions.append(pred)

        if not models_predictions:
            logger.warning(f"Ни одна подмодель не сделала прогноз")
            return np.array([])

        # Объединяем прогнозы в один массив
        meta_X = np.hstack(models_predictions)

        return meta_X

    def predict_proba(self, X: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        """
        Делает прогнозы вероятностей классов (для задач классификации).

        Args:
            X: Входные данные для прогнозирования
            **kwargs: Дополнительные аргументы для прогнозирования

        Returns:
            Массив вероятностей классов или None, если не поддерживается
        """
        if not self.is_trained:
            logger.warning(f"Модель {self.name} не обучена, невозможно сделать прогноз")
            return None

        ensemble_type = self.model_params.get("ensemble_type", "voting")

        # Получаем вероятности от всех подмоделей, которые поддерживают predict_proba
        models_probas = []

        for model in self.models:
            # Проверяем, обучена ли подмодель
            if not model.is_trained:
                continue

            # Проверяем, поддерживает ли модель predict_proba
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X, **kwargs)
                if proba is not None:
                    models_probas.append(proba)

        if not models_probas:
            logger.warning(f"Ни одна подмодель не поддерживает прогнозирование вероятностей")
            return None

        # Объединяем вероятности в зависимости от типа ансамбля
        if ensemble_type == "voting":
            # Простое среднее вероятностей
            probas = np.mean(models_probas, axis=0)

        elif ensemble_type == "weighted":
            # Взвешенное среднее вероятностей
            weights = self.model_params.get("weights")

            if weights is None or len(weights) != len(models_probas):
                # Если веса не заданы или их количество не соответствует количеству моделей,
                # используем равные веса
                weights = [1.0 / len(models_probas)] * len(models_probas)

            # Нормализуем веса, чтобы они давали в сумме 1
            weights = np.array(weights) / sum(weights)

            # Взвешенное среднее
            probas = np.average(models_probas, axis=0, weights=weights)

        elif ensemble_type == "stacking" and self.meta_model:
            # Проверяем, поддерживает ли мета-модель predict_proba
            if hasattr(self.meta_model, 'predict_proba'):
                meta_X = self._get_models_predictions(X)
                probas = self.meta_model.predict_proba(meta_X, **kwargs)
            else:
                # Если мета-модель не поддерживает predict_proba, используем простое среднее
                probas = np.mean(models_probas, axis=0)

        else:
            # По умолчанию - простое среднее
            probas = np.mean(models_probas, axis=0)

        return probas

    def add_model(self, model: BaseModel, weight: float = 1.0) -> None:
        """
        Добавляет новую модель в ансамбль.

        Args:
            model: Модель для добавления
            weight: Вес модели (для взвешенного голосования)
        """
        if model.is_trained:
            self.models.append(model)

            # Добавляем вес, если это взвешенное голосование
            if self.model_params.get("ensemble_type") == "weighted":
                weights = self.model_params.get("weights", [])
                weights.append(weight)
                self.model_params["weights"] = weights

            logger.info(f"Добавлена модель {model.name} в ансамбль {self.name}")
        else:
            logger.warning(f"Модель {model.name} не обучена, невозможно добавить в ансамбль")

    def remove_model(self, model_name: str) -> bool:
        """
        Удаляет модель из ансамбля по имени.

        Args:
            model_name: Имя модели для удаления

        Returns:
            True, если модель удалена, False в противном случае
        """
        for i, model in enumerate(self.models):
            if model.name == model_name:
                self.models.pop(i)

                # Удаляем вес, если это взвешенное голосование
                if self.model_params.get("ensemble_type") == "weighted":
                    weights = self.model_params.get("weights", [])
                    if i < len(weights):
                        weights.pop(i)
                        self.model_params["weights"] = weights

                logger.info(f"Удалена модель {model_name} из ансамбля {self.name}")
                return True

        logger.warning(f"Модель {model_name} не найдена в ансамбле {self.name}")
        return False

    def set_model_weight(self, model_name: str, weight: float) -> bool:
        """
        Устанавливает вес для модели в ансамбле.

        Args:
            model_name: Имя модели
            weight: Новый вес

        Returns:
            True, если вес установлен, False в противном случае
        """
        if self.model_params.get("ensemble_type") != "weighted":
            logger.warning(f"Установка весов поддерживается только для взвешенного голосования")
            return False

        for i, model in enumerate(self.models):
            if model.name == model_name:
                weights = self.model_params.get("weights", [])

                # Если веса не инициализированы, создаем их
                if not weights or len(weights) != len(self.models):
                    weights = [1.0] * len(self.models)

                # Устанавливаем новый вес
                weights[i] = weight
                self.model_params["weights"] = weights

                logger.info(f"Установлен вес {weight} для модели {model_name} в ансамбле {self.name}")
                return True

        logger.warning(f"Модель {model_name} не найдена в ансамбле {self.name}")
        return False

    def get_models_metrics(self) -> Dict:
        """
        Возвращает метрики для всех моделей в ансамбле.

        Returns:
            Словарь с метриками моделей
        """
        return self.metadata.get("models_metrics", {})

    def get_model_feature_importance(self, model_name: str) -> Optional[Dict]:
        """
        Возвращает важность признаков для указанной модели в ансамбле.

        Args:
            model_name: Имя модели

        Returns:
            Словарь с важностью признаков или None, если не поддерживается
        """
        for model in self.models:
            if model.name == model_name:
                if hasattr(model, 'get_metadata') and model.get_metadata('feature_importances'):
                    return model.get_metadata('feature_importances')

                if hasattr(model.model, 'feature_importances_'):
                    feature_names = model.feature_names or [f"feature_{i}" for i in
                                                            range(len(model.model.feature_importances_))]
                    return dict(zip(feature_names, model.model.feature_importances_))

                return None

        logger.warning(f"Модель {model_name} не найдена в ансамбле {self.name}")
        return None