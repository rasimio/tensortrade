"""
Модуль для RL модели для торговли.
"""
import logging
import os
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """
    Среда для обучения RL модели торговле на финансовых рынках.
    """

    def __init__(
            self,
            data: pd.DataFrame,
            feature_columns: List[str],
            window_size: int = 60,
            fee: float = 0.001,
            initial_balance: float = 10000.0,
            max_position: int = 1,
            reward_scaling: float = 1.0,
            use_position_features: bool = True,
            reward_function: Optional[Callable] = None,
            terminate_on_negative_balance: bool = True,
            early_stopping_threshold: Optional[float] = None
    ):
        """
        Инициализирует среду для торговли.

        Args:
            data: DataFrame с историческими данными
            feature_columns: Список колонок с признаками
            window_size: Размер окна наблюдения
            fee: Комиссия за сделку (например, 0.001 = 0.1%)
            initial_balance: Начальный баланс
            max_position: Максимальный размер позиции
            reward_scaling: Масштабирование вознаграждения
            use_position_features: Добавлять ли текущую позицию и баланс как признаки
            reward_function: Пользовательская функция вознаграждения
            terminate_on_negative_balance: Завершать ли эпизод при отрицательном балансе
            early_stopping_threshold: Порог для раннего завершения (если достигнута определенная доходность)
        """
        super(TradingEnvironment, self).__init__()

        self.data = data
        self.feature_columns = feature_columns
        self.window_size = window_size
        self.fee = fee
        self.initial_balance = initial_balance
        self.max_position = max_position
        self.reward_scaling = reward_scaling
        self.use_position_features = use_position_features
        self.reward_function = reward_function
        self.terminate_on_negative_balance = terminate_on_negative_balance
        self.early_stopping_threshold = early_stopping_threshold

        # Определение пространства действий
        self.action_space = spaces.Discrete(3)  # 0: держать, 1: покупать, 2: продавать

        # Определение пространства наблюдений
        n_features = len(feature_columns)
        if self.use_position_features:
            n_features += 2  # добавляем текущую позицию и баланс

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, n_features),
            dtype=np.float32
        )

        # Индекс текущего шага
        self.current_step = 0

        # Текущее состояние среды
        self.balance = initial_balance
        self.position = 0  # текущая позиция
        self.trades = []  # история сделок

        # Цена, по которой была открыта текущая позиция
        self.entry_price = 0.0

        # Общий P&L
        self.total_pnl = 0.0

        # Максимальная просадка
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance

        logger.info(f"Инициализирована торговая среда с {len(data)} точками данных и {n_features} признаками")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Union[
        np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Сбрасывает среду в начальное состояние.

        Args:
            seed: Случайное зерно
            options: Дополнительные опции

        Returns:
            Наблюдение или кортеж с наблюдением и дополнительной информацией
        """
        # Проверяем версию API
        try:
            # Для новых версий Gymnasium
            super().reset(seed=seed)
        except:
            # Для старых версий Gym, которые не требуют параметров
            if hasattr(super(), 'reset'):
                super().reset()

        # Инициализируем индекс текущего шага
        self.current_step = self.window_size

        # Сбрасываем состояние среды
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.trades = []
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = self.initial_balance

        # Получаем текущее наблюдение
        observation = self._get_observation()

        # Информация о текущем состоянии
        info = {
            "balance": self.balance,
            "position": self.position,
            "current_price": self._get_current_price(),
            "step": self.current_step,
            "total_pnl": self.total_pnl,
            "max_drawdown": self.max_drawdown
        }

        # Возвращаем результат в зависимости от версии Gym
        # Если это новая версия Gymnasium, возвращаем кортеж (observation, info)
        # Если старая версия Gym, возвращаем только observation
        # Проверим, какая версия используется, по наличию параметра seed
        if seed is not None or options is not None:
            # Новая версия Gymnasium
            return observation, info
        else:
            # Старая версия Gym
            return observation

    def step(self, action: int) -> Union[
        Tuple[np.ndarray, float, bool, Dict], Tuple[np.ndarray, float, bool, bool, Dict]]:
        """
        Выполняет шаг в среде, применяя действие.

        Args:
            action: Действие (0: держать, 1: покупать, 2: продавать)

        Returns:
            Tuple с наблюдением, вознаграждением, флагом завершения, доп. информацией (или также с флагом truncated в новых версиях)
        """
        # Проверяем, не закончились ли данные
        if self.current_step >= len(self.data) - 1:
            observation = self._get_observation()
            reward = 0.0
            done = True
            info = {
                "balance": self.balance,
                "position": self.position,
                "current_price": self._get_current_price(),
                "step": self.current_step,
                "total_pnl": self.total_pnl,
                "max_drawdown": self.max_drawdown,
                "reason": "Данные закончились"
            }

            # Проверим, какая версия gymnasium используется
            # Если определить не удается, предполагаем новую версию с truncated
            try:
                # Проверяем наличие метода render() с параметром mode
                if hasattr(self, 'render') and 'mode' in self.render.__code__.co_varnames:
                    # Старая версия Gym
                    return observation, reward, done, info
                else:
                    # Новая версия Gymnasium
                    return observation, reward, done, False, info
            except:
                # Предполагаем новую версию Gymnasium
                return observation, reward, done, False, info

        # Текущая цена
        current_price = self._get_current_price()

        # Предыдущий баланс для расчета вознаграждения
        prev_balance = self.balance
        prev_position = self.position

        # Применяем действие
        if action == 1:  # Покупаем
            if self.position < self.max_position:
                # Рассчитываем размер позиции для покупки
                position_to_buy = self.max_position - self.position

                # Рассчитываем стоимость покупки с учетом комиссии
                cost = position_to_buy * current_price * (1 + self.fee)

                # Проверяем, достаточно ли баланса
                if cost <= self.balance:
                    # Обновляем баланс
                    self.balance -= cost

                    # Обновляем позицию
                    self.position += position_to_buy

                    # Если это первая покупка, запоминаем цену входа
                    if prev_position == 0:
                        self.entry_price = current_price

                    # Записываем сделку
                    self.trades.append({
                        "step": self.current_step,
                        "type": "buy",
                        "price": current_price,
                        "quantity": position_to_buy,
                        "cost": cost,
                        "balance": self.balance
                    })

        elif action == 2:  # Продаем
            if self.position > 0:
                # Рассчитываем выручку от продажи с учетом комиссии
                revenue = self.position * current_price * (1 - self.fee)

                # Обновляем баланс
                self.balance += revenue

                # Рассчитываем P&L
                pnl = ((current_price - self.entry_price) / self.entry_price) * self.position * current_price
                self.total_pnl += pnl

                # Записываем сделку
                self.trades.append({
                    "step": self.current_step,
                    "type": "sell",
                    "price": current_price,
                    "quantity": self.position,
                    "revenue": revenue,
                    "balance": self.balance,
                    "pnl": pnl
                })

                # Обнуляем позицию и цену входа
                self.position = 0
                self.entry_price = 0.0

        # Обновляем пиковый баланс и максимальную просадку
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        current_drawdown = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # Рассчитываем вознаграждение
        if self.reward_function is not None:
            reward = self.reward_function(self, prev_balance, prev_position)
        else:
            # По умолчанию вознаграждение - изменение баланса
            reward = (self.balance - prev_balance) * self.reward_scaling

        # Переходим к следующему шагу
        self.current_step += 1

        # Получаем новое наблюдение
        observation = self._get_observation()

        # Проверяем условия завершения
        done = False

        # Завершаем эпизод, если баланс отрицательный
        if self.terminate_on_negative_balance and self.balance <= 0:
            done = True

        # Завершаем эпизод, если достигнут порог раннего завершения
        if self.early_stopping_threshold is not None:
            if (self.balance / self.initial_balance - 1) >= self.early_stopping_threshold:
                done = True

        # Информация о текущем состоянии
        info = {
            "balance": self.balance,
            "position": self.position,
            "current_price": current_price,
            "step": self.current_step,
            "total_pnl": self.total_pnl,
            "max_drawdown": self.max_drawdown
        }

        # Проверим, какая версия gymnasium используется
        try:
            # Проверяем наличие метода render() с параметром mode
            if hasattr(self, 'render') and 'mode' in self.render.__code__.co_varnames:
                # Старая версия Gym
                return observation, reward, done, info
            else:
                # Новая версия Gymnasium
                return observation, reward, done, False, info
        except:
            # Предполагаем новую версию Gymnasium
            return observation, reward, done, False, info

    def _get_observation(self) -> np.ndarray:
        """
        Формирует наблюдение для RL агента.

        Returns:
            Массив наблюдения
        """
        # Проверяем, достаточно ли данных
        if self.current_step < self.window_size:
            raise ValueError(
                f"Недостаточно данных для формирования наблюдения. current_step = {self.current_step}, window_size = {self.window_size}")

        # Получаем данные для окна наблюдения
        observation_window = self.data.iloc[self.current_step - self.window_size:self.current_step]

        # Выбираем признаки
        features = observation_window[self.feature_columns].values

        # Если нужно, добавляем информацию о позиции и балансе
        if self.use_position_features:
            # Нормализуем позицию и баланс
            normalized_position = np.full((self.window_size, 1), self.position / self.max_position)
            normalized_balance = np.full((self.window_size, 1), self.balance / self.initial_balance)

            # Объединяем все признаки
            features = np.concatenate([features, normalized_position, normalized_balance], axis=1)

        return features.astype(np.float32)

    def _get_current_price(self) -> float:
        """
        Возвращает текущую цену.

        Returns:
            Текущая цена
        """
        return self.data.iloc[self.current_step]['close']

    def close(self) -> None:
        """
        Закрывает среду.
        """
        pass

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Отображает текущее состояние среды.

        Args:
            mode: Режим отображения

        Returns:
            None или массив пикселей (для режима 'rgb_array')
        """
        # Для режима 'human' просто выводим информацию в лог
        if mode == 'human':
            current_price = self._get_current_price()
            logger.info(f"Step: {self.current_step}, Price: {current_price:.2f}, Balance: {self.balance:.2f}, "
                        f"Position: {self.position}, Total P&L: {self.total_pnl:.2f}, Max DD: {self.max_drawdown * 100:.2f}%")
            return None

        # Для режима 'rgb_array' можно реализовать визуализацию
        elif mode == 'rgb_array':
            # Здесь можно добавить код для создания визуализации
            # Например, с использованием matplotlib
            pass

        else:
            raise ValueError(f"Unsupported render mode: {mode}")


class RLModel(BaseModel):
    """
    Модель обучения с подкреплением для торговли.
    """

    def __init__(self, name: str, model_params: Dict = None):
        """
        Инициализирует RL модель.

        Args:
            name: Имя модели
            model_params: Параметры модели
                - algorithm: Алгоритм RL ('ppo', 'a2c', 'dqn', 'sac')
                - policy: Политика ('MlpPolicy', 'CnnPolicy', 'LstmPolicy')
                - policy_kwargs: Аргументы для политики
                - window_size: Размер окна для наблюдения
                - learning_rate: Скорость обучения
                - gamma: Коэффициент дисконтирования
                - batch_size: Размер батча
                - buffer_size: Размер буфера (для DQN и SAC)
                - train_freq: Частота обучения (для DQN и SAC)
                - gradient_steps: Количество шагов градиента (для DQN и SAC)
                - ent_coef: Коэффициент энтропии (для SAC)
                - env_params: Параметры для торговой среды
        """
        # Параметры по умолчанию
        default_params = {
            "algorithm": "ppo",
            "policy": "MlpPolicy",
            "policy_kwargs": None,
            "window_size": 60,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "batch_size": 64,
            "buffer_size": 100000,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": 0.01,
            "env_params": {
                "fee": 0.001,
                "initial_balance": 10000.0,
                "max_position": 1,
                "reward_scaling": 1.0,
                "use_position_features": True,
                "terminate_on_negative_balance": True,
                "early_stopping_threshold": None
            }
        }

        # Объединяем параметры по умолчанию с переданными
        if model_params:
            # Для вложенных словарей
            if model_params.get("env_params") and default_params.get("env_params"):
                default_params["env_params"].update(model_params.get("env_params", {}))
                # Удаляем, чтобы не переопределить полностью при следующем update
                if "env_params" in model_params:
                    del model_params["env_params"]

            default_params.update(model_params)

        super().__init__(name, default_params)

        self.env = None
        self.model = None
        self.vec_env = None

    def build(self) -> None:
        """
        Строит RL модель с заданными параметрами.
        Создает среду, но не инициализирует модель, так как для этого нужны данные.
        """
        self.is_built = False
        logger.info(f"Модель {self.name} будет построена при вызове train() с данными")

    def _create_environment(
            self,
            data: pd.DataFrame,
            feature_columns: List[str],
            training: bool = True,
            seed: int = 42
    ) -> gym.Env:
        """
        Создает и настраивает среду для RL агента.

        Args:
            data: DataFrame с данными
            feature_columns: Список колонок с признаками
            training: Флаг режима обучения
            seed: Зерно для воспроизводимости

        Returns:
            Настроенная среда
        """
        # Получаем параметры среды
        env_params = self.model_params.get("env_params", {})

        # Создаем директорию для логов, если она не существует
        log_dir = os.path.join("logs", "rl_training", self.name)
        os.makedirs(log_dir, exist_ok=True)

        # Создаем среду
        env = TradingEnvironment(
            data=data,
            feature_columns=feature_columns,
            window_size=self.model_params.get("window_size", 60),
            fee=env_params.get("fee", 0.001),
            initial_balance=env_params.get("initial_balance", 10000.0),
            max_position=env_params.get("max_position", 1),
            reward_scaling=env_params.get("reward_scaling", 1.0),
            use_position_features=env_params.get("use_position_features", True),
            reward_function=env_params.get("reward_function", None),
            terminate_on_negative_balance=env_params.get("terminate_on_negative_balance", True),
            early_stopping_threshold=env_params.get("early_stopping_threshold", None)
        )

        # Добавляем мониторинг среды для логирования
        env = Monitor(env, log_dir)

        # Создаем векторизованную среду (нужно для Stable Baselines3)
        vec_env = DummyVecEnv([lambda: env])

        # Нормализуем среду, если нужно
        if env_params.get("normalize_env", True):
            vec_env = VecNormalize(
                vec_env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
                gamma=self.model_params.get("gamma", 0.99),
                training=training
            )

        return vec_env

    def _create_model(self, env: gym.Env) -> Any:
        """
        Создает модель RL агента.

        Args:
            env: Среда для RL агента

        Returns:
            Модель RL агента
        """
        # Получаем параметры алгоритма
        algorithm = self.model_params.get("algorithm", "ppo").lower()
        policy = self.model_params.get("policy", "MlpPolicy")
        policy_kwargs = self.model_params.get("policy_kwargs", None)

        # Настраиваем общие параметры
        common_params = {
            "policy": policy,
            "env": env,
            "verbose": 1,
            "seed": self.model_params.get("seed", 42),
            "learning_rate": self.model_params.get("learning_rate", 0.0003),
            "gamma": self.model_params.get("gamma", 0.99),
            "tensorboard_log": os.path.join("logs", "tensorboard", self.name)
        }

        # Если заданы дополнительные параметры политики, добавляем их
        if policy_kwargs:
            common_params["policy_kwargs"] = policy_kwargs

        # Создаем модель в зависимости от алгоритма
        if algorithm == "ppo":
            model = PPO(
                **common_params,
                n_steps=self.model_params.get("n_steps", 2048),
                batch_size=self.model_params.get("batch_size", 64),
                n_epochs=self.model_params.get("n_epochs", 10),
                gae_lambda=self.model_params.get("gae_lambda", 0.95),
                clip_range=self.model_params.get("clip_range", 0.2),
                clip_range_vf=self.model_params.get("clip_range_vf", None),
                ent_coef=self.model_params.get("ent_coef", 0.01),
                max_grad_norm=self.model_params.get("max_grad_norm", 0.5)
            )

        elif algorithm == "a2c":
            model = A2C(
                **common_params,
                n_steps=self.model_params.get("n_steps", 5),
                ent_coef=self.model_params.get("ent_coef", 0.01),
                vf_coef=self.model_params.get("vf_coef", 0.5),
                max_grad_norm=self.model_params.get("max_grad_norm", 0.5),
                rms_prop_eps=self.model_params.get("rms_prop_eps", 1e-5),
                use_rms_prop=self.model_params.get("use_rms_prop", True),
                normalize_advantage=self.model_params.get("normalize_advantage", False)
            )

        elif algorithm == "dqn":
            model = DQN(
                **common_params,
                buffer_size=self.model_params.get("buffer_size", 100000),
                learning_starts=self.model_params.get("learning_starts", 1000),
                batch_size=self.model_params.get("batch_size", 64),
                tau=self.model_params.get("tau", 1.0),
                train_freq=self.model_params.get("train_freq", 4),
                gradient_steps=self.model_params.get("gradient_steps", 1),
                target_update_interval=self.model_params.get("target_update_interval", 100),
                exploration_fraction=self.model_params.get("exploration_fraction", 0.1),
                exploration_initial_eps=self.model_params.get("exploration_initial_eps", 1.0),
                exploration_final_eps=self.model_params.get("exploration_final_eps", 0.05),
                max_grad_norm=self.model_params.get("max_grad_norm", 10)
            )

        elif algorithm == "sac":
            model = SAC(
                **common_params,
                buffer_size=self.model_params.get("buffer_size", 100000),
                learning_starts=self.model_params.get("learning_starts", 1000),
                batch_size=self.model_params.get("batch_size", 64),
                tau=self.model_params.get("tau", 0.005),
                train_freq=self.model_params.get("train_freq", 1),
                gradient_steps=self.model_params.get("gradient_steps", 1),
                ent_coef=self.model_params.get("ent_coef", "auto"),
                target_update_interval=self.model_params.get("target_update_interval", 1),
                target_entropy=self.model_params.get("target_entropy", "auto")
            )

        else:
            raise ValueError(f"Неподдерживаемый алгоритм RL: {algorithm}")

        return model

    def train(
            self,
            X_train: np.ndarray,
            y_train: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            data: Optional[pd.DataFrame] = None,
            feature_columns: Optional[List[str]] = None,
            total_timesteps: int = 100000,
            eval_freq: int = 10000,
            n_eval_episodes: int = 5,
            callback: Optional[Any] = None,
            **kwargs
    ) -> Dict:
        """
        Обучает RL модель на заданных данных.

        Args:
            X_train: Обучающие данные (признаки) - не используется напрямую для RL
            y_train: Обучающие целевые значения - не используется напрямую для RL
            X_val: Валидационные данные (признаки) - не используется напрямую для RL
            y_val: Валидационные целевые значения - не используется напрямую для RL
            data: DataFrame с данными для обучения
            feature_columns: Список колонок с признаками
            total_timesteps: Общее количество шагов обучения
            eval_freq: Частота оценки модели
            n_eval_episodes: Количество эпизодов для оценки
            callback: Колбэк для обучения
            **kwargs: Дополнительные аргументы для обучения

        Returns:
            Словарь с историей обучения и метриками
        """
        # Проверяем, переданы ли необходимые данные
        if data is None:
            raise ValueError("Для обучения RL модели необходимо передать DataFrame с данными")

        if feature_columns is None:
            raise ValueError("Для обучения RL модели необходимо передать список колонок с признаками")

        # Сохраняем имена признаков
        self.feature_names = feature_columns

        # Создаем среду для обучения
        self.vec_env = self._create_environment(data, feature_columns, training=True)

        # Создаем модель
        self.model = self._create_model(self.vec_env)

        # Создаем директорию для логов, если она не существует
        log_dir = os.path.join("logs", "rl_training", self.name)
        os.makedirs(log_dir, exist_ok=True)

        # Разделяем данные на обучающую и валидационную части, если валидационные данные не переданы
        if X_val is None or y_val is None:
            # Для RL используем 80% данных для обучения и 20% для валидации
            train_size = int(len(data) * 0.8)
            train_data = data.iloc[:train_size].copy()
            val_data = data.iloc[train_size:].copy()
        else:
            train_data = data
            val_data = None

        # Создаем среду для валидации, если есть валидационные данные
        if val_data is not None:
            eval_env = self._create_environment(val_data, feature_columns, training=False)

            # Создаем колбэк для оценки модели
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(log_dir, "best_model"),
                log_path=os.path.join(log_dir, "results"),
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False
            )

            # Если колбэк не передан, используем eval_callback
            if callback is None:
                callback = eval_callback

        # Обучаем модель
        logger.info(f"Начинаем обучение модели {self.name}")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=self.name,
            **kwargs
        )

        # Устанавливаем флаг обученной модели
        self.is_trained = True

        # Сохраняем метаданные
        self.metadata["train_episodes"] = total_timesteps // self.model_params.get("n_steps", 2048)
        self.metadata["algorithm"] = self.model_params.get("algorithm", "ppo")
        self.metadata["policy"] = self.model_params.get("policy", "MlpPolicy")

        # Оцениваем модель на обучающих данных
        train_rewards, train_length = self._evaluate_model(train_data, feature_columns, n_episodes=10)
        self.metadata["train_mean_reward"] = np.mean(train_rewards)
        self.metadata["train_std_reward"] = np.std(train_rewards)

        # Оцениваем модель на валидационных данных
        if val_data is not None:
            val_rewards, val_length = self._evaluate_model(val_data, feature_columns, n_episodes=10)
            self.metadata["val_mean_reward"] = np.mean(val_rewards)
            self.metadata["val_std_reward"] = np.std(val_rewards)

        logger.info(f"Обучение модели {self.name} завершено")

        return {
            "train_mean_reward": self.metadata.get("train_mean_reward", 0),
            "train_std_reward": self.metadata.get("train_std_reward", 0),
            "val_mean_reward": self.metadata.get("val_mean_reward", 0),
            "val_std_reward": self.metadata.get("val_std_reward", 0)
        }

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Делает прогнозы (действия) на основе входных данных.

        Args:
            X: Входные данные для прогнозирования
            **kwargs: Дополнительные аргументы для прогнозирования

        Returns:
            Массив действий (0: держать, 1: покупать, 2: продавать)
        """
        if not self.is_trained:
            logger.warning(f"Модель {self.name} не обучена, невозможно сделать прогноз")
            return np.array([])

        # Проверяем размерность входных данных
        if len(X.shape) != 3:
            logger.warning(f"Неправильная размерность X: {X.shape}. Ожидается (samples, window_size, features)")
            logger.warning("Попытка преобразовать размерность...")

            if len(X.shape) == 2:
                # Если это 2D массив, предполагаем, что это (samples, features)
                # и преобразуем в (samples, 1, features)
                X = X.reshape(X.shape[0], 1, X.shape[1])

        # Делаем прогноз (получаем действия)
        deterministic = kwargs.get("deterministic", True)
        actions = []

        for i in range(len(X)):
            # Нормализуем наблюдение, если используется VecNormalize
            if isinstance(self.vec_env, VecNormalize):
                observation = self.vec_env.normalize_obs(X[i])
            else:
                observation = X[i]

            # Получаем действие
            action, _ = self.model.predict(observation, deterministic=deterministic)
            actions.append(action)

        return np.array(actions)

    def _evaluate_model(
            self,
            data: pd.DataFrame,
            feature_columns: List[str],
            n_episodes: int = 10,
            deterministic: bool = True
    ) -> Tuple[List[float], List[int]]:
        """
        Оценивает модель на заданных данных.

        Args:
            data: DataFrame с данными
            feature_columns: Список колонок с признаками
            n_episodes: Количество эпизодов для оценки
            deterministic: Флаг детерминированной политики

        Returns:
            Кортеж со списком вознаграждений и списком длин эпизодов
        """
        # Создаем среду для оценки
        eval_env = self._create_environment(data, feature_columns, training=False)

        # Списки для хранения результатов
        episode_rewards = []
        episode_lengths = []

        # Оцениваем модель
        for _ in range(n_episodes):
            # Проверяем версию возвращаемого значения reset()
            reset_result = eval_env.reset()

            # Обрабатываем разные версии API
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                obs, info = reset_result
            else:
                obs = reset_result
                info = {}

            done = False
            total_reward = 0
            step = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                step_result = eval_env.step(action)

                # Обрабатываем разные версии API (в новых версиях step возвращает 5 значений)
                if len(step_result) == 5:
                    obs, reward, done, truncated, info = step_result
                    if truncated:
                        break
                else:  # Для совместимости со старыми версиями Gym
                    obs, reward, done, info = step_result

                total_reward += reward
                step += 1

            episode_rewards.append(float(total_reward))
            episode_lengths.append(step)

        return episode_rewards, episode_lengths

    def backtest(
            self,
            data: pd.DataFrame,
            feature_columns: List[str],
            initial_balance: float = 10000.0,
            fee: float = 0.001,
            render: bool = False
    ) -> Dict:
        """
        Выполняет бэктестинг модели на исторических данных.

        Args:
            data: DataFrame с историческими данными
            feature_columns: Список колонок с признаками
            initial_balance: Начальный баланс
            fee: Комиссия за сделку (например, 0.001 = 0.1%)
            render: Флаг отображения процесса бэктестинга

        Returns:
            Словарь с результатами бэктестинга
        """
        if not self.is_trained:
            logger.warning(f"Модель {self.name} не обучена, невозможно выполнить бэктестинг")
            return {}

        # Создаем среду для бэктестинга
        env_params = self.model_params.get("env_params", {}).copy()
        env_params.update({
            "initial_balance": initial_balance,
            "fee": fee
        })

        # Создаем среду
        env = TradingEnvironment(
            data=data,
            feature_columns=feature_columns,
            window_size=self.model_params.get("window_size", 60),
            **env_params
        )

        # Выполняем бэктестинг
        obs, info = env.reset()
        done = False
        actions = []
        rewards = []
        balance_history = [initial_balance]
        position_history = [0]
        price_history = [env._get_current_price()]

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            actions.append(int(action))
            rewards.append(float(reward))
            balance_history.append(info["balance"])
            position_history.append(info["position"])
            price_history.append(info["current_price"])

            if render:
                env.render()

            if truncated:
                break

        # Формируем результаты бэктестинга
        backtest_results = {
            "initial_balance": initial_balance,
            "final_balance": balance_history[-1],
            "total_return": balance_history[-1] / initial_balance - 1,
            "total_return_pct": (balance_history[-1] / initial_balance - 1) * 100,
            "max_drawdown": env.max_drawdown * 100,
            "total_trades": len(env.trades),
            "actions": actions,
            "rewards": rewards,
            "balance_history": balance_history,
            "position_history": position_history,
            "price_history": price_history,
            "trades": env.trades
        }

        # Вычисляем дополнительные метрики
        if len(env.trades) > 0:
            # Общая прибыль и убыток
            total_pnl = env.total_pnl

            # Количество прибыльных и убыточных сделок
            profitable_trades = len([t for t in env.trades if t.get("pnl", 0) > 0])
            losing_trades = len([t for t in env.trades if t.get("pnl", 0) < 0])

            # Соотношение выигрышей и проигрышей
            win_rate = profitable_trades / len(env.trades) if len(env.trades) > 0 else 0

            # Средняя прибыль и убыток
            avg_profit = np.mean(
                [t.get("pnl", 0) for t in env.trades if t.get("pnl", 0) > 0]) if profitable_trades > 0 else 0
            avg_loss = np.mean([t.get("pnl", 0) for t in env.trades if t.get("pnl", 0) < 0]) if losing_trades > 0 else 0

            # Отношение среднего выигрыша к среднему проигрышу
            profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')

            # Фактор восстановления (Return / MaxDrawdown)
            recovery_factor = (backtest_results["total_return"] / env.max_drawdown) if env.max_drawdown > 0 else float(
                'inf')

            # Коэффициент Шарпа (если есть данные о доходности)
            returns = np.diff(balance_history) / balance_history[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

            # Добавляем метрики в результаты
            backtest_results.update({
                "total_pnl": total_pnl,
                "profitable_trades": profitable_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "win_rate_pct": win_rate * 100,
                "avg_profit": avg_profit,
                "avg_loss": avg_loss,
                "profit_loss_ratio": profit_loss_ratio,
                "recovery_factor": recovery_factor,
                "sharpe_ratio": sharpe_ratio
            })

        # Логируем результаты
        logger.info(f"Бэктестинг модели {self.name}:")
        logger.info(f"Начальный баланс: {initial_balance:.2f}")
        logger.info(f"Конечный баланс: {balance_history[-1]:.2f}")
        logger.info(f"Общая доходность: {backtest_results['total_return_pct']:.2f}%")
        logger.info(f"Максимальная просадка: {env.max_drawdown * 100:.2f}%")
        logger.info(f"Всего сделок: {len(env.trades)}")

        if len(env.trades) > 0:
            logger.info(f"Прибыльных сделок: {profitable_trades} ({win_rate * 100:.2f}%)")
            logger.info(f"Убыточных сделок: {losing_trades} ({(1 - win_rate) * 100:.2f}%)")
            logger.info(f"Отношение прибыли к убытку: {profit_loss_ratio:.2f}")
            logger.info(f"Фактор восстановления: {recovery_factor:.2f}")
            logger.info(f"Коэффициент Шарпа: {sharpe_ratio:.2f}")

        return backtest_results

    def save_stable_baselines_model(self, path: str) -> None:
        """
        Сохраняет только модель Stable Baselines (не весь объект).

        Args:
            path: Путь для сохранения модели
        """
        if not self.is_trained:
            logger.warning(f"Модель {self.name} не обучена, нечего сохранять")
            return

        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Сохраняем модель Stable Baselines
        self.model.save(path)

        # Если используется VecNormalize, сохраняем статистику нормализации
        if isinstance(self.vec_env, VecNormalize):
            vec_env_path = os.path.splitext(path)[0] + "_vecnormalize.pkl"
            self.vec_env.save(vec_env_path)

        logger.info(f"Модель Stable Baselines {self.name} сохранена в {path}")

    @classmethod
    def load_stable_baselines_model(
            cls,
            model_path: str,
            vec_env_path: Optional[str] = None,
            env: Optional[gym.Env] = None,
            name: str = "loaded_model"
    ) -> 'RLModel':
        """
        Загружает модель Stable Baselines и создает экземпляр RLModel.

        Args:
            model_path: Путь к файлу модели
            vec_env_path: Путь к файлу VecNormalize (опционально)
            env: Среда для загрузки модели (опционально)
            name: Имя для новой модели

        Returns:
            Экземпляр RLModel с загруженной моделью Stable Baselines
        """
        # Определяем алгоритм по расширению файла
        algorithm = os.path.basename(model_path).split("_")[0].lower()

        # Создаем экземпляр RLModel
        model_instance = cls(name=name, model_params={"algorithm": algorithm})

        # Если среда не передана, создаем заглушку
        if env is None:
            # Создаем простую среду для загрузки модели
            env = gym.make("CartPole-v1")
            vec_env = DummyVecEnv([lambda: env])

            # Если есть файл VecNormalize, загружаем его
            if vec_env_path and os.path.exists(vec_env_path):
                vec_env = VecNormalize.load(vec_env_path, vec_env)
                vec_env.training = False  # Отключаем обновление статистики
                vec_env.norm_reward = False  # Отключаем нормализацию вознаграждения
        else:
            vec_env = env

        # Загружаем модель Stable Baselines
        if algorithm == "ppo":
            model_instance.model = PPO.load(model_path, env=vec_env)
        elif algorithm == "a2c":
            model_instance.model = A2C.load(model_path, env=vec_env)
        elif algorithm == "dqn":
            model_instance.model = DQN.load(model_path, env=vec_env)
        elif algorithm == "sac":
            model_instance.model = SAC.load(model_path, env=vec_env)
        else:
            raise ValueError(f"Неподдерживаемый алгоритм RL: {algorithm}")

        # Устанавливаем атрибуты
        model_instance.vec_env = vec_env
        model_instance.is_trained = True

        logger.info(f"Модель Stable Baselines загружена из {model_path} в модель {name}")

        return model_instance