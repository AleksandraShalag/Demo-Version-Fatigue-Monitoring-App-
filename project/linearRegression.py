import sqlite3

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import UndefinedMetricWarning
import warnings

from tqdm import tk



def predict_time_to_threshold(data, threshold=80, min_data_points=5, time_window=60):
    """
    Прогнозирует время достижения порогового значения усталости
    Параметры:
    - data: DataFrame с колонкой 'fatigue_level'
    - threshold: пороговое значение усталости (0-100)
    - min_data_points: минимальное количество точек для анализа
    - time_window: временное окно анализа в минутах

    Возвращает:
    - predicted_time: прогнозируемое время до достижения порога (минут)
    - confidence: уверенность прогноза (0-1)
    - current_trend: текущий тренд ('rising', 'stable', 'falling')
    """

    # Инициализация результата по умолчанию
    result = {
        'predicted_time': None,
        'confidence': 0.0,
        'current_trend': 'stable',
        'error': None
    }

    try:
        # Проверка входных данных
        if len(data) < min_data_points:
            raise ValueError(f"Недостаточно данных. Требуется минимум {min_data_points} точек")

        # Подготовка данных
        y = data['fatigue_level'].values[-time_window:]
        X = np.arange(len(y)).reshape(-1, 1)

        # Обучение модели
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            model = LinearRegression()
            model.fit(X, y)

        # Расчет характеристик
        slope = model.coef_[0]
        current_value = y[-1]
        residual = np.std(y - model.predict(X))

        # Определение тренда
        if slope > 0.5:
            result['current_trend'] = 'rising'
        elif slope < -0.5:
            result['current_trend'] = 'falling'
        else:
            result['current_trend'] = 'stable'

        # Проверка уже превышенного порога
        if current_value >= threshold:
            result['predicted_time'] = 0
            result['confidence'] = 1.0
            return result

        # Прогноз времени достижения порога
        if slope <= 0:
            result['predicted_time'] = float('inf')
            result['confidence'] = 0.0
        else:
            time_to_threshold = (threshold - model.intercept_) / slope - len(y)
            result['predicted_time'] = max(0, time_to_threshold)

            # Расчет уверенности
            confidence = 1 - np.tanh(residual / 10)  # Эмпирическая формула
            result['confidence'] = np.clip(confidence, 0.0, 1.0)

    except Exception as e:
        result['error'] = str(e)

    return result
