"""
Copyright (C) 2025 Aleshenkov Yaroslav

This program is free software: you can redistribute it and/or 
modify it under the terms of the GNU General Public License as 
published by the Free Software Foundation, either version 3 of 
the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, 
but WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
GNU General Public License for more details.

You should have received a copy of the GNU General Public License 
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

"""
This is a pet project, no profit is intended, nor it is obtained
from using Open-Meteo API
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import requests
import re

base_url = "https://archive-api.open-meteo.com/v1/archive"

"""
Получение набора метеоданных с Open-Meteo
Вывод: таблица типа pandas.DataFrame
"""
def get_weather_data(latitude, longitude, start_date, end_date, hourly_params, timezone):
    hourly_params_str = re.sub(' ', ',', ' '.join(hourly_params))
    URL = base_url + f'?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&hourly={hourly_params_str}&timezone={timezone}&timeformat=unixtime'
    response = requests.get(URL)
    return pd.DataFrame(response.json()["hourly"]) #Данные за каждый час измерений

"""
Разделение набора на выборки по ключу (названию столбца в таблице)
Ввод: таблица типа pandas.DataFrame
Вывод: выборки для обучения и проверки работы регрессора
"""
def split_by_key(data, key):
    X = data.drop(key, axis=1)  #На основе этих данных программа выполняет расчеты
    y = data[key]               #Данные, которые должна выдать программа
    #Создаём выборки для обучения и проверки регрессора (модели машинного обучения для расчетов)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=128)
    return X_train, X_test, y_train, y_test

"""
Обучение и расчеты по выборкам.
Ввод: выборки для обучения и проверки; на основе X делаются расчеты, y - значения на выводе, для обучения (y_train) и ожидаемые значения (y_test), для проверки
Вывод: результат расчётов, метрики качества (оценка по R^2 Score и среднеквадратичная ошибка)
"""
def train_and_predict(X_train, X_test, y_train, y_test):
    #Для массивов хаотичных данных, таких как погодные параметры, модель случайного леса подходит лучше линейных моделей
    predictor = RandomForestRegressor(random_state=12)
    predictor.fit(X_train, y_train) #Обучение
    result = predictor.predict(X_test) #Рассчитывается результат
    return {"result":result.tolist(), "r2_score":r2_score(y_test, result), "mean_squared_error":mean_squared_error(y_test, result)}