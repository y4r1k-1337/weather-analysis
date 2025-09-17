import predictor
import sys

if len(sys.argv) != 2:
    print("Usage: python predictor_demo.py [key]")
    exit(0)

params = ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature',
        'precipitation', 'rain', 'snowfall', 'snow_depth', 'surface_pressure', 'pressure_msl',
        'cloud_cover', 'wind_speed_10m', 'wind_gusts_10m', 'wind_direction_10m',
        'soil_temperature_0_to_7cm', 'soil_moisture_0_to_7cm', 'is_day', 'sunshine_duration']

data = predictor.get_weather_data(56.2, 44, '2020-01-01', '2025-09-01', params, timezone='Europe%2FMoscow')
X_train, X_test, y_train, y_test = predictor.split_by_key(data, "temperature_2m")
result = predictor.train_and_predict(X_train, X_test, y_train, y_test)
print(result["r2_score"])