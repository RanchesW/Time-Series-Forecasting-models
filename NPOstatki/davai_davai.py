import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Загрузка данных
file_path = '313_sorted.csv'
df = pd.read_csv(file_path)

# Конвертируем колонку с датами в формат datetime
df['ДАТА'] = pd.to_datetime(df['ДАТА'], format='%Y-%m-%d')

# Если час = 24, уменьшаем его до 0 и добавляем день к дате
df.loc[df['HOUR'] == 24, 'HOUR'] = 0
df.loc[df['HOUR'] == 0, 'ДАТА'] = df['ДАТА'] + pd.Timedelta(days=1)

# Фильтруем данные для АИ-92-К4/К5 и 2-го танка, исключая данные с 27 до 28 августа 2023 года
filtered_df = df[(df['PRODNAME'] == 'АИ-92-К4/К5') & (df['TANKNUM'] == 2.0) & 
                 ~df['ДАТА'].between('2023-08-27', '2023-08-28')]

# Создаем столбец с полным временем (дата + час)
filtered_df['ds'] = pd.to_datetime(filtered_df['ДАТА'].astype(str) + ' ' + filtered_df['HOUR'].astype(str) + ':00:00')

# Добавляем столбец с днем недели для поиска аналогичных дней
filtered_df['weekday'] = filtered_df['ds'].dt.weekday

# Функция для поиска аналогичных дней недели и времени в прошлых данных
def find_similar_days(target_weekday, target_hour, data, window_size=3):
    similar_days = data[
        (data['weekday'] == target_weekday) & 
        (data['HOUR'] == target_hour)
    ]
    # Рассчитываем среднее значение по найденным аналогичным дням
    rolling_mean = similar_days['КОЛИЧЕСТВО'].rolling(window=window_size, min_periods=1).mean().iloc[-1]
    return rolling_mean

# Прогноз на 27-28 августа 2023 года
forecast_dates = pd.date_range(start='2023-08-27', end='2023-08-28', freq='H')
forecast_values = []

for date in forecast_dates:
    target_weekday = date.weekday()
    target_hour = date.hour
    rolling_mean_value = find_similar_days(target_weekday, target_hour, filtered_df)
    forecast_values.append(rolling_mean_value)

# Создаем DataFrame для прогноза на основе скользящего среднего
forecast_df = pd.DataFrame({'ds': forecast_dates, 'yhat': forecast_values})

# Обучение Prophet на данных до 27 августа 2023 года
training_df = filtered_df[filtered_df['ds'] < '2023-08-27']
prophet_model = Prophet()
prophet_model.fit(training_df[['ds', 'КОЛИЧЕСТВО']].rename(columns={'КОЛИЧЕСТВО': 'y'}))

# Прогноз на период с 27 до 28 августа 2023 года с помощью Prophet
future = pd.DataFrame({'ds': forecast_dates})
forecast_prophet = prophet_model.predict(future)

# Добавление данных объема за август
volume_data = {
    'DATE': [
        '2023-08-01', '2023-08-02', '2023-08-03', '2023-08-04', '2023-08-05', 
        '2023-08-06', '2023-08-07', '2023-08-08', '2023-08-09', '2023-08-10',
        '2023-08-11', '2023-08-12', '2023-08-13', '2023-08-14', '2023-08-15',
        '2023-08-16', '2023-08-17', '2023-08-18', '2023-08-19', '2023-08-20',
        '2023-08-21', '2023-08-22', '2023-08-23', '2023-08-24', '2023-08-25',
        '2023-08-26', '2023-08-27', '2023-08-28', '2023-08-29'
    ],
    'VOLUME': [
        8910, 7749, 10119, 9999, 12137, 8336, 9388, 10261, 12593, 21970, 
        15339, 13272, 9171, 14012, 17759, 11444, 12630, 10697, 12565, 10818, 
        21004, 7638, 10048, 13153, 10228, 13428, 11106, 13952, 4809
    ]
}

volume_df = pd.DataFrame(volume_data)
volume_df['DATE'] = pd.to_datetime(volume_df['DATE']).dt.date  # Приводим к типу даты

# Объединение прогноза с данными объёма
forecast_prophet['DATE'] = forecast_prophet['ds'].dt.date
forecast_prophet = forecast_prophet.merge(volume_df[['DATE', 'VOLUME']], on='DATE', how='left')

# Рассчитываем уровнемер, начиная с объёма на начало каждого нового дня
forecast_prophet['уровнемер'] = forecast_prophet.groupby('DATE').apply(
    lambda x: x['VOLUME'].iloc[0] - x['yhat'].cumsum()
).reset_index(drop=True)

# Проверка, когда уровень достигнет мертвого остатка (2998 литров)
dead_stock = 2998
below_dead_stock = forecast_prophet[forecast_prophet['уровнемер'] <= dead_stock]

if not below_dead_stock.empty:
    first_date_reach_dead_stock = below_dead_stock.iloc[0][['DATE', 'ds']]
    print(f"Мертвый остаток будет достигнут к: {first_date_reach_dead_stock['DATE']} в {first_date_reach_dead_stock['ds'].hour}:00")
else:
    print("Мертвый остаток не будет достигнут в пределах 27-28 августа 2023 года.")

# Фактические данные за 27-28 августа 2023 года
actual_df = df[(df['PRODNAME'] == 'АИ-92-К4/К5') & (df['TANKNUM'] == 2.0) & 
               df['ДАТА'].between('2023-08-27', '2023-08-28')]

actual_df['ds'] = pd.to_datetime(actual_df['ДАТА'].astype(str) + ' ' + actual_df['HOUR'].astype(str) + ':00:00')

# Визуализация
plt.figure(figsize=(16, 8))

# Исторические данные до 27 августа
plt.plot(filtered_df['ds'], filtered_df['КОЛИЧЕСТВО'], label='Фактический объем до 27 августа', color='blue')

# Прогноз скользящего среднего на период с 27 до 28 августа
plt.plot(forecast_df['ds'], forecast_df['yhat'], label='Прогноз (скользящее среднее)', color='orange')

# Прогноз Prophet на период с 27 до 28 августа
plt.plot(forecast_prophet['ds'], forecast_prophet['yhat'], label='Прогноз Prophet', color='red')

# Линия уровнемера
plt.plot(forecast_prophet['ds'], forecast_prophet['уровнемер'], label='Уровнемер', color='purple', linestyle='--')

# Линия мертвого остатка
plt.axhline(y=dead_stock, color='green', linestyle='--', label='Мертвый остаток (2998 литров)')

# Фактические данные за 27-28 августа
plt.plot(actual_df['ds'], actual_df['КОЛИЧЕСТВО'], label='Фактический объем за 27-28 августа', color='green')

# Добавление легенды и заголовка
plt.legend()
plt.title('Прогноз на 27-28 августа 2023 года с фактическими данными, аналогичными днями недели и Prophet')
plt.xlabel('Дата и время')
plt.ylabel('Объем реализации')

# Показать график
plt.show()
