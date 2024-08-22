import os
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import cx_Oracle
from prophet import Prophet

# Настройка Oracle клиента
oracle_client_path = r"C:\Users\aalik\Documents\instantclient_23_4"
if not os.path.exists(oracle_client_path):
    raise Exception(f"Oracle client path does not exist: {oracle_client_path}")

os.environ["PATH"] = oracle_client_path + ";" + os.environ["PATH"]

# Функция для подключения и извлечения данных из Oracle через SQLAlchemy
def fetch_data_from_oracle(user, password, host, port, service_name, query):
    dsn = URL.create(
        "oracle+cx_oracle",
        username=user,
        password=password,
        host=host,
        port=port,
        database=service_name,
    )
    engine = create_engine(dsn)
    
    # Выполнение запроса и получение данных
    with engine.connect() as connection:
        result = connection.execution_options(stream_results=True).execute(query)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    engine.dispose()
    return df

# Параметры подключения к Oracle
user = ''
password = ''
host = ''
port = 1521
service_name = 'ORCL'

# SQL-запрос для извлечения данных о реализации топлива
query = """
SELECT  rec.OBJECTCODE AS АЗС_Code,
         item.prodname AS prodname, 
         item.tanknum AS tanknum, 
         TO_CHAR(rec.STARTTIMESTAMP, 'dd.mm.yy') AS ДАТА,
         TO_CHAR(TO_NUMBER(TO_CHAR(rec.STARTTIMESTAMP, 'hh24')) + 1) AS HOUR,
         SUM(item.Volume) AS КОЛИЧЕСТВО
FROM BI.RECEIPTS rec
RIGHT JOIN bi.ZREPORTS z 
    ON (z.OBJECTCODE = rec.objectcode 
        AND z.ZREPORTNUM = rec.znum 
        AND z.STARTTIMESTAMP = rec.zSTARTTIMESTAMP 
        AND EXTRACT(YEAR FROM z.STARTTIMESTAMP) IN (2022, 2023, 2024))
LEFT JOIN BI.RECEIPTITEMS item 
    ON item.RECEIPTID = rec.id
WHERE rec.objectcode = 'Z313' 
    AND ITEM.PRODTYPE = 0 
    AND (item.BRUTO <> 0 OR (item.VOLUME <> 0 AND ITEM.PRODTYPE = 0 
        AND rec.objectcode IN ('X347', 'X345', 'X343', 'X364', 'X344', 'X348', 'X342', 'X351', 'X349', 'X341', 'X346', 'X350')))
GROUP BY rec.OBJECTCODE, 
         item.prodname, 
         item.tanknum, 
         TO_CHAR(rec.STARTTIMESTAMP, 'dd.mm.yy'), 
         TO_CHAR(TO_NUMBER(TO_CHAR(rec.STARTTIMESTAMP, 'hh24')) + 1)
"""

# Получение данных из Oracle
df_oracle = fetch_data_from_oracle(user, password, host, port, service_name, query)

# Удаляем возможные пробелы и другие скрытые символы в именах колонок
df_oracle.columns = df_oracle.columns.str.strip()

# Преобразуем дату и час в формат datetime и объединим их
df_oracle['Дата и Время'] = pd.to_datetime(df_oracle['ДАТА'], format='%d.%m.%y') + pd.to_timedelta(df_oracle['hour'].astype(int), unit='h')

# Устанавливаем временную метку как индекс
df_oracle.set_index('Дата и Время', inplace=True)

# Группировка данных по каждому часу и расчет среднего значения
df_oracle = df_oracle.groupby([df_oracle.index.floor('H'), 'prodname']).agg({'КОЛИЧЕСТВО': 'mean'}).reset_index()

# Получаем список уникальных типов НП
prod_types = df_oracle['prodname'].unique()

# Визуализация для каждого типа НП
for prod in prod_types:
    df_prod = df_oracle[df_oracle['prodname'] == prod].copy()

    # Расчет текущего уровня топлива и мертвого остатка
    last_date = df_prod['Дата и Время'].max().date()
    current_fuel_level = df_prod[df_prod['Дата и Время'].dt.date == last_date]['КОЛИЧЕСТВО'].sum()

    # Находим минимальные значения по каждому дню и вычисляем мертвый остаток
    df_prod['Дата'] = df_prod['Дата и Время'].dt.date
    daily_min_volume = df_prod.groupby('Дата')['КОЛИЧЕСТВО'].min()
    dead_stock_volume = daily_min_volume.mean()

    # Добавление скользящего среднего
    df_prod['rolling_mean'] = df_prod['КОЛИЧЕСТВО'].rolling(window=24).mean()

    # Подготовка данных для Prophet
    df_prophet = df_prod[['Дата и Время', 'rolling_mean']].dropna().rename(columns={'Дата и Время': 'ds', 'rolling_mean': 'y'})

    # Инициализация модели Prophet
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(df_prophet)

    # Прогнозирование на 24 часа вперед
    future = model.make_future_dataframe(periods=24, freq='h')
    forecast = model.predict(future)

    # Визуализация фактических данных, скользящего среднего и прогноза
    plt.figure(figsize=(14, 7))
    plt.plot(df_prod['Дата и Время'], df_prod['КОЛИЧЕСТВО'], label='Фактический объем')
    plt.plot(df_prod['Дата и Время'], df_prod['rolling_mean'], label='Скользящее среднее')
    plt.plot(forecast['ds'], forecast['yhat'], label='Прогнозируемый объем')
    plt.axhline(y=dead_stock_volume, color='r', linestyle='--', label='Мертвый остаток')
    plt.axhline(y=current_fuel_level, color='g', linestyle='--', label='Текущий уровнемер')
    plt.xlabel('Дата и время')
    plt.ylabel('Объем')
    plt.title(f'Прогноз и фактические данные для {prod}')
    plt.legend()
    plt.show()
