import os
import logging
import warnings
warnings.filterwarnings('ignore')
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)
logging.getLogger("netlib").setLevel(logging.ERROR)
logging.getLogger("com.github.fommil.netlib").setLevel(logging.ERROR)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_unixtime, to_timestamp, year, month, dayofmonth, log, avg, count, min, max, when, lit, date_add, lag, hour, dayofweek, stddev, corr, percentile_approx, sum, expr
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyspark import StorageLevel
import seaborn as sns

# Инициализация Spark сессии с оптимизациями для больших данных
spark = SparkSession.builder \
    .appName("BitcoinMarketAnalysis") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "4g") \
    .config("spark.sql.parquet.compression.codec", "snappy") \
    .config("spark.sql.parquet.mergeSchema", "true") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.default.parallelism", "200") \
    .config("spark.hadoop.fs.defaultFS", "file:///") \
    .getOrCreate()

def load_and_preprocess_data():
    """Загрузка и предобработка данных"""
    schema = "Timestamp DOUBLE, Open DOUBLE, High DOUBLE, Low DOUBLE, Close DOUBLE, Volume DOUBLE"
    
    df = spark.read.csv("bitcoin.csv", 
                       header=True, 
                       schema=schema) \
           .repartition(200)
    
    # Базовая фильтрация
    df = df.filter((col("Volume") > 0) & 
                  (col("Open") > 0) & 
                  (col("Close") > 0) &
                  (col("High") > 0) &
                  (col("Low") > 0))
    
    # Временные признаки
    df = df.withColumn("Date", to_timestamp(from_unixtime(col("Timestamp")))).drop("Timestamp")
    df = df.withColumn("Year", year(col("Date")))
    df = df.withColumn("Month", month(col("Date")))
    df = df.withColumn("Day", dayofmonth(col("Date")))
    df = df.withColumn("Hour", hour(col("Date")))
    df = df.withColumn("DayOfWeek", dayofweek(col("Date")))
    
    # Расчет дополнительных метрик
    df = df.withColumn("PriceChangePerc", (col("Close") - col("Open")) / col("Open"))
    df = df.withColumn("Volatility", (col("High") - col("Low")) / col("Open"))
    df = df.withColumn("LogVolume", log(col("Volume") + 1))
    
    return df

def analyze_trends(df):
    """Анализ трендов"""
    # Годовые тренды
    yearly_stats = df.groupBy(year("Date").alias("Year")) \
        .agg(
            avg("Close").alias("avg_price"),
            max("Close").alias("max_price"),
            min("Close").alias("min_price"),
            avg("Volume").alias("avg_volume")
        ) \
        .orderBy("Year")
    
    # Месячные паттерны
    monthly_stats = df.groupBy(month("Date").alias("Month")) \
        .agg(
            avg("Close").alias("avg_price"),
            avg("Volume").alias("avg_volume")
        ) \
        .orderBy("Month")
    
    # Дневные паттерны
    daily_stats = df.groupBy(dayofweek("Date").alias("DayOfWeek")) \
        .agg(
            avg("Close").alias("avg_price"),
            avg("Volume").alias("avg_volume")
        ) \
        .orderBy("DayOfWeek")
    
    return yearly_stats, monthly_stats, daily_stats

def calculate_technical_indicators(df):
    """Расчет технических индикаторов"""
    # Глобальное окно для корректного расчета скользящих средних без разрывов
    window_spec_global = Window.orderBy("Date")

    # Рассчитываем лаг один раз
    df = df.withColumn("PrevClose", lag("Close", 1).over(window_spec_global))

    # Скользящие средние
    for period in [7, 12, 14, 20, 26, 30, 90]:
        df = df.withColumn(f"MA{period}", 
                         avg("Close").over(window_spec_global.rowsBetween(-period + 1, 0)))
    
    # Bollinger Bands (20-периодное окно)
    df = df.withColumn("StdDev20", 
                     stddev("Close").over(window_spec_global.rowsBetween(-19, 0)))
    df = df.withColumn("UpperBand", col("MA20") + (col("StdDev20") * 2))
    df = df.withColumn("LowerBand", col("MA20") - (col("StdDev20") * 2))

    # MACD (12, 26, 9)
    # Используем аппроксимацию EMA через SMA из-за ограничений Spark
    df = df.withColumn("MACD", col("MA12") - col("MA26"))
    # Signal line for MACD (9-периодная MA от MACD)
    df = df.withColumn("SignalLine", avg("MACD").over(window_spec_global.rowsBetween(-8, 0)))
    df = df.withColumn("MACD_Hist", col("MACD") - col("SignalLine"))
    
    # RSI (Relative Strength Index)
    # Используем абсолютное изменение цены, как в классической формуле
    df = df.withColumn("PriceChange", col("Close") - col("PrevClose"))
    
    for period in [14, 30]:
        window = window_spec_global.rowsBetween(-period + 1, 0)
        df = df.withColumn(f"Gain{period}", 
                         when(col("PriceChange") > 0, col("PriceChange")).otherwise(0))
        df = df.withColumn(f"Loss{period}", 
                         when(col("PriceChange") < 0, -col("PriceChange")).otherwise(0))
        
        # Для RSI лучше использовать EMA, но SMA - хорошее приближение в Spark
        df = df.withColumn(f"AvgGain{period}", 
                         avg(f"Gain{period}").over(window))
        df = df.withColumn(f"AvgLoss{period}", 
                         avg(f"Loss{period}").over(window))
        
        # Избегаем деления на ноль
        df = df.withColumn(f"RS{period}", 
                         when(col(f"AvgLoss{period}") > 0, col(f"AvgGain{period}") / col(f"AvgLoss{period}"))
                         .otherwise(None)) # Если потерь нет, RS стремится к бесконечности
        df = df.withColumn(f"RSI{period}", 
                         when(col(f"RS{period}").isNotNull(), 100 - (100 / (1 + col(f"RS{period}"))))
                         .otherwise(100)) # Если потерь не было (AvgLoss=0), RSI принимается за 100
    
    # Удаляем временные колонки, чтобы не засорять DataFrame
    df = df.drop("PrevClose", "PriceChange")
    for period in [14, 30]:
        df = df.drop(f"Gain{period}", f"Loss{period}", f"AvgGain{period}", f"AvgLoss{period}", f"RS{period}")

    return df

def analyze_volatility(df):
    """Анализ волатильности"""
    # Годовая волатильность
    yearly_vol = df.groupBy(year("Date").alias("Year")) \
        .agg(
            (stddev("Close") / avg("Close") * 100).alias("YearlyVolatility")
        ) \
        .orderBy("Year")
    
    # Общая статистика волатильности
    vol_stats = df.select(
        (stddev("Close") / avg("Close") * 100).alias("OverallVolatility")
    )
    
    return yearly_vol, vol_stats

def analyze_volumes(df):
    """Анализ объемов торгов"""
    # Фильтруем данные, оставляя только записи с положительным объемом
    df_filtered = df.filter(col("Volume") > 0)
    
    # Рассчитываем статистику объемов для определения выбросов
    volume_stats = df_filtered.select(
        min("Volume").alias("min_volume"),
        max("Volume").alias("max_volume"),
        avg("Volume").alias("avg_volume"),
        stddev("Volume").alias("std_volume"),
        percentile_approx("Volume", 0.5).alias("median_volume"),
        sum("Volume").alias("total_volume"),
        count("Volume").alias("filtered_records")
    ).collect()[0]
    
    print("\nСтатистика после фильтрации нулевых объемов:")
    print(f"Количество записей: {volume_stats['filtered_records']:,}")
    print(f"Минимальный объем: {volume_stats['min_volume']:,.2f}")
    print(f"Максимальный объем: {volume_stats['max_volume']:,.2f}")
    print(f"Средний объем: {volume_stats['avg_volume']:,.2f}")
    print(f"Медианный объем: {volume_stats['median_volume']:,.2f}")
    print(f"Стандартное отклонение: {volume_stats['std_volume']:,.2f}")
    print(f"Общий объем торгов: {volume_stats['total_volume']:,.2f}")
    
    # Определяем границы для выбросов (3 стандартных отклонения)
    lower_bound = volume_stats.avg_volume - 3 * volume_stats.std_volume
    if lower_bound < 0:
        lower_bound = 0
    upper_bound = volume_stats.avg_volume + 3 * volume_stats.std_volume
    
    print(f"\nГраницы для выбросов:")
    print(f"Нижняя граница: {lower_bound:,.2f}")
    print(f"Верхняя граница: {upper_bound:,.2f}")
    
    # Фильтруем выбросы, но сохраняем все данные для анализа по дням
    df_filtered = df_filtered.filter(
        (col("Volume") >= lower_bound) & 
        (col("Volume") <= upper_bound))
    
    # Проверяем количество записей после фильтрации
    total_records = df_filtered.count()
    if total_records == 0:
        raise ValueError("Нет данных для анализа объемов после фильтрации")
    
    # Объемы по дням недели (используем исходные данные без фильтрации выбросов)
    volume_by_day = df.filter(col("Volume") > 0).groupBy(dayofweek("Date").alias("DayOfWeek")) \
        .agg(
            percentile_approx("Volume", 0.5).alias("median_volume"),
            sum("Volume").alias("total_volume"),
            avg("Volume").alias("avg_volume"),
            count("Volume").alias("count")
        ) \
        .orderBy("DayOfWeek")
    
    # Корреляция объема с ценой (используем отфильтрованные данные)
    volume_price_corr = df_filtered.select(
        corr("Volume", "Close").alias("correlation")
    )
    
    return volume_price_corr, volume_by_day, volume_stats

def validate_data(df):
    """Валидация данных"""
    # Проверка на выбросы в ценах
    price_stats = df.select(
        min("Close").alias("min_price"),
        max("Close").alias("max_price"),
        avg("Close").alias("avg_price"),
        stddev("Close").alias("std_price")
    ).collect()[0]
    
    # Определяем границы для выбросов (5 стандартных отклонений вместо 3)
    lower_bound = price_stats.avg_price - 5 * price_stats.std_price
    upper_bound = price_stats.avg_price + 5 * price_stats.std_price
    
    # Фильтруем только явные выбросы, сохраняя исторические максимумы
    df_cleaned = df.filter(
        (col("Close") >= lower_bound) | 
        (col("Close") >= price_stats.max_price * 0.95)  # Сохраняем значения близкие к максимуму
    )
    
    print(f"\nСтатистика цен после очистки:")
    print(f"Минимальная цена: ${df_cleaned.select(min('Close')).collect()[0][0]:,.2f}")
    print(f"Максимальная цена: ${df_cleaned.select(max('Close')).collect()[0][0]:,.2f}")
    print(f"Средняя цена: ${df_cleaned.select(avg('Close')).collect()[0][0]:,.2f}")
    
    return df_cleaned

def visualize_results(df, yearly_stats, monthly_stats, daily_stats, yearly_vol):
    """Визуализация результатов анализа"""
    try:
        # Создаем директорию для графиков, если она не существует
        os.makedirs("analysis_results/plots", exist_ok=True)
        
        print("\nНачинаем создание визуализаций...")
        
        # Преобразование в pandas для визуализации
        print("Преобразование данных в pandas...")
        yearly_stats_pd = yearly_stats.toPandas()
        monthly_stats_pd = monthly_stats.toPandas()
        daily_stats_pd = daily_stats.toPandas()
        yearly_vol_pd = yearly_vol.toPandas()
        volume_by_day_pd = df.groupBy("DayOfWeek").agg(avg("Volume").alias("avg_volume")).orderBy("DayOfWeek").toPandas()
        
        # Для детальных графиков возьмем данные за последний год
        print("Подготовка данных для детальных графиков (последний год)...")
        last_year = df.select(max("Year")).collect()[0][0]
        df_last_year_pd = df.filter(col("Year") == last_year).toPandas()
        df_last_year_pd = df_last_year_pd.sort_values('Date').set_index('Date')
        
        # Создание основных графиков
        print("Создание основных графиков...")
        plt.figure(figsize=(15, 12))
        
        # 1. Годовые тренды
        plt.subplot(2, 2, 1)
        plt.plot(yearly_stats_pd['Year'], yearly_stats_pd['avg_price'])
        plt.title('Годовые тренды цены')
        plt.xlabel('Год')
        plt.ylabel('Средняя цена')
        plt.grid(True)
        
        # 2. Месячные паттерны
        plt.subplot(2, 2, 2)
        plt.bar(monthly_stats_pd['Month'], monthly_stats_pd['avg_price'])
        plt.title('Месячные паттерны цены')
        plt.xlabel('Месяц')
        plt.ylabel('Средняя цена')
        plt.grid(True)
        
        # 3. Волатильность по годам
        plt.subplot(2, 2, 3)
        plt.plot(yearly_vol_pd['Year'], yearly_vol_pd['YearlyVolatility'])
        plt.title('Годовая волатильность')
        plt.xlabel('Год')
        plt.ylabel('Волатильность (%)')
        plt.grid(True)
        
        # 4. Объем по дням недели
        plt.subplot(2, 2, 4)
        plt.bar(volume_by_day_pd['DayOfWeek'], volume_by_day_pd['avg_volume'])
        plt.title('Средний объем по дням недели')
        plt.xlabel('День недели')
        plt.ylabel('Средний объем')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Сохранение основного графика
        print("Сохранение основного графика...")
        plt.savefig('analysis_results/plots/bitcoin_analysis_main.png')
        plt.close()
        
        # Дополнительные визуализации с техническими индикаторами
        print("\nСоздание графиков с техническими индикаторами...")
        
        # 1. Цена с Полосами Боллинджера
        plt.figure(figsize=(15, 7))
        plt.plot(df_last_year_pd.index, df_last_year_pd['Close'], label='Цена Close', color='blue')
        plt.plot(df_last_year_pd.index, df_last_year_pd['UpperBand'], label='Верхняя полоса Боллинджера', color='red', linestyle='--')
        plt.plot(df_last_year_pd.index, df_last_year_pd['LowerBand'], label='Нижняя полоса Боллинджера', color='green', linestyle='--')
        plt.plot(df_last_year_pd.index, df_last_year_pd['MA20'], label='MA 20', color='orange', linestyle=':')
        plt.title('Цена Bitcoin с Полосами Боллинджера (за последний год)')
        plt.xlabel('Дата')
        plt.ylabel('Цена')
        plt.legend()
        plt.grid(True)
        plt.savefig('analysis_results/plots/bollinger_bands.png')
        plt.close()

        # 2. MACD индикатор
        plt.figure(figsize=(15, 7))
        plt.plot(df_last_year_pd.index, df_last_year_pd['MACD'], label='MACD', color='blue')
        plt.plot(df_last_year_pd.index, df_last_year_pd['SignalLine'], label='Сигнальная линия', color='red', linestyle='--')
        plt.bar(df_last_year_pd.index, df_last_year_pd['MACD_Hist'], label='Гистограмма', color='gray', alpha=0.5)
        plt.title('Индикатор MACD (за последний год)')
        plt.xlabel('Дата')
        plt.ylabel('Значение')
        plt.legend()
        plt.grid(True)
        plt.savefig('analysis_results/plots/macd.png')
        plt.close()

        # 3. RSI индикатор
        plt.figure(figsize=(15, 7))
        plt.plot(df_last_year_pd.index, df_last_year_pd['RSI14'], label='RSI 14', color='purple')
        plt.axhline(70, linestyle='--', color='red', label='Перекупленность (70)')
        plt.axhline(30, linestyle='--', color='green', label='Перепроданность (30)')
        plt.title('Индикатор RSI (за последний год)')
        plt.xlabel('Дата')
        plt.ylabel('Значение RSI')
        plt.legend()
        plt.grid(True)
        plt.savefig('analysis_results/plots/rsi.png')
        plt.close()

        # 4. Корреляционная матрица
        corr_matrix = pd.DataFrame({
            'Цена': yearly_stats_pd['avg_price'],
            'Объем': yearly_stats_pd['avg_volume'],
            'Волатильность': yearly_vol_pd['YearlyVolatility']
        }).corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Корреляционная матрица (годовые данные)')
        plt.savefig('analysis_results/plots/correlation_matrix.png')
        plt.close()
        
        # Проверка создания файлов
        print("\nПроверка созданных графиков:")
        for filename in ['bitcoin_analysis_main.png', 'bollinger_bands.png', 'macd.png', 'rsi.png', 'correlation_matrix.png']:
            filepath = f"analysis_results/plots/{filename}"
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"- {filename}: {size} байт")
            else:
                print(f"ОШИБКА: График {filename} не был сохранен!")
        
    except Exception as e:
        print(f"\nОШИБКА при создании визуализаций: {str(e)}")
        raise e

def save_analysis_results(df, yearly_stats, monthly_stats, daily_stats, yearly_vol, vol_stats, volume_price_corr, volume_by_day):
    """Сохранение результатов анализа"""
    try:
        # Создаем директорию для результатов, если она не существует
        os.makedirs("analysis_results/data", exist_ok=True)
        
        print("\nНачинаем сохранение результатов анализа...")
        
        # Функция для сохранения DataFrame в CSV
        def save_to_csv(pdf, filename):
            # Сохраняем в CSV
            filepath = f"analysis_results/data/{filename}.csv"
            pdf.to_csv(filepath, index=False)
            print(f"Сохранен файл: {filename}.csv")
        
        # Сохранение всех результатов
        print("Сохранение годовой статистики...")
        save_to_csv(yearly_stats.toPandas(), "yearly_stats")
        
        print("Сохранение месячной статистики...")
        save_to_csv(monthly_stats.toPandas(), "monthly_stats")
        
        print("Сохранение дневной статистики...")
        save_to_csv(daily_stats.toPandas(), "daily_stats")
        
        print("Сохранение годовой волатильности...")
        save_to_csv(yearly_vol.toPandas(), "yearly_volatility")
        
        print("Сохранение объемов по дням...")
        save_to_csv(volume_by_day.toPandas(), "volume_by_day")
        
        print("Сохранение корреляций...")
        save_to_csv(volume_price_corr.toPandas(), "volume_price_correlation")
        
        # Сохранение последних 1000 записей с тех. индикаторами
        print("Сохранение последних 1000 записей с индикаторами...")
        df_with_indicators_pd = df.orderBy(col("Date").desc()).limit(1000).toPandas()
        save_to_csv(df_with_indicators_pd, "latest_data_with_indicators")

        # Проверяем, что файлы созданы
        print("\nПроверка созданных файлов:")
        for filename in ["yearly_stats", "monthly_stats", "daily_stats", 
                        "yearly_volatility", "volume_by_day", "volume_price_correlation", "latest_data_with_indicators"]:
            filepath = f"analysis_results/data/{filename}.csv"
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"- {filename}.csv: {size} байт")
            else:
                print(f"ОШИБКА: Файл {filename}.csv не создан!")
        
        print("\nРезультаты сохранены в директории 'analysis_results/data'")
        
    except Exception as e:
        print(f"\nОШИБКА при сохранении результатов: {str(e)}")
        raise e

def print_summary_results(yearly_stats, monthly_stats, daily_stats, yearly_vol, vol_stats, volume_price_corr, volume_by_day, volume_stats, df):
    """Вывод кратких результатов анализа в консоль"""
    print("\n=== КРАТКИЕ РЕЗУЛЬТАТЫ АНАЛИЗА ===")
    
    # Преобразуем в pandas для удобства вывода
    yearly_stats_pd = yearly_stats.toPandas()
    monthly_stats_pd = monthly_stats.toPandas()
    daily_stats_pd = daily_stats.toPandas()
    yearly_vol_pd = yearly_vol.toPandas()
    volume_by_day_pd = volume_by_day.toPandas()
    
    # Словари для преобразования числовых значений в названия
    month_names = {
        1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
        5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
        9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь"
    }
    
    day_names = {
        1: "Воскресенье", 2: "Понедельник", 3: "Вторник",
        4: "Среда", 5: "Четверг", 6: "Пятница", 7: "Суббота"
    }
    
    # Годовые тренды
    print("\nГодовые тренды:")
    print(f"Средняя цена за последний год: ${yearly_stats_pd['avg_price'].iloc[-1]:,.2f}")
    print(f"Максимальная цена за все время: ${yearly_stats_pd['max_price'].max():,.2f}")
    print(f"Минимальная цена за все время: ${yearly_stats_pd['min_price'].min():,.2f}")
    
    # Месячные паттерны
    print("\nМесячные паттерны:")
    best_month = monthly_stats_pd.loc[monthly_stats_pd['avg_price'].idxmax()]
    worst_month = monthly_stats_pd.loc[monthly_stats_pd['avg_price'].idxmin()]
    print(f"Лучший месяц для торговли: {month_names[int(best_month['Month'])]} (средняя цена: ${best_month['avg_price']:,.2f})")
    print(f"Худший месяц для торговли: {month_names[int(worst_month['Month'])]} (средняя цена: ${worst_month['avg_price']:,.2f})")
    
    # Волатильность
    print("\nВолатильность:")
    print(f"Средняя годовая волатильность: {yearly_vol_pd['YearlyVolatility'].mean():.2f}%")
    print(f"Максимальная годовая волатильность: {yearly_vol_pd['YearlyVolatility'].max():.2f}%")
    
    # Объемы
    print("\nОбъемы торгов (без нулевых значений):")
    print(f"Минимальный объем: {volume_stats['min_volume']:,.0f}")
    print(f"Максимальный объем: {volume_stats['max_volume']:,.0f}")
    print(f"Средний объем: {volume_stats['avg_volume']:,.0f}")
    print(f"Медианный объем: {volume_stats['median_volume']:,.0f}")
    print(f"Общий объем торгов: {volume_stats['total_volume']:,.0f}")
    
    # Объемы по дням недели
    print("\nОбъемы по дням недели:")
    for _, row in volume_by_day_pd.iterrows():
        day_name = day_names[int(row['DayOfWeek'])]
        print(f"{day_name}:")
        print(f"  Количество записей: {row['count']}")
        print(f"  Медианный объем: {row['median_volume']:,.0f}")
        print(f"  Средний объем: {row['avg_volume']:,.0f}")
        print(f"  Общий объем: {row['total_volume']:,.0f}")
    
    # Корреляции
    print("\nКорреляции:")
    print(f"Корреляция объема с ценой: {volume_price_corr.toPandas()['correlation'].iloc[0]:.2f}")
    
    print("\n--- Последние значения технических индикаторов ---")
    latest_record = df.orderBy(col("Date").desc()).first()
    if latest_record:
        print(f"Дата последней записи: {latest_record['Date']}")
        print(f"Последняя цена Close: ${latest_record['Close']:,.2f}")
        print(f"RSI(14): {latest_record['RSI14']:.2f}")
        print(f"MACD: {latest_record['MACD']:.2f} (Сигнальная линия: {latest_record['SignalLine']:.2f})")
        print(f"Полосы Боллинджера (20):")
        print(f"  Верхняя: ${latest_record['UpperBand']:,.2f}")
        print(f"  Средняя (MA20): ${latest_record['MA20']:,.2f}")
        print(f"  Нижняя: ${latest_record['LowerBand']:,.2f}")

    print("\n=== КОНЕЦ ОТЧЕТА ===")

def main():
    """Основная функция"""
    try:
        # Загрузка и предобработка данных
        df = load_and_preprocess_data()
        
        # Валидация данных
        df = validate_data(df)
        
        # Фильтрация нулевых объемов
        df = df.filter(col("Volume") > 0)
        
        # Кэшируем DataFrame для повторного использования
        df.cache()
        
        # Расчет технических индикаторов
        df = calculate_technical_indicators(df)
        
        # Анализ трендов
        yearly_stats, monthly_stats, daily_stats = analyze_trends(df)
        
        # Анализ волатильности
        yearly_vol, vol_stats = analyze_volatility(df)
        
        # Анализ объемов
        volume_price_corr, volume_by_day, volume_stats = analyze_volumes(df)
        
        # Визуализация результатов
        visualize_results(df, yearly_stats, monthly_stats, daily_stats, 
                         yearly_vol)
        
        # Сохранение результатов
        save_analysis_results(df, yearly_stats, monthly_stats, daily_stats,
                            yearly_vol, vol_stats, volume_price_corr, volume_by_day)
        
        # Вывод итоговых результатов
        print_summary_results(yearly_stats, monthly_stats, daily_stats,
                            yearly_vol, vol_stats, volume_price_corr, volume_by_day, volume_stats, df)
        
        # Освобождаем кэш
        df.unpersist()
        
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
        raise

if __name__ == "__main__":
    main()
