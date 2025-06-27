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
import numpy as np
from pyspark import StorageLevel
import matplotlib.pyplot as plt

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
    .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2") \
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

def save_analysis_results(df, yearly_stats, monthly_stats, daily_stats, yearly_vol, vol_stats, volume_price_corr, volume_by_day):
    """Сохранение результатов анализа в формате Parquet для использования в BI-системах."""
    try:
        # Создаем директорию для результатов, если она не существует
        output_dir = "analysis_results/data_parquet"
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nНачинаем сохранение результатов анализа в формате Parquet...")
        
        # Функция для сохранения DataFrame в Parquet
        def save_to_parquet(sdf, filename):
            # Сохраняем в Parquet
            filepath = os.path.join(output_dir, filename)
            # Используем coalesce(1) чтобы сохранить результат в один файл для удобства
            sdf.coalesce(1).write.mode("overwrite").parquet(filepath)
            print(f"Сохранены данные: {filepath}")
        
        # Сохранение всех результатов
        print("Сохранение годовой статистики...")
        save_to_parquet(yearly_stats, "yearly_stats")
        
        print("Сохранение месячной статистики...")
        save_to_parquet(monthly_stats, "monthly_stats")
        
        print("Сохранение дневной статистики...")
        save_to_parquet(daily_stats, "daily_stats")
        
        print("Сохранение годовой волатильности...")
        save_to_parquet(yearly_vol, "yearly_volatility")
        
        print("Сохранение объемов по дням...")
        save_to_parquet(volume_by_day, "volume_by_day")
        
        print("Сохранение корреляций...")
        save_to_parquet(volume_price_corr, "volume_price_correlation")
        
        # Сохранение последних 1000 записей с тех. индикаторами
        print("Сохранение последних 1000 записей с индикаторами...")
        df_with_indicators = df.orderBy(col("Date").desc()).limit(1000)
        save_to_parquet(df_with_indicators, "latest_data_with_indicators")

        # Проверяем, что директории созданы
        print("\nПроверка созданных директорий:")
        for dirname in ["yearly_stats", "monthly_stats", "daily_stats", 
                        "yearly_volatility", "volume_by_day", "volume_price_correlation", "latest_data_with_indicators"]:
            dirpath = os.path.join(output_dir, dirname)
            if os.path.isdir(dirpath):
                if len(os.listdir(dirpath)) > 0:
                    print(f"- {dirname}: OK")
                else:
                    print(f"ПРЕДУПРЕЖДЕНИЕ: Директория {dirname} пуста!")
            else:
                print(f"ОШИБКА: Директория {dirname} не создана!")
        
        print(f"\nРезультаты сохранены в директории '{output_dir}'. Эти файлы можно импортировать в Power BI или другие BI-инструменты.")
        
    except Exception as e:
        print(f"\nОШИБКА при сохранении результатов: {str(e)}")
        raise e

def print_summary_results(yearly_stats, monthly_stats, daily_stats, yearly_vol, vol_stats, volume_price_corr, volume_by_day, volume_stats, df):
    """Вывод кратких результатов анализа в консоль"""
    print("\n=== КРАТКИЕ РЕЗУЛЬТАТЫ АНАЛИЗА ===")
    
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
    yearly_stats_collected = yearly_stats.orderBy("Year").collect()
    if yearly_stats_collected:
        print(f"Средняя цена за последний год: ${yearly_stats_collected[-1]['avg_price']:,.2f}")
        
        # Рассчитываем общие мин/макс через Spark, чтобы избежать коллизии имен с Python-функциями
        overall_price_stats = yearly_stats.agg(
            max("max_price").alias("overall_max"),
            min("min_price").alias("overall_min")
        ).first()
        
        if overall_price_stats:
            print(f"Максимальная цена за все время: ${overall_price_stats['overall_max']:,.2f}")
            print(f"Минимальная цена за все время: ${overall_price_stats['overall_min']:,.2f}")
    
    # Месячные паттерны
    print("\nМесячные паттерны:")
    best_month_row = monthly_stats.orderBy(col("avg_price").desc()).first()
    worst_month_row = monthly_stats.orderBy(col("avg_price").asc()).first()
    if best_month_row and worst_month_row:
        print(f"Лучший месяц для торговли: {month_names.get(best_month_row['Month'], 'N/A')} (средняя цена: ${best_month_row['avg_price']:,.2f})")
        print(f"Худший месяц для торговли: {month_names.get(worst_month_row['Month'], 'N/A')} (средняя цена: ${worst_month_row['avg_price']:,.2f})")
    
    # Волатильность
    print("\nВолатильность:")
    volatility_agg = yearly_vol.agg(
        avg("YearlyVolatility").alias("avg_vol"),
        max("YearlyVolatility").alias("max_vol")
    ).first()
    if volatility_agg:
        print(f"Средняя годовая волатильность: {volatility_agg['avg_vol']:.2f}%")
        print(f"Максимальная годовая волатильность: {volatility_agg['max_vol']:.2f}%")

    # Объемы
    if volume_stats:
        print("\nОбъемы торгов (без нулевых значений):")
        print(f"Минимальный объем: {volume_stats['min_volume']:,.0f}")
        print(f"Максимальный объем: {volume_stats['max_volume']:,.0f}")
        print(f"Средний объем: {volume_stats['avg_volume']:,.0f}")
        print(f"Медианный объем: {volume_stats['median_volume']:,.0f}")
        print(f"Общий объем торгов: {volume_stats['total_volume']:,.0f}")
    
    # Объемы по дням недели
    print("\nОбъемы по дням недели:")
    volume_by_day_collected = volume_by_day.orderBy("DayOfWeek").collect()
    for row in volume_by_day_collected:
        day_name = day_names.get(row['DayOfWeek'], 'N/A')
        print(f"{day_name}:")
        print(f"  Количество записей: {row['count']:,}")
        print(f"  Медианный объем: {row['median_volume']:,.0f}")
        print(f"  Средний объем: {row['avg_volume']:,.0f}")
        print(f"  Общий объем: {row['total_volume']:,.0f}")
    
    # Корреляции
    print("\nКорреляции:")
    corr_value = volume_price_corr.first()
    if corr_value:
        print(f"Корреляция объема с ценой: {corr_value['correlation']:.2f}")
    
    print("\n--- Последние значения технических индикаторов ---")
    latest_record = df.orderBy(col("Date").desc()).first()
    if latest_record:
        print(f"Дата последней записи: {latest_record['Date']}")
        print(f"Последняя цена Close: ${latest_record['Close']:,.2f}")
        
        # Функция для безопасного форматирования
        def format_indicator(value, is_price=False):
            if value is not None and isinstance(value, (int, float)):
                prefix = "$" if is_price else ""
                return f"{prefix}{value:,.2f}"
            return "N/A"

        # Преобразуем Row в dict для безопасного использования .get()
        latest_record_dict = latest_record.asDict()

        print(f"RSI(14): {format_indicator(latest_record_dict.get('RSI14'))}")
        print(f"MACD: {format_indicator(latest_record_dict.get('MACD'))} (Сигнальная линия: {format_indicator(latest_record_dict.get('SignalLine'))})")
        print(f"Полосы Боллинджера (20):")
        print(f"  Верхняя: {format_indicator(latest_record_dict.get('UpperBand'), is_price=True)}")
        print(f"  Средняя (MA20): {format_indicator(latest_record_dict.get('MA20'), is_price=True)}")
        print(f"  Нижняя: {format_indicator(latest_record_dict.get('LowerBand'), is_price=True)}")

    print("\n=== КОНЕЦ ОТЧЕТА ===")

def plot_technical_indicators(df):
    """Визуализация технических индикаторов: полосы Боллинджера, MACD, RSI за последний год"""
    try:
        os.makedirs("analysis_results/plots", exist_ok=True)
        # Определяем последний год в данных
        last_year = df.select(max("Year")).collect()[0][0]
        # Фильтруем данные за последний год
        df_last_year = df.filter(col("Year") == last_year)
        df_pd = df_last_year.orderBy(col("Date")).toPandas()
        df_pd = df_pd.set_index('Date')

        # 1. Полосы Боллинджера
        plt.figure(figsize=(15, 7))
        plt.plot(df_pd.index, df_pd['Close'], label='Цена Close', color='blue')
        plt.plot(df_pd.index, df_pd['UpperBand'], label='Верхняя полоса Боллинджера', color='red', linestyle='--')
        plt.plot(df_pd.index, df_pd['LowerBand'], label='Нижняя полоса Боллинджера', color='green', linestyle='--')
        plt.plot(df_pd.index, df_pd['MA20'], label='MA 20', color='orange', linestyle=':')
        plt.title('Цена Bitcoin с Полосами Боллинджера (за последний год)')
        plt.xlabel('Дата')
        plt.ylabel('Цена')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('analysis_results/plots/bollinger_bands.png')
        plt.close()

        # 2. MACD
        plt.figure(figsize=(15, 7))
        plt.plot(df_pd.index, df_pd['MACD'], label='MACD', color='blue')
        plt.plot(df_pd.index, df_pd['SignalLine'], label='Сигнальная линия', color='red', linestyle='--')
        plt.bar(df_pd.index, df_pd['MACD_Hist'], label='Гистограмма', color='gray', alpha=0.5)
        plt.title('Индикатор MACD (за последний год)')
        plt.xlabel('Дата')
        plt.ylabel('Значение')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('analysis_results/plots/macd.png')
        plt.close()

        # 3. RSI
        plt.figure(figsize=(15, 7))
        plt.plot(df_pd.index, df_pd['RSI14'], label='RSI 14', color='purple')
        plt.axhline(70, linestyle='--', color='red', label='Перекупленность (70)')
        plt.axhline(30, linestyle='--', color='green', label='Перепроданность (30)')
        plt.title('Индикатор RSI (за последний год)')
        plt.xlabel('Дата')
        plt.ylabel('Значение RSI')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('analysis_results/plots/rsi.png')
        plt.close()

        print("Графики технических индикаторов за последний год сохранены в analysis_results/plots/")
    except Exception as e:
        print(f"Ошибка при построении графиков технических индикаторов: {str(e)}")
        raise

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
        
        # Сохранение результатов
        save_analysis_results(df, yearly_stats, monthly_stats, daily_stats,
                            yearly_vol, vol_stats, volume_price_corr, volume_by_day)
        
        # Визуализация технических индикаторов
        plot_technical_indicators(df)
        
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
