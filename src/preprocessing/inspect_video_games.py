from pyspark.sql import SparkSession
from pyspark.sql.functions import count, avg, min, max


def main():
    spark = (
        SparkSession.builder
        .appName("InspectVideoGames")
        .master("local[*]")
        .getOrCreate()
    )

    path = "data/raw/Video_Games.csv.gz"

    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(path)
    )

    print("\n=== Schema ===")
    df.printSchema()

    print("\n=== First 10 Rows ===")
    df.show(10, truncate=False)

    print("\n=== Total Interactions ===")
    print(df.count())

    print("\n=== Unique Users ===")
    print(df.select("user_id").distinct().count())

    print("\n=== Unique Products ===")
    print(df.select("parent_asin").distinct().count())

    print("\n=== Rating Distribution ===")
    (
        df.groupBy("rating")
        .count()
        .orderBy("rating")
        .show()
    )

    print("\n=== Reviews Per User ===")
    (
        df.groupBy("user_id")
        .count()
        .agg(
            avg("count").alias("average"),
            min("count").alias("minimum"),
            max("count").alias("maximum")
        )
        .show()
    )

    print("\n=== Reviews Per Product ===")
    (
        df.groupBy("parent_asin")
        .count()
        .agg(
            avg("count").alias("average"),
            min("count").alias("minimum"),
            max("count").alias("maximum")
        )
        .show()
    )

    print("\n=== Duplicate User-Product Interactions ===")
    duplicate_count = (
        df.groupBy("user_id", "parent_asin")
        .count()
        .filter("count > 1")
        .count()
    )
    print(duplicate_count)

    print("\n=== Timestamp Range ===")
    (
        df.selectExpr(
            "from_unixtime(min(timestamp) / 1000) as earliest",
            "from_unixtime(max(timestamp) / 1000) as latest"
        )
        .show(truncate=False)
    )

    print("\n=== Interactions Per Year ===")
    (
        df.selectExpr(
            "year(from_unixtime(timestamp / 1000)) as year"
        )
        .groupBy("year")
        .count()
        .orderBy("year")
        .show()
    )

    spark.stop()


if __name__ == "__main__":
    main()
