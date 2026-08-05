from pyspark.sql import SparkSession


def main():
    spark = (
        SparkSession.builder
        .appName("Inspect Amazon Beauty Reviews")
        .getOrCreate()
    )

    reviews = spark.read.json("data/raw/All_Beauty.jsonl.gz")

    print("\n=== Schema ===")
    reviews.printSchema()

    print("\n=== First 5 Rows ===")
    reviews.show(5, truncate=False)

    print("\n=== Total Reviews ===")
    print(reviews.count())

    spark.stop()


if __name__ == "__main__":
    main()
