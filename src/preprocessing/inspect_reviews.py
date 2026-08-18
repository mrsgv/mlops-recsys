from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    countDistinct,
    count,
    avg,
    min,
    max,
)

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

    print("\n=== Unique Users ===")
    print(reviews.select(countDistinct("user_id")).collect()[0][0])

    print("\n=== Unique Products ===")
    print(reviews.select(countDistinct("parent_asin")).collect()[0][0])

    print("\n=== Reviews Per User ===")

    user_stats = (
        reviews
        .groupBy("user_id")
        .agg(count("*").alias("review_count"))
    )

    user_stats.select(
        avg("review_count").alias("average"),
        min("review_count").alias("minimum"),
        max("review_count").alias("maximum"),
    ).show()

    print("\n=== Reviews Per Product ===")

    product_stats = (
        reviews
        .groupBy("parent_asin")
        .agg(count("*").alias("review_count"))
    )

    product_stats.select(
        avg("review_count").alias("average"),
        min("review_count").alias("minimum"),
        max("review_count").alias("maximum"),
    ).show()

    print("\n=== User Interaction Distribution ===")
    print("\n=== Users and Interactions by Minimum User History ===")

    thresholds = [1, 2, 3, 4, 5, 10]

    for threshold in thresholds:
        filtered_users = user_stats.filter(
            col("review_count") >= threshold
        )

        user_count = filtered_users.count()

        interaction_count = (
            filtered_users
            .selectExpr("sum(review_count) as total_interactions")
            .collect()[0]["total_interactions"]
        )

        print(
            f"Minimum {threshold} reviews: "
            f"{user_count} users, "
            f"{interaction_count} interactions"
        )

    user_stats.groupBy("review_count").count().orderBy("review_count").show(
        30,
        truncate=False
    )

    print("\n=== Products and Interactions by Minimum Product History ===")

    product_thresholds = [1, 2, 3, 5, 10, 20]

    for threshold in product_thresholds:
        filtered_products = product_stats.filter(
            col("review_count") >= threshold
        )

        product_count = filtered_products.count()

        interaction_count = (
            filtered_products
            .selectExpr("sum(review_count) as total_interactions")
            .collect()[0]["total_interactions"]
        )

        print(
            f"Minimum {threshold} reviews: "
            f"{product_count} products, "
            f"{interaction_count} interactions"
        )
        
    print("\n=== Rating Distribution ===")

    (
        reviews
        .groupBy("rating")
        .count()
        .orderBy("rating")
        .show()
    )

    spark.stop()


if __name__ == "__main__":
    main()
