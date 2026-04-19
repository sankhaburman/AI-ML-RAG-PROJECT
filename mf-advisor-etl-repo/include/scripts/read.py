from pyspark.sql import SparkSession

def main():
    spark = SparkSession.builder \
        .appName("PySpark Example") \
        .getOrCreate()

    df = spark.read.csv("./include/data.csv", header=True)

    # Print schema
    print("Printing Schema!!!!")
    df.printSchema()

    # Show all columns and values (safe limit)
    df.show(100, truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()