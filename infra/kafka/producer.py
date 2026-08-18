import json
import time

import pandas as pd
from kafka import KafkaProducer


KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "user-interactions"
DATA_PATH = "data/processed/video_games.parquet"

EVENTS_TO_SEND = 100
DELAY_SECONDS = 0.05


def main():
    df = pd.read_parquet(DATA_PATH)

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    )

    print(f"Loaded {len(df):,} interactions")
    print(f"Sending {EVENTS_TO_SEND} events to '{TOPIC}'...")

    sample = df.head(EVENTS_TO_SEND)

    for _, row in sample.iterrows():
        event = {
            "user_idx": int(row["user_idx"]),
            "item_idx": int(row["item_idx"]),
            "rating": float(row["rating"]),
            "timestamp": int(row["timestamp"]),
        }

        producer.send(TOPIC, value=event)
        print(event)

        time.sleep(DELAY_SECONDS)

    producer.flush()
    producer.close()

    print("\nProducer finished.")


if __name__ == "__main__":
    main()
