"""Export existing Gemma results from detections.csv to a focused CSV."""

import csv

from config import output_file
from log import GEMMA_FIELDS


def main():
    source = output_file("detections.csv")
    destination = output_file("gemma_predictions.csv")

    with open(source, newline="", encoding="utf-8") as src, open(
        destination, "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=GEMMA_FIELDS)
        writer.writeheader()

        count = 0
        for row in reader:
            if not row.get("hazard"):
                continue
            writer.writerow(
                {
                    "timestamp": row["timestamp"],
                    "image": row["image"],
                    "pipeline": row["pipeline"],
                    "yolo_label": row["label"],
                    "yolo_confidence": row["confidence"],
                    "species": row["species"],
                    "common_name": row["common_name"],
                    "bioclip_confidence": row["species_score"],
                    "prediction": row["hazard"],
                    "reason": row["hazard_reason"],
                }
            )
            count += 1

    print(f"Exported {count} Gemma predictions to {destination}")


if __name__ == "__main__":
    main()
