
import os
import pandas as pd


class DatasetManager:

    def __init__(
        self,
        dataset_path="/content/SignVisionAI/data/sign_dataset.csv"
    ):
        self.dataset_path = dataset_path

    def save_samples(self, samples, label):

        rows = []

        for features in samples:

            row = list(features)
            row.append(label)

            rows.append(row)

        columns = [
            f"feature_{i}"
            for i in range(63)
        ]

        columns.append("label")

        new_data = pd.DataFrame(
            rows,
            columns=columns
        )

        if os.path.exists(self.dataset_path):

            old_data = pd.read_csv(
                self.dataset_path
            )

            new_data = pd.concat(
                [old_data, new_data],
                ignore_index=True
            )

        new_data.to_csv(
            self.dataset_path,
            index=False
        )

        print(
            f"{len(samples)} samples saved "
            f"for sign '{label}'"
        )
