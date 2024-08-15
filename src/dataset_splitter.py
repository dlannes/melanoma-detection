import os
import pandas as pd
from sklearn import model_selection


def split(csv_file: str, img_dir: str, output_dir: str) -> None:
    df = pd.read_csv(csv_file)
    df["image_path"] = df["image_id"].apply(lambda x: os.path.join(img_dir, f"{x}.jpg"))
    df = df[df["image_path"].apply(os.path.exists)]

    train_df, eval_df = model_selection.train_test_split(
        df, test_size=0.2, random_state=42
    )

    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, "train.csv")
    eval_file = os.path.join(output_dir, "eval.csv")
    train_df.to_csv(train_file, index=False)
    eval_df.to_csv(eval_file, index=False)

    print(f"Training data saved to {train_file}")
    print(f"Evaluation data saved to {eval_file}")
