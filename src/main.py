import argparse
import model
import dataset_splitter


def main():
    parser = argparse.ArgumentParser(description="Melanoma Detection Script")
    parser.add_argument(
        "mode",
        choices=["train", "eval", "infer", "split"],
        help="Mode: train, eval, infer, or split",
    )
    parser.add_argument(
        "csv_file", nargs="?", help="Path to the CSV file (train or eval)"
    )
    parser.add_argument(
        "batch_size", nargs="?", type=int, help="Batch size for DataLoader"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs for training"
    )
    parser.add_argument(
        "--model_path",
        help="Path to the model file for evaluation or inference",
    )
    parser.add_argument("--image_path", help="Path to the image file for inference")
    parser.add_argument("--img_dir", help="Directory containing images for splitting")
    parser.add_argument(
        "--output_dir", default=".", help="Directory to save the split datasets"
    )

    args = parser.parse_args()

    match args.mode:
        case "train":
            if not args.csv_file or not args.batch_size:
                parser.error("Training requires --csv_file and --batch_size")
            model.train(
                args.csv_file, args.batch_size, args.learning_rate, args.num_epochs
            )
        case "eval":
            if not args.csv_file or not args.batch_size:
                parser.error("Evaluation requires --csv_file and --batch_size")
            model.evaluate(args.csv_file, args.batch_size, args.model_path)
        case "infer":
            if not args.image_path:
                parser.error("Inference requires --image_path")
            model.inference(args.image_path, args.model_path)
        case "split":
            if not args.csv_file or not args.img_dir:
                parser.error("Splitting requires --csv_file and --img_dir")
            dataset_splitter.split(args.csv_file, args.img_dir, args.output_dir)


if __name__ == "__main__":
    main()
