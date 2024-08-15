# Melanoma Detection with CNN

A command-line tool for detecting melanoma from dermatoscopic images using a Convolutional Neural Network (CNN). This tool supports dataset splitting, model training, evaluation, and inference.

## Usage

### 1. Split Dataset
Split the original dataset into training and evaluation sets:

```bash
python tool.py split original_dataset.csv --img_dir path/to/images --output_dir path/to/output
```

### 2. Train the Model
Train the model using the training dataset:

```bash
python tool.py train path/to/output/train.csv 32 --learning_rate 0.001 --num_epochs 10
```

### 3. Evaluate the Model
Evaluate the model using the evaluation dataset:

```bash
python tool.py eval path/to/output/eval.csv 32 --model_path melanoma_cnn.pth
```

### 4. Run Inference
Predict melanoma on a single image:

```bash
python tool.py infer --image_path path/to/image.jpg --model_path melanoma_cnn.pth
```

## Requirements

- Python 3.x
- Install the dependencies with `pip install -r requirements.txt`
