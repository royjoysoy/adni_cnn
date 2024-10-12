# Alzheimer's Disease Classification using SimCLR

This project implements a 3D Convolutional Neural Network (CNN) with SimCLR (Simple Framework for Contrastive Learning of Visual Representations) for Alzheimer's disease classification using MRI images. The model is designed to classify brain scans into three categories: Alzheimer's Disease (AD), Mild Cognitive Impairment (MCI), and Healthy Control (HC).

## Project Overview

The main components of this project include:

1. Data loading and preprocessing
2. SimCLR model implementation
3. Training with various data augmentation techniques
4. Linear evaluation
5. Final model evaluation

## Prerequisites

- Python 3.7+
- PyTorch
- torchio
- nibabel
- scikit-image
- pandas
- numpy
- matplotlib
- seaborn
- tqdm

## Project Structure

- `data_loading_RS_oct.py`: Contains functions for data loading and preprocessing
- `model_RS_oct.py`: Defines the SimCLR3DCNN model architecture
- `training_RS_oct.py`: Implements the training loop, linear evaluation, and final evaluation
- `utils_RS_oct.py`: Utility functions for argument parsing, model saving/loading, and hardware info
- `main_RS_oct.py`: The main script that orchestrates the entire training process

## Usage

To run the training process:
python main_RS_oct.py --data_path /path/to/your/data.csv --save_dir /path/to/save/results

Additional arguments can be found in the `get_args()` function in `utils_RS_oct.py`.

## Features

- SimCLR training with various data augmentation techniques
- Ablation study to compare different augmentation methods
- Linear evaluation of the trained model
- Final evaluation with detailed metrics and confusion matrix
- TensorBoard integration for monitoring training progress
- Mixed precision training for improved performance

## Results

The script will output:

- Training and validation losses
- Best model checkpoints
- Linear evaluation accuracy
- Final evaluation report with precision, recall, F1-score, F3-score, and F5-score for each class
- Confusion matrix
- Training time reports

## Contributing

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT License](https://opensource.org/licenses/MIT)
