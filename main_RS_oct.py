import argparse
import logging
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import torch
import torch.optim as optim
import torchio as tio
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from sklearn.model_selection import train_test_split

from model_RS_oct import SimCLR3DCNN
from data_loading_RS_oct import get_data_loaders, get_mr_transforms
from training_RS_oct import run_ablation_study, linear_eval, final_evaluation
from utils_RS_oct import get_args, save_model, plot_learning_curves, get_hardware_info

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    args = get_args()
    logging.info("Starting the SimCLR training process from scratch")

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    current_datetime = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H-%M-%S")

    # Add file handler
    file_handler = logging.FileHandler(os.path.join(args.save_dir, f'output_{current_datetime}.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    # Redirect stdout and stderr to the log file
    sys.stdout = open(os.path.join(args.save_dir, f'output_{current_datetime}.log'), 'w')
    sys.stderr = sys.stdout

    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    simclr_batch_size = args.simclr_batch_size # 16 parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training') # SimCLR batch siz
    linear_eval_batch_size = args.linear_eval_batch_size 
    simclr_epochs = args.simclr_epochs # 50 # parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train') # SimCLR epoch set
    linear_eval_epochs = args.linear_eval_epochs
    num_classes = 3
    scaler = GradScaler()

    logging.info(f"SimCLR parameters: feature_dim={feature_dim}, temperature={temperature}")

    logging.info(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    logging.info(f"Data split completed. Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    # Set up transforms
    logging.info("Setting up data transforms")
    # redaundant from mr_transforms in data_loading_RS_oct.py file
    # base_transform = tio.Compose([
    #    tio.ToCanonical(),
    #    #tio.RescaleIntensity(out_min_max=(0, 1)),
    #    #tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    #    #tio.CropOrPad((64, 64, 64))
    # ])
    mr_transforms = get_mr_transforms()
    transforms = {
        'base': mr_transforms['base'],
        'flip': mr_transforms['flip'],
        'anisotropy': mr_transforms['anisotropy'],
        'swap': mr_transforms['swap'],
        'elastic': mr_transforms['elastic'],
        'bias_field': mr_transforms['bias_field'],
        'blur': mr_transforms['blur'],
        'gamma': mr_transforms['gamma'],
        'spike': mr_transforms['spike'],
        'ghost': mr_transforms['ghost'],
        'noise': mr_transforms['noise'],
        'motion': mr_transforms['motion'],
        'mixup': mr_transforms['mixup'],
        'cutmix': mr_transforms['cutmix']     
    }
    logging.info(f"Transforms set up. Available augmentations: {list(transforms.keys())}")

    # Create data loaders
    logging.info("Creating data loaders")
    train_loaders, val_loader, test_loader = get_data_loaders(train_df, val_df, test_df, simclr_batch_size, transforms)
    logging.info(f"Data loaders created. Number of train loaders: {len(train_loaders)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    net = SimCLR3DCNN(feature_dim=feature_dim, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    simclr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.scheduler_factor, 
                                         patience=args.scheduler_patience, verbose=True)
    logging.info("Model, optimizer, and learning rate scheduler initialized")

    # Initialize TensorBoard writer
    writer = SummaryWriter(os.path.join(args.save_dir, f'tensorboard_logs_{current_datetime}'))
    logging.info("TensorBoard writer initialized")

    # Run ablation study with time tracking
    logging.info(f"Starting ablation study for {simclr_epochs} epochs")
    simclr_start_time = time.time()
    results, best_models, total_simclr_time = run_ablation_study(net, train_loaders, val_loader, optimizer, 
                                                                 simclr_scheduler, simclr_epochs, temperature, 
                                                                 simclr_batch_size, k, save_top_n=5, writer=writer, 
                                                                 scaler=scaler)
    simclr_end_time = time.time()
    simclr_time = simclr_end_time - simclr_start_time

    # Log ablation study results
    logging.info("Ablation study results:")
    with open(os.path.join(args.save_dir, f'ablation_results_{current_datetime}.txt'), 'w') as f:
        for aug_type, data in results.items():
            avg_loss = sum(data['train_loss']) / len(data['train_loss'])
            avg_acc = sum(data['val_accuracy']) / len(data['val_accuracy'])
            result_line = f"{aug_type}: Avg Loss: {avg_loss:.4f}, Avg Val Accuracy: {avg_acc:.2f}%"
            logging.info(result_line)
            f.write(result_line + '\n')
    
    logging.info(f"SimCLR training completed. Final SimCLR epoch: {simclr_epochs}")

    # Save the final model
    model_path = os.path.join(args.save_dir, f'latest_model_{current_datetime}.pth')
    save_model(net, optimizer, simclr_epochs, model_path)
    logging.info(f"Final model saved to {model_path}")

    # Save the model path
    with open(os.path.join(args.save_dir, f'latest_model_path_{current_datetime}.txt'), 'w') as f:
        f.write(model_path)
    logging.info(f"Latest model path saved to {os.path.join(args.save_dir, f'latest_model_path_{current_datetime}.txt')}")

    # Generate and save learning curves
    plot_learning_curves(results, args.save_dir, current_datetime)
    logging.info("Learning curves generated and saved")

    # Save and rank the top models
    for i, model_info in enumerate(best_models):
        model_path = os.path.join(args.save_dir, f'model_rank_{i+1}_{current_datetime}.pth')
        torch.save(model_info['state_dict'], model_path)
        logging.info(f"Rank {i+1}: SimCLR Epoch {model_info['epoch']}, Aug: {model_info['aug_type']}, "
                     f"Accuracy: {model_info['accuracy']:.2f}%, Saved as: {os.path.basename(model_path)}")

    # Linear evaluation on the best model
    logging.info("Starting Linear Evaluation on the best model")
    if best_models:
        net.load_state_dict(best_models[0]['state_dict'])
        logging.info("Loaded best model from ablation study")
    else:
        logging.warning("No best models found. Using current model state for linear evaluation.")

    net = net.to(device)
    net.add_classifier()
    net.eval()
    logging.info("Model prepared for linear evaluation")
    linear_optimizer = optim.AdamW(net.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    logging.info("Linear evaluation optimizer created")
    linear_eval_start_time = time.time()
    best_linear_eval_acc, best_epoch, linear_eval_time = linear_eval(net, train_loaders['base'], val_loader, # only based on 'base' transformation in the linearl eval!
                                                                     linear_optimizer, linear_eval_epochs=linear_eval_epochs, linear_eval_batch_size=linear_eval_batch_size, writer=writer) # linear eval epochs 25
    logging.info(f"Best Linear Evaluation Accuracy: {best_linear_eval_acc:.2f}% at epoch {best_epoch+1}")

    # Final evaluation
    logging.info("Starting final evaluation")
    final_report, final_accuracy = final_evaluation(net, test_loader, device, args.save_dir)

    # Print final results
    logging.info(f"Total SimCLR epochs: {simclr_epochs}")
    logging.info(f"Total Linear Evaluation epochs: {linear_eval_epochs}")
    logging.info("\nFinal Evaluation Results:")
    logging.info(f"Accuracy: {final_report['accuracy']:.4f}")

    # Log F1-scores
    macro_f1 = final_report.get('macro avg', {}).get('f1-score', 'N/A')
    weighted_f1 = final_report.get('weighted avg', {}).get('f1-score', 'N/A')
    logging.info(f"Macro Avg F1-score: {macro_f1:.4f}" if isinstance(macro_f1, float) else f"Macro Avg F1-score: {macro_f1}")
    logging.info(f"Weighted Avg F1-score: {weighted_f1:.4f}" if isinstance(weighted_f1, float) else f"Weighted Avg F1-score: {weighted_f1}")

    # Save final evaluation results to a text file
    with open(os.path.join(args.save_dir, f'final_evaluation_results_{current_datetime}.txt'), 'w') as f:
        f.write(f"Accuracy: {final_report['accuracy']:.4f}\n")
        f.write(f"Macro Avg F1-score: {macro_f1:.4f}\n" if isinstance(macro_f1, float) else f"Macro Avg F1-score: {macro_f1}\n")
        f.write(f"Weighted Avg F1-score: {weighted_f1:.4f}\n" if isinstance(weighted_f1, float) else f"Weighted Avg F1-score: {weighted_f1}\n")
        f.write("\nPer-class metrics:\n")
        for i, class_name in enumerate(['CN', 'MCI', 'Dementia']):
            f.write(f"{class_name}:\n")
            f.write(f"  Precision: {final_report['precision'][i]:.4f}\n")
            f.write(f"  Recall: {final_report['recall'][i]:.4f}\n")
            f.write(f"  F1-score: {final_report['f1'][i]:.4f}\n")
            f.write(f"  F3-score: {final_report['f3'][i]:.4f}\n")
            f.write(f"  F5-score: {final_report['f5'][i]:.4f}\n")
            f.write(f"  Support: {final_report['support'][i]}\n")

    logging.info("\nPer-class metrics:")
    for i, class_name in enumerate(['CN', 'MCI', 'Dementia']):
        logging.info(f"{class_name}:")
        logging.info(f"  Precision: {final_report['precision'][i]:.4f}")
        logging.info(f"  Recall: {final_report['recall'][i]:.4f}")
        logging.info(f"  F1-score: {final_report['f1'][i]:.4f}")
        logging.info(f"  F3-score: {final_report['f3'][i]:.4f}")
        logging.info(f"  F5-score: {final_report['f5'][i]:.4f}")
        logging.info(f"  Support: {final_report['support'][i]}")

    logging.info("\nConfusion matrix and detailed classification report have been saved.")

    # Close the TensorBoard writer
    writer.close()

    # Print out the requested information
    print("\n--- Training Time Report ---")
    print(f"1) SimCLR Training Time: {total_simclr_time / 60:.2f} minutes")
    print(f"2) Linear Evaluation Time: {linear_eval_time / 60:.2f} minutes")
    print(f"3) Total Training Time: {(total_simclr_time + linear_eval_time) / 60:.2f} minutes")
    print("\n4) Data Augmentation Time by Type:")
    for aug_type, data in results.items():
        print(f"   {aug_type}: {data['aug_time'] / 60:.2f} minutes")
    print(f"\n5) Hardware Used: {get_hardware_info()}")
    print(f"6) Batch Size: SimCLR - {simclr_batch_size}, Linear Evaluation - {linear_eval_batch_size}")
    print(f"7) Number of Epochs: SimCLR - {simclr_epochs}, Linear Evaluation - {linear_eval_epochs}")
    print(f"8) Learning Rate: {args.lr}")
    print(f"9) Optimizer: {type(optimizer).__name__}")

    # Save the above information to a text file
    with open(os.path.join(args.save_dir, f'training_report_{current_datetime}.txt'), 'w') as f:
        f.write("--- Training Time Report ---\n")
        f.write(f"1) SimCLR Training Time: {total_simclr_time / 60:.2f} minutes\n")
        f.write(f"2) Linear Evaluation Time: {linear_eval_time / 60:.2f} minutes\n")
        f.write(f"3) Total Training Time: {(total_simclr_time + linear_eval_time) / 60:.2f} minutes\n")
        f.write("\n4) Data Augmentation Time by Type:\n")
        for aug_type, data in results.items():
            f.write(f"   {aug_type}: {data['aug_time'] / 60:.2f} minutes\n")
        f.write(f"\n5) Hardware Used: {get_hardware_info()}\n")
        f.write(f"6) Batch Size: SimCLR - {simclr_batch_size}, Linear Evaluation - {linear_eval_batch_size}\n")
        f.write(f"7) Number of Epochs: SimCLR - {simclr_epochs}, Linear Evaluation - {linear_eval_epochs}\n")
        f.write(f"8) Learning Rate: {args.lr}\n")
        f.write(f"9) Optimizer: {type(optimizer).__name__}\n")

    logging.info("Ablation study, model ranking, linear evaluation, and final evaluation completed.")
    logging.info("SimCLR training process completed")

if __name__ == '__main__':
    main()