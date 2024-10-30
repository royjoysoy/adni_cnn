import logging
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import torch
import torch.optim as optim
import torchio as tio
from torch.optim.lr_scheduler import CosineAnnealingLR # model 2 update 1
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from sklearn.model_selection import train_test_split

from model_m3 import SimCLR3DCNN # model 2 update 2
from data_loading_m3 import get_data_loaders, get_mr_transforms
from training_m3 import run_ablation_study, linear_eval, final_evaluation
from utils_m3 import TrainingConfig, save_model, plot_learning_curves, get_hardware_info
from training_m3 import linear_eval_all_augs 

from fine_tunning_m3 import fine_tune, load_fine_tuned_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    config = TrainingConfig.get_default_config()
    logging.info("Starting the SimCLR training process from scratch")

    # Create save directory if it doesn't exist
    os.makedirs(config.save_dir, exist_ok=True)

    current_datetime = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H-%M-%S")

    # Add file handler
    file_handler = logging.FileHandler(os.path.join(config.save_dir, f'output_{current_datetime}.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    # Redirect stdout and stderr to the log file
    sys.stdout = open(os.path.join(config.save_dir, f'output_{current_datetime}.log'), 'w')
    sys.stderr = sys.stdout

    feature_dim, temperature, k = config.feature_dim, config.temperature, config.k
    simclr_batch_size = config.simclr_batch_size 
    linear_eval_batch_size = config.linear_eval_batch_size 
    simclr_epochs = config.simclr_epochs 
    linear_eval_epochs = config.linear_eval_epochs
    num_classes = 3
    scaler = GradScaler()

    logging.info(f"SimCLR parameters: feature_dim={feature_dim}, temperature={temperature}")

    logging.info(f"Loading data from {config.data_path}")
    df = pd.read_csv(config.data_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    logging.info(f"Data split completed. Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    # Set up transforms
    logging.info("Setting up data transforms")

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
    net = SimCLR3DCNN(feature_dim=feature_dim, num_classes=3).to(device) # model 2 update 3
    optimizer = optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    logging.info("Model, optimizer, and learning rate scheduler initialized")

    # Initialize TensorBoard writer
    writer = SummaryWriter(os.path.join(config.save_dir, f'tensorboard_logs_{current_datetime}'))
    logging.info("TensorBoard writer initialized")

    # Run ablation study with time tracking
    logging.info(f"Starting ablation study for {simclr_epochs} epochs")
    simclr_start_time = time.time()
    # model 2 update 4
    results, best_models, total_simclr_time = run_ablation_study(
        net=net,
        train_loaders=train_loaders,
        val_loader=val_loader,
        optimizer=optimizer,
        simclr_epochs=config.simclr_epochs,
        temperature=config.temperature,
        simclr_batch_size=config.simclr_batch_size,
        k=config.k,
        save_top_n=5,
        writer=writer,
        scaler=scaler,
        warmup_epochs=config.warmup_epochs
    )
    simclr_end_time = time.time()
    simclr_time = simclr_end_time - simclr_start_time

    # Log ablation study results
    logging.info("Ablation study results:")
    with open(os.path.join(config.save_dir, f'ablation_results_{current_datetime}.txt'), 'w') as f:
        for aug_type, data in results.items():
            avg_loss = sum(data['train_loss']) / len(data['train_loss'])
            avg_acc = sum(data['val_accuracy']) / len(data['val_accuracy'])
            result_line = f"{aug_type}: Avg Loss: {avg_loss:.4f}, Avg Val Accuracy: {avg_acc:.2f}%"
            logging.info(result_line)
            f.write(result_line + '\n')
    
    logging.info(f"SimCLR training completed. Final SimCLR epoch: {simclr_epochs}")

    # Save the final model
    model_path = os.path.join(config.save_dir, f'latest_model_{current_datetime}.pth')
    save_model(net, optimizer, simclr_epochs, model_path)
    logging.info(f"Final model saved to {model_path}")

    # Save the model path
    with open(os.path.join(config.save_dir, f'latest_model_path_{current_datetime}.txt'), 'w') as f:
        f.write(model_path)
    logging.info(f"Latest model path saved to {os.path.join(config.save_dir, f'latest_model_path_{current_datetime}.txt')}")

    # Generate and save learning curves
    plot_learning_curves(results, config.save_dir, current_datetime)
    logging.info("Learning curves generated and saved")

    # Save and rank the top models
    for i, model_info in enumerate(best_models):
        model_path = os.path.join(config.save_dir, f'model_rank_{i+1}_{current_datetime}.pth')
        torch.save(model_info['state_dict'], model_path)
        logging.info(f"Rank {i+1}: SimCLR Epoch {model_info['simclr_epoch']}, "
                     f"Accuracy: {model_info['accuracy']:.2f}%, Saved as: {os.path.basename(model_path)}")

    # Add the new logging statements here, before linear evaluation
    logging.info("Starting Linear Evaluation phase...")
    logging.info(f"Number of best models available: {len(best_models)}")
    if best_models:
        logging.info(f"Best validation accuracy from ablation study: {best_models[0]['accuracy']:.2f}%")

    # Linear evaluation on the best model
    logging.info("Starting Linear Evaluation on all augmentation types")
    if best_models:
        net.load_state_dict(best_models[0]['state_dict'])
        logging.info("Loaded best model from ablation study")
    else:
        logging.warning("No best models found. Using current model state for linear evaluation.")

    net = net.to(device)
    net.eval()
    logging.info("Model prepared for linear evaluation")

    # Add the debug lines here:
    logging.info("\nDebug: Checking data loader structures")
    for aug_type, loader in train_loaders.items():
        batch = next(iter(loader))
        logging.info(f"Augmentation {aug_type} - Batch structure: {[x.shape if torch.is_tensor(x) else type(x) for x in batch]}")
        logging.info(f"Number of elements in batch: {len(batch)}")
    
    batch = next(iter(val_loader))
    logging.info(f"Validation loader - Batch structure: {[x.shape if torch.is_tensor(x) else type(x) for x in batch]}")
    logging.info(f"Number of elements in validation batch: {len(batch)}")
    logging.info("Debug: Data loader check complete\n")

    # Start linear evaluation
    linear_eval_start_time = time.time()
    linear_eval_results = linear_eval_all_augs(
        net=net,
        train_loaders=train_loaders,
        val_loader=val_loader,
        config=config,
        writer=writer
    )
    linear_eval_end_time = time.time()
    linear_eval_time = linear_eval_end_time - linear_eval_start_time

    # Log results for each augmentation type
    logging.info("\nLinear Evaluation Results for all augmentation types:")
    best_aug_acc = 0
    best_aug_type = None
    
    for aug_type, result in linear_eval_results.items():
        logging.info(f"\nResults for {aug_type}:")
        logging.info(f"Best Accuracy: {result['best_accuracy']:.2f}%")
        logging.info(f"Best Epoch: {result['best_epoch'] + 1}")
        logging.info(f"Evaluation Time: {result['eval_time'] / 60:.2f} minutes")
        
        if result['best_accuracy'] > best_aug_acc:
            best_aug_acc = result['best_accuracy']
            best_aug_type = aug_type

    logging.info(f"\nBest performing augmentation: {best_aug_type} with accuracy: {best_aug_acc:.2f}%")

    # Start fine-tuning phase
    logging.info("\nStarting fine-tuning phase...")
    # Load the best model from linear evaluation for fine-tuning
    best_linear_model_path = os.path.join(config.save_dir, f'best_linear_eval_model_{best_aug_type}.pth')
    if os.path.exists(best_linear_model_path):
        checkpoint = torch.load(best_linear_model_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded best linear evaluation model from {best_aug_type} augmentation for fine-tuning")
    else:
        logging.warning("No best linear evaluation model found. Using current model state for fine-tuning")

    net = net.to(device)
    
    # Make sure the classifier is added before fine-tuning
    net.add_classifier()
    
    logging.info(f"Using '{config.ft_unfreeze_strategy}' unfreezing strategy for fine-tuning")
    
    fine_tuning_start_time = time.time()
    best_ft_acc, best_ft_epoch = fine_tune(
        net=net,
        train_loader=train_loaders['base'],
        val_loader=val_loader,
        config=config,
        writer=writer,
        save_dir=config.save_dir,
        unfreeze_strategy=config.ft_unfreeze_strategy
    )
    fine_tuning_time = time.time() - fine_tuning_start_time
    
    logging.info(f"\nFine-tuning completed:")
    logging.info(f"Best validation accuracy: {best_ft_acc:.2f}%")
    logging.info(f"Best epoch: {best_ft_epoch}")
    logging.info(f"Fine-tuning time: {fine_tuning_time / 60:.2f} minutes")
    
    # Load the best fine-tuned model for final evaluation
    net, _ = load_fine_tuned_model(net, os.path.join(config.save_dir, 'best_fine_tuned_model.pth'))
    
    # Final evaluation with the fine-tuned model
    logging.info("\nStarting final evaluation with fine-tuned model...")
    final_report, final_accuracy = final_evaluation(net, test_loader, device, config.save_dir)

    # Print final results
    logging.info(f"Total SimCLR epochs: {simclr_epochs}")
    logging.info(f"Total Linear Evaluation epochs: {linear_eval_epochs}")
    logging.info(f"Total Fine-tuning epochs: {config.ft_epochs}")
    logging.info("\nFinal Evaluation Results:")
    logging.info(f"Accuracy: {final_report['accuracy']:.4f}")

    # Log F1-scores
    macro_f1 = final_report.get('macro avg', {}).get('f1-score', 'N/A')
    weighted_f1 = final_report.get('weighted avg', {}).get('f1-score', 'N/A')
    logging.info(f"Macro Avg F1-score: {macro_f1:.4f}" if isinstance(macro_f1, float) else f"Macro Avg F1-score: {macro_f1}")
    logging.info(f"Weighted Avg F1-score: {weighted_f1:.4f}" if isinstance(weighted_f1, float) else f"Weighted Avg F1-score: {weighted_f1}")

    # Save final evaluation results to a text file
    with open(os.path.join(config.save_dir, f'final_evaluation_results_{current_datetime}.txt'), 'w') as f:
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
    print(f"3) Fine-tuning Time: {fine_tuning_time / 60:.2f} minutes")
    print(f"4) Total Training Time: {(total_simclr_time + linear_eval_time + fine_tuning_time) / 60:.2f} minutes")
    print("\n5) Data Augmentation Time by Type:")
    for aug_type, data in results.items():
        print(f"   {aug_type}: {data['aug_time'] / 60:.2f} minutes")
    print(f"\n6) Hardware Used: {get_hardware_info()}")
    print(f"7) Batch Size: SimCLR - {simclr_batch_size}, Linear Evaluation - {linear_eval_batch_size}, Fine-tuning - {simclr_batch_size}")
    print(f"8) Number of Epochs: SimCLR - {simclr_epochs}, Linear Evaluation - {linear_eval_epochs}, Fine-tuning - {config.ft_epochs}")
    print(f"9) Learning Rate: SimCLR - {config.lr}, Fine-tuning encoder - {config.ft_encoder_lr}, Fine-tuning classifier - {config.ft_classifier_lr}")
    print(f"10) Optimizer: {type(optimizer).__name__}")

    # Save the training report
    with open(os.path.join(config.save_dir, f'training_report_{current_datetime}.txt'), 'w') as f:
        f.write("--- Training Time Report ---\n")
        f.write(f"1) SimCLR Training Time: {total_simclr_time / 60:.2f} minutes\n")
        f.write(f"2) Linear Evaluation Time: {linear_eval_time / 60:.2f} minutes\n")
        f.write(f"3) Fine-tuning Time: {fine_tuning_time / 60:.2f} minutes\n")
        f.write(f"4) Total Training Time: {(total_simclr_time + linear_eval_time + fine_tuning_time) / 60:.2f} minutes\n")
        f.write("\n5) Data Augmentation Time by Type:\n")
        for aug_type, data in results.items():
            f.write(f"   {aug_type}: {data['aug_time'] / 60:.2f} minutes\n")
        f.write(f"\n6) Hardware Used: {get_hardware_info()}\n")
        f.write(f"7) Batch Size: SimCLR - {simclr_batch_size}, Linear Evaluation - {linear_eval_batch_size}\n")
        f.write(f"8) Number of Epochs: SimCLR - {simclr_epochs}, Linear Evaluation - {linear_eval_epochs}, Fine-tuning - {config.ft_epochs}\n")
        f.write(f"9) Learning Rate: SimCLR - {config.lr}, Fine-tuning encoder - {config.ft_encoder_lr}, Fine-tuning classifier - {config.ft_classifier_lr}\n")
        f.write(f"10) Optimizer: {type(optimizer).__name__}\n")

    logging.info("Ablation study, model ranking, linear evaluation, fine-tuning, and final evaluation completed.")
    logging.info("SimCLR training process completed")

if __name__ == '__main__':
    main()
