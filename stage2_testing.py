import argparse
import os
import time
import logging
import datetime
import json
import random
from multiprocessing import cpu_count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR  
from collections import Counter

 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm  
from model import ECL
from loss import ECLoss
from data.dataset import CustomImageDataset
from lib.train_util import AverageMeter
from data.transform import get_transforms, TwoCropTransform
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
import numpy as np

torch.autograd.set_detect_anomaly(True)

 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def args_func():
    """Parses command line arguments for training."""
    parser = argparse.ArgumentParser('argument for enhanced contrastive learner training')

     
    parser.add_argument('--epochs', type=int, default=7, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=48, help='batch_size PER GPU (actual processed batch is 2*batch_size due to TwoCrop)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
     
    parser.add_argument('--temp', type=float, default=0.07, help='temperature for loss function')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help='Optimizer to use (Adam or SGD)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR', choices=['CosineAnnealingLR', 'None'], help='Learning rate scheduler')


     
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='Number of steps to accumulate gradients over. Simulates batch_size * grad_accum_steps.')
    parser.add_argument('--use_amp', action='store_true', help='Use Automatic Mixed Precision (AMP) for training')
    parser.add_argument('--init_scale', type=float, default=65536.0,
                        help='Initial scale factor for gradient scaling in AMP')

     
    parser.add_argument('--data_folder', type=str, default="/home/teaching/deepfake/UADFV_frames_new", help='path to DF dataset')
    parser.add_argument('--pseudo_label_file', type=str, default="/home/teaching/deepfake/project/files/Unsupervised_DF_Detection/output_gpu/UADFV_real_labels_new/image_pseudo_labels_gpu_70purity_video_level.json", help='path to pseudo_label.json')
    parser.add_argument('--image_size', type=int, default=299, help='parameter for image transforms (e.g., RandomResizedCrop)')
    parser.add_argument('--backbone', default='xception', choices=['xception', 'swinv2', 'convnext_base'], help='Backbone architecture to use')

     
    parser.add_argument('--select_confidence_epoch', type=int, default=2, help='epoch frequency to select confidence samples (0 to disable)')
    parser.add_argument('--k', type=float, default=0.6, help='select top k confidence samples proportion (0.0 to 1.0)')
    parser.add_argument('--confidence_batch_size', type=int, default=256, help='Batch size for confidence sample selection (can be larger than training batch size)')
    parser.add_argument('--current_epoch', type=int, default=0, help='current epoch for resuming training')  

     
    default_workers = max(1, min(cpu_count() // 2, 8))
    parser.add_argument('--num_workers', type=int, default=default_workers, help='Number of CPU processes for data loading.')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency (in batches)')
    parser.add_argument('--tsne_freq', type=int, default=1, help='frequency (in epochs) to run t-SNE visualization (0 to disable)')
    parser.add_argument('--checkpoint_freq', type=int, default=2, help='frequency (in epochs) to save checkpoints (0 to disable)')

    parser.add_argument('--output_dir', type=str, default='/home/teaching/deepfake/project/files/Unsupervised_DF_Detection', help='Directory to save checkpoints and logs')
    parser.add_argument('--plot_dir', type=str, default='/home/teaching/deepfake/project/files/Unsupervised_DF_Detection/plots', help='Directory to save purity plot')

     
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume training from (e.g., ./save/SupCon/.../ckpt_epoch_10.pth)')


    args = parser.parse_args()



    logging.info(f"Using {args.num_workers} workers for data loading.")
    if args.grad_accum_steps > 1:
        logging.info(f"Using gradient accumulation with {args.grad_accum_steps} steps. Effective batch size: {args.batch_size * args.grad_accum_steps * 2}")
    if args.use_amp:
        logging.info("Using Automatic Mixed Precision (AMP).")
    if args.resume:
        logging.info(f"Attempting to resume training from checkpoint: {args.resume}")
    return args


def set_loader(pseudo_label_dict, args, is_confidence_selection=False):
    """Creates DataLoader instances with appropriate error handling and collate function."""
    if not isinstance(pseudo_label_dict, dict) or not pseudo_label_dict:
        logging.error("Invalid or empty pseudo_label_dict provided to set_loader.")
        return None

    # Use different transforms and settings based on the purpose of the loader
    if is_confidence_selection:
        # Use a fixed-size transform suitable for feature extraction/evaluation
        transform = get_transforms(name="val", size=args.image_size, backbone=args.backbone)
        dataset = CustomImageDataset(pseudo_label_dict, args.data_folder, transform)
        shuffle = False  # No need to shuffle for feature extraction
        # Use a potentially different batch size for confidence selection if specified
        current_batch_size = getattr(args, 'confidence_batch_size', args.batch_size)
        # Adjust workers, potentially more for faster loading during evaluation
        adjusted_workers = min(args.num_workers * 2, 16) if torch.cuda.is_available() else args.num_workers
        persistent_workers = False  # Not typically needed for a single pass
        drop_last = False  # Keep all samples for clustering
        logging.info(f"DataLoader for Confidence Selection: Batch Size={current_batch_size}, Workers={adjusted_workers}")
        logging.info(f"Using fixed-size transform (Resize + CenterCrop to {args.image_size}x{args.image_size}) for confidence selection.")
    else:
        # Use training transforms (like TwoCropTransform) and settings
        train_transform = get_transforms(name="train", size=args.image_size, backbone=args.backbone)
        dataset = CustomImageDataset(pseudo_label_dict, args.data_folder, TwoCropTransform(train_transform))
        shuffle = True  # Shuffle training data
        current_batch_size = args.batch_size
        # Adjust workers for training, potentially fewer to balance with GPU work
        adjusted_workers = min(args.num_workers, 8 if torch.cuda.is_available() else 4)
        persistent_workers = True if adjusted_workers > 0 else False  # Keep workers alive between epochs if possible
        drop_last = True  # Drop the last incomplete batch in training
        logging.info(f"DataLoader for Training: Batch Size={current_batch_size}, Workers={adjusted_workers}")
        logging.info("Using TwoCropTransform for training.")

    # Prefetch factor calculation
    prefetch_factor = 2 if adjusted_workers > 0 else None

    # Create and return the DataLoader
    try:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=current_batch_size,
            shuffle=shuffle,
            num_workers=adjusted_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=custom_collate_fn, # Assuming custom_collate_fn is defined
            drop_last=drop_last
        )
        logging.info(f"Created DataLoader with {len(dataset)} samples. Batch Size={current_batch_size}, "
                    f"Shuffle={shuffle}, Workers={adjusted_workers}, PinMemory={data_loader.pin_memory}, "
                    f"PersistentWorkers={persistent_workers}, DropLast={drop_last}")
        return data_loader
    except Exception as e:
        logging.error(f"Failed to create DataLoader: {e}")
        return None


# Helper function to calculate clustering purity based on filename heuristic
def calculate_purity_from_filenames(image_names, cluster_labels):
    """
    Calculates the purity of clustering results by comparing cluster labels
    to true labels derived from image filenames ('fake' in name -> 0, else -> 1).

    Args:
        image_names (list): List of image filenames corresponding to the data points.
        cluster_labels (list or np.array): The cluster assignments from the clustering algorithm.

    Returns:
        float: The purity score (between 0 and 1) based on the filename heuristic.
    """
    if len(image_names) != len(cluster_labels) or len(image_names) == 0:
        logging.warning("Cannot calculate purity: Mismatch in lengths or empty data.")
        return 0.0

    # Derive true labels from filenames
    filename_true_labels = []
    for name in image_names:
        # Assuming 'fake' in filename indicates true label 0, otherwise 1
        true_label = 0 if 'fake' in name.lower() else 1
        filename_true_labels.append(true_label)

    total_samples = len(filename_true_labels)
    correct_assignments = 0
    unique_clusters = np.unique(cluster_labels)

    for cluster_id in unique_clusters:
        # Get indices of samples belonging to this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue

        # Get the filename-derived true labels for these samples
        true_labels_in_cluster = [filename_true_labels[i] for i in cluster_indices]

        # Find the most frequent true label in this cluster
        if true_labels_in_cluster:
            majority_label, count = Counter(true_labels_in_cluster).most_common(1)[0]
            # Add the number of samples correctly assigned to this cluster's majority class
            correct_assignments += count

    purity = correct_assignments / total_samples
    return purity


def set_model(args):
    """Initializes the model and criterion."""
    try:
         
         
        model = ECL(backbone_type=args.backbone) 
        logging.info(f"Initialized ECL model - {args.backbone}")  
    except Exception as e:
        logging.error(f"Failed to initialize ECL model: {e}")
        raise  

    criterion = ECLoss(temperature=args.temp)

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logging.info(f"CUDA available. Found {gpu_count} GPU(s).")
        if gpu_count > 1:
            logging.info("Using DataParallel for multi-GPU.")
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True  
        logging.info("Model and criterion moved to GPU. cuDNN benchmark enabled.")
    else:
        logging.warning("CUDA not available. Training on CPU will be very slow.")

    return model, criterion

def set_optimizer(args, model):
    """Sets up the optimizer based on arguments."""
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)
        logging.info("Using Adam optimizer.")
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
        logging.info(f"Using SGD optimizer with momentum {args.momentum}.")
    else:
        logging.error(f"Unknown optimizer: {args.optimizer}")
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    return optimizer

def set_scheduler(args, optimizer, epochs, train_loader):
    """Sets up the learning rate scheduler."""
    if args.scheduler == 'CosineAnnealingLR':
         
         
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
        logging.info(f"Using CosineAnnealingLR scheduler with T_max={epochs}.")
    elif args.scheduler == 'None':
        scheduler = None
        logging.info("No learning rate scheduler used.")
    else:
        logging.error(f"Unknown scheduler: {args.scheduler}")
        raise ValueError(f"Unknown scheduler: {args.scheduler}")
    return scheduler



def train(train_loader, model, criterion, optimizer, epoch, args, scaler, scheduler=None):
    """Runs one training epoch with optional gradient accumulation and AMP."""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    num_batches = len(train_loader)
    if num_batches == 0:
        logging.error(f"Train loader for epoch {epoch} has zero length. Cannot train.")
        return None

    effective_steps_per_epoch = num_batches // args.grad_accum_steps
    if num_batches % args.grad_accum_steps != 0:
        effective_steps_per_epoch += 1
    start_time = time.time()

    for idx, batch_data in enumerate(train_loader):

        data_time.update(time.time() - end)

        if batch_data is None:
            logging.warning(f"Skipping None batch returned by collate_fn at index {idx} in epoch {epoch}.")
            end = time.time()
            continue

        try:
            images, labels, _ = batch_data
        except ValueError as e:
            logging.error(f"Error unpacking batch data at index {idx}: {e}. Batch data structure: {type(batch_data)}")
            end = time.time()
            continue

        images_cat = None
        if isinstance(images, list) and len(images) == 2:
            try:
                images_cat = torch.cat([images[0], images[1]], dim=0)
            except Exception as e:
                logging.error(f"Error concatenating image crops at batch index {idx}: {e}")
                logging.error(f"Image 0 shape: {images[0].shape if isinstance(images[0], torch.Tensor) else 'N/A'}, Image 1 shape: {images[1].shape if isinstance(images[1], torch.Tensor) else 'N/A'}")
                end = time.time()
                continue
        elif isinstance(images, torch.Tensor):
            images_cat = images
        else:
            logging.error(f"Unexpected image format in batch {idx}. Type: {type(images)}. Skipping batch.")
            end = time.time()
            continue

        bsz = labels.shape[0]

        if torch.cuda.is_available():
            try:
                images_cat = images_cat.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            except Exception as e:
                logging.error(f"Error moving batch {idx} data to CUDA: {e}")
                if "CUDA out of memory" in str(e):
                    logging.error("CUDA OOM during data transfer. Try reducing batch size.")
                end = time.time()
                continue

        features = None # Initialize features variable

        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=args.use_amp):
            try:
                output = model(images_cat)

                # Handle model output - get the actual features tensor
                if isinstance(output, tuple):
                    # Assuming the features you need are the second element (index 1)
                    # This matches your previous logic `features = output[1]`
                    if len(output) > 1 and isinstance(output[1], torch.Tensor):
                         features = output[1]
                         # Optional: check other elements if needed, e.g., output[0]
                         # if isinstance(output[0], torch.Tensor) and (torch.isnan(output[0]).any() or torch.isinf(output[0]).any()):
                         #     logging.warning(f"NaN/Inf detected in the first element of model output tuple at batch {idx}!")
                    else:
                         logging.error(f"Model output is a tuple but doesn't have expected structure at batch {idx}. Tuple length: {len(output)}. Skipping.")
                         end = time.time()
                         continue # Skip this batch

                elif isinstance(output, torch.Tensor):
                    features = output
                else:
                    logging.error(f"Unexpected model output type in training (batch {idx}): {type(output)}. Skipping.")
                    end = time.time()
                    continue # Skip this batch

                # === PLACE NaN/INF CHECK HERE, AFTER GETTING THE TENSOR ===
                if features is None or not isinstance(features, torch.Tensor) or torch.isnan(features).any() or torch.isinf(features).any():
                    print("NaN or Inf detected in features tensor after model output handling!")
                    if features is not None and isinstance(features, torch.Tensor):
                        print(f"Features shape: {features.shape}")
                        print("Features stats:", features.min(), features.max(), features.mean())
                    else:
                        print("Features variable is None or not a Tensor.")
                    # Optionally save the batch that caused this:
                    # torch.save(images_cat, f"bad_batch_input_epoch_{epoch}_idx_{idx}.pt")
                    # If you want to see the model output tuple:
                    # if isinstance(output, tuple): torch.save(output, f"bad_batch_output_epoch_{epoch}_idx_{idx}.pt")

                    # Consider just logging and skipping the batch if you don't want to stop training immediately
                    # logging.error("Features tensor contains NaN/Inf. Skipping batch.")
                    # end = time.time()
                    # continue # Skip this batch
                    # OR raise the error to stop:
                    raise RuntimeError("Features tensor contains NaN/Inf.")
                # ==========================================================


                if not features.requires_grad:
                    # This warning might indicate an issue with model configuration (e.g., if you want grad)
                    logging.warning(f"Features do not require grad at batch {idx}. Check model definition and if model is in train() mode.")


            except Exception as e:
                logging.error(f"Error during model forward pass at batch {idx}: {e}")
                if torch.cuda.is_available() and "CUDA out of memory" in str(e):
                    logging.error("CUDA OOM during forward pass. Try reducing batch size or enabling AMP (--use_amp).")
                end = time.time()
                continue # Skip this batch


            # --- Logic to prepare features_for_loss from the 'features' tensor ---
            features_for_loss = None
            if isinstance(images, list) and len(images) == 2:
                try:
                    # This block expects 'features' to contain concatenated features from two crops
                    if features.shape[0] != 2 * bsz:
                         logging.error(f"Feature shape mismatch for splitting at batch {idx}: Expected {2 * bsz}, got {features.shape[0]}. Skipping.")
                         end = time.time()
                         continue

                    f1, f2 = torch.split(features, [bsz, bsz], dim=0)

                    features_for_loss = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    expected_feature_dim = f1.shape[1]

                    if features_for_loss.shape != (bsz, 2, expected_feature_dim):
                         logging.error(f"Final features_for_loss shape mismatch at batch {idx}: Expected ({bsz}, 2, {expected_feature_dim}), got {features_for_loss.shape}. Skipping.")
                         end = time.time()
                         continue

                except Exception as e:
                    logging.error(f"Error splitting/reshaping features at batch {idx}: {e}")
                    logging.error(f"Features shape: {features.shape if isinstance(features, torch.Tensor) else 'N/A'}, bsz: {bsz}")
                    end = time.time()
                    continue

            elif isinstance(features, torch.Tensor):
                # This block handles cases where the loader yields single images/features
                # If your loader *always* yields two crops, this case might indicate an issue
                features_for_loss = features.unsqueeze(1) # Add a view dimension
                # logging.warning("Training loader yielded single features, not two crops. Check transform and model output for contrastive loss.")
            else:
                 # This case should ideally be caught by the check after model output handling
                 logging.error(f"Unexpected features format for loss calculation at batch {idx}: {type(features)}. Skipping.")
                 end = time.time()
                 continue


            # === Final check on features_for_loss before criterion ===
            if features_for_loss is None or not isinstance(features_for_loss, torch.Tensor) or features_for_loss.shape[0] == 0:
                 logging.warning(f"Invalid features_for_loss tensor before criterion at batch {idx}. Skipping loss calculation and backward pass.")
                 end = time.time()
                 continue

            # Check for NaN/Inf one last time on the input to the loss
            if torch.isnan(features_for_loss).any() or torch.isinf(features_for_loss).any():
                print("NaN or Inf detected in final features_for_loss tensor before criterion!")
                print(f"features_for_loss shape: {features_for_loss.shape}")
                print("features_for_loss stats:", features_for_loss.min(), features_for_loss.max(), features_for_loss.mean())

                raise RuntimeError("features_for_loss tensor contains NaN/Inf before criterion.")


            loss = None
            try:

                loss = criterion(features_for_loss, labels)

                # === Add check after loss calculation ===
                if loss is not None and (torch.isnan(loss).any() or torch.isinf(loss).any()):
                    print("NaN or Inf detected in the calculated loss value!")
                    print(f"Loss value: {loss.item() if isinstance(loss, torch.Tensor) else 'N/A'}")
                    raise RuntimeError("Calculated loss value is NaN/Inf.")
                # ======================================


            except Exception as e:
                logging.error(f"Error calculating loss at batch {idx}: {e}")
                logging.error(f"Features for loss shape: {features_for_loss.shape if isinstance(features_for_loss, torch.Tensor) else 'N/A'}, Labels shape: {labels.shape if isinstance(labels, torch.Tensor) else 'N/A'}")
                end = time.time()
                continue # Skip this batch


            # Gradient accumulation scaling
            if args.grad_accum_steps > 1:
                 if loss is not None and torch.isfinite(loss): # Use isfinite to check for NaN or Inf
                      loss = loss / args.grad_accum_steps
                 else:
                      # This case should ideally be caught by the loss check above
                      logging.warning(f"Invalid loss detected before gradient accumulation division at batch {idx}. Skipping backward pass.")
                      end = time.time()
                      continue

        # Backward pass and optimizer step (outside autocast but inside the loop)
        if loss is not None and torch.isfinite(loss) and loss.requires_grad:
            try:
                # Backpropagate the loss
                scaler.scale(loss).backward()

            except RuntimeError as e:
                if "element 0 of tensors does not require grad" in str(e):
                    logging.error(f"Gradient computation error at batch {idx}. Skipping batch.")
                    optimizer.zero_grad()
                    end = time.time()
                    continue
                logging.error(f"RuntimeError during backward pass at batch {idx}: {e}")
                raise e # Re-raise the error after logging

            except Exception as e:
                logging.error(f"Error during backward pass at batch {idx}: {e}")
                if torch.cuda.is_available() and "CUDA out of memory" in str(e):
                    logging.error("CUDA OOM during backward pass. Try reducing batch size, enabling AMP (--use_amp), or increasing gradient accumulation steps (--grad_accum_steps).")
                optimizer.zero_grad()
                end = time.time()
                continue # Skip this batch

        elif loss is not None and torch.isfinite(loss) and not loss.requires_grad:
            logging.warning(f"Loss does not require grad at batch {idx}. Skipping backward pass. Check model output and criterion.")
            end = time.time()
            continue
        else:
            # This case should be caught by checks above, but included for safety
            logging.warning(f"Skipping backward pass for batch {idx} due to invalid loss value (None, NaN, or Inf).")
            end = time.time()
            continue

        # Optimizer step (every args.grad_accum_steps batches)
        if (idx + 1) % args.grad_accum_steps == 0:
            try:
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Keep clipping

                # Optimizer step
                scaler.step(optimizer)
                # Update the scaler for the next iteration
                scaler.update()
                # Zero gradients
                optimizer.zero_grad()

            except RuntimeError as e:
                if "No inf checks were recorded" in str(e):
                    logging.warning("GradScaler error (No inf checks). Recreating scaler.")
                    # This can sometimes happen, recreating scaler helps
                    scaler = GradScaler(enabled=args.use_amp, init_scale=args.init_scale) # Assuming args.init_scale exists
                    optimizer.zero_grad()
                    end = time.time()
                    continue
                logging.error(f"RuntimeError during optimizer step at batch index {idx}: {e}")
                raise e # Re-raise after logging

            except Exception as e:
                 logging.error(f"Error during optimizer step at batch index {idx} (effective step {(idx + 1) // args.grad_accum_steps}): {e}")
                 # Decide if you want to break or continue. Breaking stops the epoch.
                 # break # Option to stop the epoch on optimizer error
                 # Or just log and skip the rest of optimizer/scheduler steps for this iteration
                 optimizer.zero_grad()
                 end = time.time()
                 continue # Skip to the next batch

        # Update loss meter with the *unscaled* loss for correct reporting
        if loss is not None and torch.isfinite(loss):
             # If using gradient accumulation, add the scaled loss, losses.avg will be correct after aggregation
             # losses.update(loss.item(), bsz) # Use this if updating meter with scaled loss
             # Or update with unscaled loss value if you want the average loss per effective step
             # Need to store the unscaled loss value before division for grad accum
             # Let's update the meter with the scaled loss as it's what was backpropped
             losses.update(loss.item(), bsz)


        batch_time.update(time.time() - end)
        end = time.time()

        # Log training progress (log based on effective steps)
        if (idx + 1) % args.grad_accum_steps == 0 or (idx + 1) == num_batches: # Log on last batch too
             current_effective_step = (idx + 1) // args.grad_accum_steps
             # Handle case where last batch doesn't complete accumulation steps
             if (idx + 1) % args.grad_accum_steps != 0 and (idx + 1) == num_batches:
                 current_effective_step += 1

             total_effective_steps = max(1, num_batches // args.grad_accum_steps)
             if num_batches % args.grad_accum_steps != 0:
                  total_effective_steps += 1

             current_lr = optimizer.param_groups[0]['lr']
             logging.info(f'Train: [{epoch}][{current_effective_step}/{total_effective_steps}]\t'
                         f'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         f'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         f'loss {losses.val:.3f} ({losses.avg:.3f})\t'
                         f'LR {current_lr:.6f}')


    # Perform final optimizer step for remaining gradients if batch size is not divisible by grad_accum_steps
    if num_batches > 0 and num_batches % args.grad_accum_steps != 0:
        logging.info(f"Epoch {epoch}: Performing final optimizer step for remaining gradients.")
        try:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        except Exception as e:
            logging.error(f"Error during final optimizer step for epoch {epoch}: {e}")

    # Scheduler step after the epoch
    if scheduler is not None:
        scheduler.step()
        logging.info(f"Scheduler stepped. New learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    epoch_time = time.time() - start_time
    logging.info(f"Epoch {epoch} training completed in {epoch_time:.2f} seconds. Average Loss: {losses.avg:.4f}")

    if losses.count == 0:
        logging.warning(f"Epoch {epoch} finished with no valid batches processed. Average loss is undefined.")
        return None
    return losses.avg



def save_purity_plot(purity_history, plot_dir, filename="purity_vs_epoch.png"):
    """
    Generates and saves a plot of K-Means purity vs. epoch.

    Args:
        purity_history (list): A list of tuples, where each tuple is (epoch, purity_score).
        plot_dir (str): Directory to save the plot.
        filename (str): Name of the plot file.
    """
    if not purity_history:
        logging.info("No purity data collected to plot.")
        return

    epochs = [item[0] for item in purity_history]
    purities = [item[1] for item in purity_history]

    # Ensure plot directory exists
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, filename)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, purities, marker='o', linestyle='-')
    plt.title('K-Means Purity vs. Epoch (Filename Heuristic)')
    plt.xlabel('Epoch')
    plt.ylabel('Purity Score')
    plt.grid(True)
    plt.ylim(0, 1.1) # Purity is between 0 and 1
    # Use all recorded epochs as ticks
    plt.xticks(epochs)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True)) # Ensure integer ticks if possible


    try:
        plt.savefig(plot_path)
        logging.info(f"Purity plot saved to {plot_path}")
    except Exception as e:
        logging.error(f"Error saving purity plot to {plot_path}: {e}")

    plt.close() # Close the plot figure


def select_confidence_sample(model, dataloader, args):
    """
    Selects confident samples based on clustering with enhanced logging and purity calculation.
    Assumes the dataloader yields batches including image_names.
    Calculates purity based on filename heuristic ('fake' in name -> 0, else -> 1).

    Returns:
        tuple: (result_dict, purity_score). result_dict contains selected image_names
               and their pseudo-labels ({image_name: cluster_id_string}).
               purity_score is the calculated purity (compared to filename heuristic)
               or None if purity could not be calculated.
    """
    logging.info("Starting confidence sample selection with K-Means...")
    start_time = time.time()
    model.eval()

    if dataloader is None:
        logging.error("Dataloader provided to select_confidence_sample is None.")
        return {}, None # Return empty dict and None purity

    # --- Feature Extraction ---
    is_data_parallel = isinstance(model, torch.nn.DataParallel)
    eval_model = model.module if is_data_parallel else model

    all_features = []
    all_data_names = []

    logging.info("Extracting features...")
    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
             if batch_data is None:
                 logging.warning(f"Skipping None batch returned by collate_fn at index {i} during confidence selection.")
                 continue

             try:
                 # Assuming batch_data is (images, ..., image_names) - adjust if your loader is different
                 # We only need images and names for feature extraction here
                 img, _, image_names = batch_data

                 if not isinstance(img, torch.Tensor) or img.ndim != 4:
                     logging.warning(f"Invalid image tensor received in batch {i} during feature extraction. Shape: {img.shape if isinstance(img, torch.Tensor) else type(img)}. Skipping.")
                     continue
                 if not isinstance(image_names, (list, tuple)):
                      logging.warning(f"Invalid image_names type received in batch {i}: {type(image_names)}. Skipping.")
                      continue
                 if img.shape[0] != len(image_names):
                      logging.warning(f"Mismatch between image tensor size ({img.shape[0]}) and image names list size ({len(image_names)}) in batch {i}. Skipping.")
                      continue

             except (ValueError, TypeError) as e:
                  logging.error(f"Error unpacking batch data at index {i} during feature extraction: {e}")
                  continue

             if torch.cuda.is_available():
                 try:
                     img = img.cuda(non_blocking=True)
                 except Exception as e:
                     logging.error(f"Error moving image batch {i} to CUDA during feature extraction: {e}")
                     if "CUDA out of memory" in str(e):
                          logging.error("CUDA OOM during data transfer for feature extraction. Try reducing --confidence_batch_size.")
                     continue

             try:
                 output = eval_model(img)
                 # Adjust this based on your model's output format for features
                 if isinstance(output, tuple) and len(output) >= 2:
                       feat_contrast = output[1] # Assuming contrastive features are the second element
                 elif isinstance(output, torch.Tensor):
                      feat_contrast = output    # Assuming the output itself is the feature tensor
                 else:
                     logging.error(f"Unexpected model output type in feature extraction (batch {i}): {type(output)}. Skipping.")
                     continue

                 if not isinstance(feat_contrast, torch.Tensor):
                     logging.error(f"Model output features are not a tensor (batch {i}): {type(feat_contrast)}. Skipping.")
                     continue
                 if feat_contrast.shape[0] != img.shape[0]:
                     logging.warning(f"Mismatch between feature batch size ({feat_contrast.shape[0]}) and image batch size ({img.shape[0]}) in batch {i}. Skipping.")
                     continue

             except Exception as e:
                  logging.error(f"Error during model forward pass in feature extraction (batch {i}): {e}")
                  if torch.cuda.is_available() and "CUDA out of memory" in str(e):
                       logging.error("CUDA OOM during feature extraction forward pass. Try reducing --confidence_batch_size.")
                  continue

             try:
                  features_np = feat_contrast.detach().cpu().numpy()
             except Exception as e:
                  logging.error(f"Error moving features to CPU/NumPy in feature extraction (batch {i}): {e}")
                  continue

             if np.isnan(features_np).any() or np.isinf(features_np).any():
                  logging.warning(f"NaN or Inf detected in features for batch {i}. Skipping batch.")
                  continue

             all_features.append(features_np)
             all_data_names.extend(list(image_names))

             if (i + 1) % args.print_freq == 0:
                     logging.info(f"Feature extraction: Batch [{i+1}/{len(dataloader)}]")

    if not all_features:
         logging.error("No valid features collected. Cannot proceed with clustering.")
         return {}, None

    try:
        all_features = np.concatenate(all_features, axis=0)
        if len(all_data_names) != all_features.shape[0]:
            logging.error(f"CRITICAL: Mismatch between total features ({all_features.shape[0]}) and total image names ({len(all_data_names)}) after concatenation. Cannot proceed.")
            return {}, None
        logging.info(f"Successfully extracted {all_features.shape[0]} features for {len(all_data_names)} images. Feature shape: {all_features.shape}")
        if np.isnan(all_features).any() or np.isinf(all_features).any():
             logging.error("NaN or Inf values found in concatenated features. Clustering cannot proceed.")
             return {}, None
    except Exception as e:
         logging.error(f"Error concatenating features: {e}")
         return {}, None


    # --- K-Means Clustering ---
    logging.info(f"Running K-Means clustering (k=2) on {all_features.shape[0]} samples...")
    if all_features.shape[0] < 2:
        logging.warning(f"Not enough samples ({all_features.shape[0]}) for K-Means clustering (need at least 2). Skipping selection.")
        return {}, None

    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto' if hasattr(KMeans, 'n_init') else 10)
    try:
        kmeans.fit(all_features)
        labels = kmeans.labels_ # Cluster assignments for ALL samples
        logging.info("K-Means clustering finished.")
        logging.info(f"Cluster label counts from K-Means: {np.bincount(labels)}")
    except ValueError as e:
        logging.error(f"K-Means clustering failed: {e}. This might happen if features have NaN/Inf or are all identical.")
        logging.error(f"Features shape: {all_features.shape}, dtype: {all_features.dtype}")
        logging.error(f"Feature stats - Mean: {np.mean(all_features):.4f}, Std: {np.std(all_features):.4f}, Min: {np.min(all_features):.4f}, Max: {np.max(all_features):.4f}")
        return {}, None
    except Exception as e:
        logging.error(f"K-Means clustering failed with unexpected error: {e}")
        return {}, None

    # --- Calculate and Print Purity After K-Means (based on filename heuristic) ---
    purity_score = None # Initialize purity score
    try:
        # Calculate purity using the filename heuristic and K-Means labels
        purity_score = calculate_purity_from_filenames(all_data_names, labels)
        logging.info(f"Purity After K-Means (compared to filename heuristic): {purity_score:.4f}")
    except Exception as e:
         logging.error(f"Error calculating Purity After K-Means (filename heuristic): {e}")
         purity_score = None


    # --- Confidence Sample Selection (using cluster labels from ALL features) ---
    cluster_centers = kmeans.cluster_centers_
    try:
        # Calculate distances for ALL samples to their assigned cluster center
        all_distances = cosine_distances(all_features, cluster_centers)
        distances_to_assigned_center = all_distances[np.arange(all_features.shape[0]), labels]

    except Exception as e:
        logging.error(f"Failed to calculate distances to cluster centers for selection: {e}")
        return {}, purity_score # Return generated purity even if selection fails

    cluster_0_indices = np.where(labels == 0)[0]
    cluster_1_indices = np.where(labels == 1)[0]

    logging.info(f"--- Selection Stats ---")
    logging.info(f"Cluster 0 size (from K-Means): {len(cluster_0_indices)}")
    logging.info(f"Cluster 1 size (from K-Means): {len(cluster_1_indices)}")

    if len(cluster_0_indices) == 0 or len(cluster_1_indices) == 0:
        logging.warning("One or both clusters are empty after K-Means. Cannot select confident samples from both.")
        # Return empty dict, but include the calculated purity if available
        return {}, purity_score

    # Get distances for samples within each cluster
    distances_cluster_0 = distances_to_assigned_center[cluster_0_indices]
    distances_cluster_1 = distances_to_assigned_center[cluster_1_indices]

    if len(distances_cluster_0) > 0:
         logging.info(f"Cluster 0 distances (to center 0) - Min: {np.min(distances_cluster_0):.4f}, Max: {np.max(distances_cluster_0):.4f}, Mean: {np.mean(distances_cluster_0):.4f}")
    else:
         logging.info("Cluster 0 has no samples, cannot calculate distances.")
    if len(distances_cluster_1) > 0:
         logging.info(f"Cluster 1 distances (to center 1) - Min: {np.min(distances_cluster_1):.4f}, Max: {np.max(distances_cluster_1):.4f}, Mean: {np.mean(distances_cluster_1):.4f}")
    else:
         logging.info("Cluster 1 has no samples, cannot calculate distances.")


    # Select top K samples closest to the center in each cluster
    # base_k is likely the *proportion* of samples to select from each cluster
    base_k = args.k # Assuming args.k is a float, e.g., 0.1 for 10%

    K_0 = int(base_k * len(cluster_0_indices))
    K_1 = int(base_k * len(cluster_1_indices))
    logging.info(f"Selection target (k={args.k} proportion): Keep Top {K_0} from Cluster 0, Top {K_1} from Cluster 1 based on minimum distance.")

    top_k_cluster_0_data = []
    if K_0 > 0 and len(distances_cluster_0) > 0:
        try:
            # Sort indices by distance and select the top K_0 (smallest distances)
            sorted_indices_0_in_cluster = np.argsort(distances_cluster_0)[:K_0]
            # Map back to the original indices in all_features/all_data_names
            original_indices_0 = cluster_0_indices[sorted_indices_0_in_cluster]
            # Collect the image names and their assigned cluster label (pseudo-label)
            top_k_cluster_0_data = [(all_data_names[i], 0) for i in original_indices_0]
        except Exception as e:
            logging.error(f"Error during sorting/selection for Cluster 0: {e}")


    top_k_cluster_1_data = []
    if K_1 > 0 and len(distances_cluster_1) > 0:
       try:
            # Sort indices by distance and select the top K_1 (smallest distances)
            sorted_indices_1_in_cluster = np.argsort(distances_cluster_1)[:K_1]
            # Map back to the original indices in all_features/all_data_names
            original_indices_1 = cluster_1_indices[sorted_indices_1_in_cluster]
            # Collect the image names and their assigned cluster label (pseudo-label)
            top_k_cluster_1_data = [(all_data_names[i], 1) for i in original_indices_1]
       except Exception as e:
            logging.error(f"Error during sorting/selection for Cluster 1: {e}")

    # Combine selected samples
    result_list = top_k_cluster_0_data + top_k_cluster_1_data
    # Convert to dictionary {image_name: pseudo_label (as string)}
    result_dict = {image_name: str(label) for image_name, label in result_list}

    # --- Logging Final Selection Stats ---
    if result_dict:
        # Count selected pseudo-labels (cluster IDs)
        selected_pseudo_labels = [label for _, label in result_list]
        pseudo_label_counts = {
            '0': selected_pseudo_labels.count(0),
            '1': selected_pseudo_labels.count(1)
        }
        logging.info(f"Final selected pseudo-label counts in result_dict: {pseudo_label_counts}")

        # Log purity of the *selected* samples if filename heuristic is applicable
        try:
            selected_image_names = [name for name, _ in result_list]
            # Calculate purity of selected samples based on filename heuristic
            purity_selected = calculate_purity_from_filenames(selected_image_names, selected_pseudo_labels)
            logging.info(f"Purity of Selected Samples (compared to filename heuristic): {purity_selected:.4f}")
        except Exception as e:
            logging.warning(f"Could not calculate purity of selected samples based on filename heuristic: {e}")


    else:
        logging.info("Result dictionary is empty after selection.")


    end_time = time.time()
    logging.info(f"Confidence sample selection finished in {end_time - start_time:.2f} seconds. Selected {len(result_dict)} samples in total.")

    # Check if the number of selected samples is significantly less than expected
    expected_selection = K_0 + K_1
    if expected_selection > 0 and len(result_dict) < expected_selection * 0.9: # Check if less than 90% of expected
        logging.warning(f"Selected significantly fewer samples ({len(result_dict)}) than targeted ({expected_selection}). Check data quality or clustering results.")
    elif not result_dict and expected_selection > 0:
         logging.warning(f"Confidence selection resulted in an empty dictionary, although targeted selection was {expected_selection}. Check data quality or clustering results.")


    return result_dict, purity_score # Return both the dictionary and the purity score



def create_save_paths(args):
    """Creates directories for saving models and logs."""
    try:
         
        data_folder_name = os.path.basename(os.path.normpath(args.data_folder)) if args.data_folder and os.path.exists(args.data_folder) else "dataset"
    except Exception as e:
        logging.warning(f"Could not parse or access data_folder '{args.data_folder}': {e}. Using 'dataset'.")
        data_folder_name = "dataset"

     
    base_save_dir = f'./save/ECL_{args.backbone}'
    model_path_base = os.path.join(base_save_dir, f'{data_folder_name}_models')
    tb_path_base = os.path.join(base_save_dir, f'{data_folder_name}_tensorboard')

     
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = run_timestamp

     
    tb_folder = os.path.join(tb_path_base, run_folder_name)
    try:
        os.makedirs(tb_folder, exist_ok=True)
        logging.info(f"TensorBoard logs will be saved to: {tb_folder}")
    except OSError as e:
        logging.error(f"Failed to create TensorBoard directory {tb_folder}: {e}")
        tb_folder = None  

     
    save_folder = os.path.join(model_path_base, run_folder_name)
    try:
        os.makedirs(save_folder, exist_ok=True)
        logging.info(f"Model checkpoints will be saved to: {save_folder}")
    except OSError as e:
        logging.error(f"Failed to create model save directory {save_folder}: {e}")
        save_folder = None  


     
    if save_folder:  
        try:
            args_path = os.path.join(save_folder, 'args.json')
            args_dict = vars(args)  
             
            serializable_args = {}
            for k, v in args_dict.items():
                try:
                    json.dumps({k: v})  
                    serializable_args[k] = v
                except (TypeError, OverflowError):
                    logging.warning(f"Argument '{k}' with value '{v}' is not JSON serializable. Skipping.")
                    serializable_args[k] = str(v)  

            with open(args_path, 'w') as f:
                json.dump(serializable_args, f, indent=4)
            logging.info(f"Saved script arguments to {args_path}")
        except Exception as e:
            logging.warning(f"Could not save arguments to JSON: {e}")


    return tb_folder, save_folder

def custom_collate_fn(batch):
    """
    Custom collate function that handles None values (failed loads)
    and relies on transforms ensuring consistent tensor sizes before stacking.
    """
    original_batch_size = len(batch)
     
    batch = [item for item in batch if item is not None]
    filtered_batch_size = len(batch)

    if original_batch_size != filtered_batch_size:
        logging.warning(f"Filtered out {original_batch_size - filtered_batch_size} samples due to loading errors.")

     
    if filtered_batch_size == 0:
        return None

    try:
         
         
        components = list(zip(*batch))
        images = components[0]  
        labels = components[1]  
        paths = components[2]   
    except Exception as e:
        logging.error(f"Error during batch unzipping in collate_fn: {e}. Batch structure might be incorrect.")
         
        if batch:
             logging.error(f"Example batch item structure: {type(batch[0])}, contents: {batch[0]}")
        return None  

     
    stacked_imgs = None
    try:
         
        if isinstance(images[0], (list, tuple)) and len(images[0]) == 2:
             
             
            img1s = [item[0] for item in images]
            img2s = [item[1] for item in images]
             
            img1_stack = torch.stack(img1s)
            img2_stack = torch.stack(img2s)
            stacked_imgs = [img1_stack, img2_stack]  
         
        elif isinstance(images[0], torch.Tensor):
             
            stacked_imgs = torch.stack(images)  
        else:
            logging.error(f"Unexpected image data type or structure in collate_fn: {type(images[0])}")
            return None  
    except RuntimeError as e:
         
        logging.error(f"Failed to stack images in collate_fn due to size mismatch (RuntimeError): {e}")
         
        if images and isinstance(images[0], (list, tuple)):
             for i, item in enumerate(images):
                  logging.error(f" Item {i} shapes: {item[0].shape if isinstance(item[0], torch.Tensor) else 'N/A'}, {item[1].shape if isinstance(item[1], torch.Tensor) else 'N/A'}")
        elif images and isinstance(images[0], torch.Tensor):
             for i, img in enumerate(images):
                  logging.error(f" Item {i} shape: {img.shape if isinstance(img, torch.Tensor) else 'N/A'}")
        else:
             logging.error("Could not log image shapes due to empty or malformed images list.")
        return None  
    except Exception as e:
         logging.error(f"Unexpected error during image stacking in collate_fn: {e}")
         return None  


     
    labels_tensor = None
    try:
        labels_tensor = torch.tensor(labels, dtype=torch.long)
    except Exception as e:
        logging.error(f"Failed to convert labels to tensor in collate_fn: {e}")
        return None  

     
    return stacked_imgs, labels_tensor, paths


def tsne_visualization(model, dataloader, epoch, save_dir, args):
    """Memory-optimized t-SNE visualization with RANDOM SAMPLING."""
    if args.tsne_freq <= 0:
        logging.info("t-SNE visualization disabled.")
        return

    logging.info(f"Generating t-SNE visualization for epoch {epoch} with random sampling...")
    start_time = time.time()
    model.eval()  

    if dataloader is None or dataloader.dataset is None:
        logging.error("Dataloader or its dataset is None for tsne_visualization. Skipping.")
        return  

     
    is_data_parallel = isinstance(model, torch.nn.DataParallel)
    eval_model = model.module if is_data_parallel else model

    try:
        dataset = dataloader.dataset
        dataset_size = len(dataset)
    except Exception as e:
        logging.error(f"Could not access dataset or its size for t-SNE sampling: {e}. Skipping.")
        return

     
    max_samples = min(5000, dataset_size)
     
     
    min_tsne_samples = max(4, args.tsne_freq + 1 if hasattr(args, 'tsne_freq') else 31)  
    if max_samples < min_tsne_samples:
        logging.warning(f"Dataset size ({dataset_size}) or max_samples ({max_samples}) too small for meaningful t-SNE (need at least {min_tsne_samples}). Skipping.")
        return

    logging.info(f"Randomly sampling {max_samples} indices from dataset size {dataset_size} for t-SNE...")
    try:
         
        all_indices = list(range(dataset_size))
        sampled_indices = random.sample(all_indices, max_samples)
    except ValueError as e:
        logging.error(f"Error sampling indices for t-SNE: {e}. Dataset size: {dataset_size}, max_samples: {max_samples}. Skipping.")
        return
    except Exception as e:
        logging.error(f"Unexpected error during index sampling for t-SNE: {e}. Skipping.")
        return


    all_features = []
    all_labels = []  

     
     
    try:
        temp_sampler = torch.utils.data.SequentialSampler(sampled_indices)
         
         
        temp_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataloader.batch_size,  
            sampler=temp_sampler,
            num_workers=dataloader.num_workers,  
            pin_memory=dataloader.pin_memory,
            persistent_workers=False,  
            collate_fn=getattr(dataloader, 'collate_fn', None),  
            drop_last=False  
        )
        logging.info(f"Created temporary DataLoader for t-SNE with {len(temp_loader)} batches.")
    except Exception as e:
        logging.error(f"Failed to create temporary DataLoader for t-SNE: {e}. Skipping.")
        return

    collected_count = 0
    with torch.no_grad():  
        for i, batch_data in enumerate(temp_loader):
             
            if batch_data is None:
                logging.warning(f"Skipping None batch returned by collate_fn at index {i} during t-SNE feature extraction.")
                continue

            try:
                  
                  
                  
                 if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                      img, labels = batch_data[:2]  
                 else:
                      logging.warning(f"Unexpected batch data format at index {i} during t-SNE extraction: {type(batch_data)}. Skipping.")
                      continue


                  
                 if not isinstance(img, torch.Tensor) or img.ndim != 4:
                     logging.warning(f"Invalid image tensor received in batch {i} during t-SNE extraction. Shape: {img.shape if isinstance(img, torch.Tensor) else type(img)}. Skipping.")
                     continue
                 if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
                       
                      if isinstance(labels, (list, np.ndarray)):
                           try:
                                labels = torch.tensor(labels, dtype=torch.long)  
                                if labels.ndim != 1: raise ValueError("Conversion failed")
                                logging.warning(f"Converted non-tensor/non-1D labels to tensor in batch {i}.")
                           except Exception as conv_e:
                                logging.warning(f"Invalid labels format and conversion failed in batch {i}: {type(labels)}. Error: {conv_e}. Skipping.")
                                continue
                      else:
                           logging.warning(f"Invalid labels tensor received in batch {i} during t-SNE extraction. Shape: {labels.shape if isinstance(labels, torch.Tensor) else type(labels)}. Skipping.")
                           continue

                  
                 if img.shape[0] != labels.shape[0]:
                      logging.warning(f"Mismatch between image tensor size ({img.shape[0]}) and labels tensor size ({labels.shape[0]}) in batch {i}. Skipping.")
                      continue


            except Exception as e:
                 logging.error(f"Error unpacking or validating batch data at index {i} during t-SNE extraction: {e}")
                 continue  

             
            if torch.cuda.is_available():
                try:
                    img = img.cuda(non_blocking=True)
                except Exception as e:
                    logging.error(f"Error moving image batch {i} to CUDA during t-SNE extraction: {e}")
                    if "CUDA out of memory" in str(e):
                         logging.error("CUDA OOM during data transfer for t-SNE. Try reducing batch size.")
                    continue  

             
            try:
                 
                output = eval_model(img)
                if isinstance(output, tuple) and len(output) >= 1:
                      features = output[0]  
                       
                elif isinstance(output, torch.Tensor):
                     features = output  
                else:
                    logging.error(f"Unexpected model output type in t-SNE extraction (batch {i}): {type(output)}. Skipping.")
                    continue  

                 
                if not isinstance(features, torch.Tensor) or features.ndim != 2:
                    logging.error(f"Model output features are not a 2D tensor (batch {i}). Shape: {features.shape if isinstance(features, torch.Tensor) else type(features)}. Skipping.")
                    continue  
                 
                if features.shape[0] != img.shape[0]:
                    logging.warning(f"Mismatch between feature batch size ({features.shape[0]}) and image batch size ({img.shape[0]}) in batch {i}. Skipping.")
                    continue


            except Exception as e:
                 logging.error(f"Error during model forward pass in t-SNE extraction (batch {i}): {e}")
                 if torch.cuda.is_available() and "CUDA out of memory" in str(e):
                      logging.error("CUDA OOM during t-SNE feature extraction forward pass. Try reducing batch size.")
                 continue  

             
            try:
                 features_np = features.detach().cpu().numpy()
                 labels_np = labels.detach().cpu().numpy()
            except Exception as e:
                 logging.error(f"Error moving features/labels to CPU/NumPy in t-SNE extraction (batch {i}): {e}")
                 continue  

             
            if np.isnan(features_np).any() or np.isinf(features_np).any():
                 logging.warning(f"NaN or Inf detected in features for batch {i}. Skipping batch.")
                 continue  

             
            all_features.append(features_np)
            all_labels.append(labels_np)
            collected_count += features_np.shape[0]

             
            if (i + 1) % (max(1, len(temp_loader) // 20)) == 0:  
                    logging.info(f"t-SNE Feature extraction: Batch [{i+1}/{len(temp_loader)}], Collected {collected_count}/{max_samples} samples.")

     
    if not all_features:
         logging.error("No valid features collected for t-SNE. Skipping visualization.")
         return  

    try:
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        num_samples = all_features.shape[0]  

        logging.info(f"Collected {num_samples} features and {all_labels.shape[0]} labels for t-SNE.")
         
        if num_samples != all_labels.shape[0]:
             logging.error(f"Mismatch between collected features ({num_samples}) and labels ({all_labels.shape[0]}) for t-SNE. Skipping.")
             return
        if num_samples < min_tsne_samples:
             logging.warning(f"Not enough features collected ({num_samples}) for t-SNE (need at least {min_tsne_samples}). Skipping.")
             return
        if np.isnan(all_features).any() or np.isinf(all_features).any():
             logging.error("NaN or Inf values found in concatenated t-SNE features. Skipping visualization.")
             return


    except Exception as e:
         logging.error(f"Error concatenating features/labels for t-SNE: {e}. Skipping.")
         return


     
    logging.info(f"Running t-SNE on {num_samples} samples...")
     

    try:
         
         
        tsne_perplexity = min(30, max(5, num_samples - 1))
        logging.info(f"Using t-SNE perplexity: {tsne_perplexity}")

        tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity)
        features_2d = tsne.fit_transform(all_features)  
        logging.info("t-SNE dimensionality reduction finished.")
    except ValueError as e:
        logging.error(f"t-SNE failed (ValueError): {e}. This might happen if data is not suitable (e.g., all points identical) or not enough samples for perplexity. Features shape: {all_features.shape}")
        logging.error(f"Feature stats - Mean: {np.mean(all_features):.4f}, Std: {np.std(all_features):.4f}, Min: {np.min(all_features):.4f}, Max: {np.max(all_features):.4f}")
        return
    except Exception as e:
        logging.error(f"t-SNE failed with unexpected error: {e}. Skipping visualization.")
        return


     
    logging.info("Plotting t-SNE results...")
    try:
        plt.figure(figsize=(10, 8))
         
        unique_labels = np.unique(all_labels)
        num_classes = len(unique_labels)
         
        if num_classes < 2:
             logging.warning(f"Only {num_classes} unique label(s) found. t-SNE plot might not be informative.")
              
             if num_classes == 1:
                  plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.5)
             else:  
                  logging.warning("No labels found for plotting.")
                  plt.text(0.5, 0.5, "No data to plot", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        else:
            colors = cm.rainbow(np.linspace(0, 1, num_classes))
             
            label_to_idx = {label: i for i, label in enumerate(unique_labels)}

            for label in unique_labels:
                indices = np.where(all_labels == label)
                plt.scatter(features_2d[indices, 0], features_2d[indices, 1],
                            color=colors[label_to_idx[label]], label=f'Label {label}', alpha=0.5)

            plt.legend(title="Labels")  

        plt.title(f't-SNE of Features at Epoch {epoch}')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True)

         
        if save_dir:
             
            os.makedirs(save_dir, exist_ok=True)
            plot_filename = f'tsne_epoch_{epoch:03d}.png'
            plot_path = os.path.join(save_dir, plot_filename)
            plt.savefig(plot_path)
            logging.info(f"Saved t-SNE plot to {plot_path}")
        else:
            logging.warning("Save directory not provided. t-SNE plot not saved.")

        plt.close()  

    except Exception as e:
        logging.error(f"Error during t-SNE plotting: {e}. Skipping plot saving.")

    end_time = time.time()
    logging.info(f"t-SNE visualization finished in {end_time - start_time:.2f} seconds.")


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """Saves model, optimizer, scaler, and epoch state."""
    try:
        torch.save(state, filename)
        logging.info(f"Checkpoint saved to {filename}")
    except Exception as e:
        logging.error(f"Failed to save checkpoint to {filename}: {e}")


def load_checkpoint(checkpoint_path, model, optimizer, scaler, scheduler=None):
    """Loads checkpoint state."""
    if not os.path.isfile(checkpoint_path):
        logging.warning(f"Checkpoint file not found at {checkpoint_path}. Starting from scratch.")
        return 0  

    logging.info(f"Loading checkpoint from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

         
        if isinstance(model, torch.nn.DataParallel):
             model.module.load_state_dict(checkpoint['model'], strict=False)
        else:
             model.load_state_dict(checkpoint['model'], strict=False)
        logging.info("Model state loaded.")

         
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info("Optimizer state loaded.")

         
        if 'scaler' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
            logging.info("GradScaler state loaded.")
        elif 'scaler' in checkpoint and scaler is None:
             logging.warning("Checkpoint contains scaler state, but AMP is disabled. Skipping scaler load.")
        elif 'scaler' not in checkpoint and scaler is not None:
             logging.warning("Checkpoint does not contain scaler state, but AMP is enabled. Starting scaler from scratch.")


         
        if scheduler is not None and 'scheduler' in checkpoint:
             scheduler.load_state_dict(checkpoint['scheduler'])
             logging.info("Scheduler state loaded.")
        elif scheduler is not None and 'scheduler' not in checkpoint:
             logging.warning("Checkpoint does not contain scheduler state. Starting scheduler from scratch.")
        elif scheduler is None and 'scheduler' in checkpoint:
             logging.warning("Checkpoint contains scheduler state, but no scheduler is used. Skipping scheduler load.")


         
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f"Resumed from epoch {checkpoint['epoch']}. Starting training from epoch {start_epoch}.")

        return start_epoch

    except Exception as e:
        logging.error(f"Failed to load checkpoint from {checkpoint_path}: {e}. Starting from scratch.")
        return 0  
    

def main():
    """Main semi-supervised training loop with periodic pseudo-labeling and initial purity calculation."""
    args = args_func()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create save directories
    tb_folder, save_folder = create_save_paths(args) # Assuming create_save_paths is defined
    if not save_folder:
        logging.error("Failed to create save directory. Cannot save checkpoints or plots. Exiting.")
        return

    # Load Initial Labels (potentially noisy) from the specified file
    # This dictionary will be updated with pseudo-labels during training
    full_dataset_labels = {}
    if args.pseudo_label_file and os.path.isfile(args.pseudo_label_file):
        try:
            with open(args.pseudo_label_file, 'r') as f:
                full_dataset_labels = json.load(f)
            logging.info(f"Loaded {len(full_dataset_labels)} initial labels from {args.pseudo_label_file}")
        except Exception as e:
            logging.error(f"Failed to load initial label file {args.pseudo_label_file}: {e}. Exiting.")
            return

    # Ensure we have some initial labels to start
    if not full_dataset_labels:
        logging.error("No initial labels loaded. Cannot start training. Exiting.")
        return

    # Set up model and Criterion
    model, criterion = set_model(args) # Assuming set_model is defined
    if model is None or criterion is None:
        logging.error("Model or criterion initialization failed. Exiting.")
        return

    # Set up optimizer
    optimizer = set_optimizer(args, model) # Assuming set_optimizer is defined

    # Set up GradScaler for AMP
    scaler = GradScaler(enabled=args.use_amp, init_scale=args.init_scale)
    if args.use_amp:
        logging.info(f"Initialized GradScaler with init_scale={args.init_scale}.")

    # Set up learning rate scheduler
    scheduler = set_scheduler(args, optimizer, args.epochs, None) # Assuming set_scheduler is defined

    # Load checkpoint if resuming
    start_epoch = args.current_epoch
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scaler, scheduler) # Assuming load_checkpoint is defined
        # If loading from a checkpoint, the full_dataset_labels would also be loaded here

    # List to store purity scores and corresponding epochs for plotting
    purity_scores_history = []

    # --- Calculate and record initial purity (Epoch start_epoch) ---
    # This happens before any training epochs start
    if args.select_confidence_epoch > 0: # Only calculate initial purity if clustering is enabled
        logging.info(f"Calculating initial K-Means purity (Epoch {start_epoch}) based on filename heuristic...")
        # Create a DataLoader for clustering/evaluation using the initial labels
        # We use the same source as the training data for clustering
        clustering_data_source = full_dataset_labels
        clustering_loader_initial = set_loader(clustering_data_source, args, is_confidence_selection=True)

        if clustering_loader_initial is None:
            logging.error("Failed to create initial clustering dataloader. Cannot calculate initial purity.")
        else:
            # Call select_confidence_sample. It will perform clustering on initial features
            # and return the purity compared to the filename heuristic.
            # We don't use the generated pseudo_labels_generated here, only the purity.
            _, initial_purity = select_confidence_sample(model, clustering_loader_initial, args)

            # Store the initial purity score if it was calculated
            if initial_purity is not None:
                purity_scores_history.append((start_epoch, initial_purity)) # Use start_epoch for initial point


    # --- Main Semi-Supervised Training Loop ---
    for epoch in range(start_epoch + 1, args.epochs + 1): # Start from start_epoch + 1 if resuming
        logging.info(f"Starting Epoch {epoch}/{args.epochs}")

        # --- Create DataLoader for the current epoch using the current labels ---
        # This loader uses the potentially updated full_dataset_labels
        train_loader = set_loader(full_dataset_labels, args, is_confidence_selection=False) # Assuming set_loader is defined

        if train_loader is None:
            logging.error("Failed to create training dataloader for epoch {epoch}. Exiting training loop.")
            break # Exit training loop if DataLoader creation fails

        # Run one training epoch using the current set of labels (initial noisy/pseudo + new pseudo-labels)
        logging.info(f"Starting training epoch {epoch} with {len(full_dataset_labels)} samples...")
        loss = train(train_loader, model, criterion, optimizer, epoch, args, scaler, scheduler) # Assuming train is defined

        if loss is None:
             logging.error(f"Training failed for epoch {epoch}. Exiting training loop.")
             break

        logging.info(f"Epoch {epoch} finished. Average Loss: {loss:.4f}")

        # --- Periodically perform pseudo-labeling and update training data ---
        # Perform this step every args.select_confidence_epoch
        if args.select_confidence_epoch > 0 and epoch % args.select_confidence_epoch == 0:
             logging.info(f"Epoch {epoch}: Starting pseudo-labeling step.")

             # Create a DataLoader for feature extraction and clustering.
             # This loader uses the current set of labels as the source for which samples to process.
             clustering_data_source = full_dataset_labels
             clustering_loader = set_loader(clustering_data_source, args, is_confidence_selection=True)

             if clustering_loader is None:
                  logging.error("Failed to create clustering dataloader. Skipping pseudo-labeling.")
             else:
                  # Call the function to get pseudo-labels and purity
                  # Purity is now calculated based on filename heuristic internally
                  pseudo_labels_generated, current_purity = select_confidence_sample(model, clustering_loader, args)
                  logging.info(f"Epoch {epoch}: select_confidence_sample finished. Generated {len(pseudo_labels_generated)} pseudo-labels.")

                  # Store the purity score if it was calculated
                  if current_purity is not None:
                      purity_scores_history.append((epoch, current_purity))

                  # --- Update the main label dictionary with generated pseudo-labels ---
                  logging.info(f"Updating training labels with {len(pseudo_labels_generated)} pseudo-labels...")
                  full_dataset_labels.update(pseudo_labels_generated) # Add/overwrite labels with new pseudo-labels
                  logging.info(f"Total samples in training set after update: {len(full_dataset_labels)}")

                  # The train_loader will be recreated at the start of the *next* epoch
                  # using this updated full_dataset_labels dictionary.


        # Run t-SNE visualization periodically
        if args.tsne_freq > 0 and epoch % args.tsne_freq == 0:
             # Create a DataLoader for t-SNE visualization using the current labels
             # This will visualize clusters colored by their current labels (initial noisy/pseudo or new pseudo)
             tsne_loader = set_loader(full_dataset_labels, args, is_confidence_selection=True)
             if tsne_loader is None:
                  logging.error("Failed to create t-SNE dataloader for visualization. Skipping.")
             else:
                  tsne_visualization(model, tsne_loader, epoch, tb_folder, args) # Assuming tsne_visualization is defined


        # Save checkpoint periodically
        if args.checkpoint_freq > 0 and epoch % args.checkpoint_freq == 0:
            if save_folder:
                 checkpoint_filename = os.path.join(save_folder, f'ckpt_epoch_{epoch:03d}.pth')
                 checkpoint_state = {
                     'epoch': epoch,
                     'model': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'args': vars(args),
                     'loss': loss,
                     'full_dataset_labels': full_dataset_labels # Save the updated labels
                 }
                 if args.use_amp:
                     checkpoint_state['scaler'] = scaler.state_dict()
                 if scheduler is not None:
                     checkpoint_state['scheduler'] = scheduler.state_dict()

                 save_checkpoint(checkpoint_state, checkpoint_filename) # Assuming save_checkpoint is defined
            else:
                 logging.warning("Save folder not available. Skipping checkpoint saving.")


    logging.info("Training finished.")

    # --- Generate and save the purity plot after training ---
    save_purity_plot(purity_scores_history, args.plot_dir)


if __name__ == '__main__':
    main()
