import os
import json
import logging
import torch
import traceback
import numpy as np
import argparse
from collections import OrderedDict
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
from pathlib import Path   
from datetime import datetime   

  
from model import ECL
  
from data.transform import get_transforms

  
  
  

  
class ClassifierDataset(Dataset):
    def __init__(self, image_dir, pseudo_label_dict, transform=None):
        self.image_dir = image_dir
        self.pseudo_label_dict = pseudo_label_dict
        self.transform = transform
        valid_image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
        self.image_names = [
            name for name in pseudo_label_dict.keys()
            if os.path.exists(os.path.join(image_dir, name)) and os.path.splitext(name)[1].lower() in valid_image_extensions
        ]
        if not self.image_names:
            logging.warning(f"ClassifierDataset initialized with 0 images. Check image_dir and pseudo_label_dict integrity.")
        else:
            logging.info(f"Initialized ClassifierDataset with {len(self.image_names)} images.")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_value = self.pseudo_label_dict.get(img_name)
        if label_value is None:
             logging.warning(f"Pseudo-label not found for {img_name} (index {idx}). Skipping.")
             return None   
        try:
             label = int(label_value)
        except (ValueError, TypeError):
             logging.warning(f"Invalid pseudo-label format for {img_name}: {label_value}. Skipping.")
             return None   

        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}", exc_info=True)
            return None   

  
def collate_fn_skip_none(batch):
      
    original_len = len(batch)
    batch = [item for item in batch if item is not None and item[0] is not None and item[1] is not None]
    if len(batch) < original_len:
        logging.debug(f"Skipped {original_len - len(batch)} None items in collate_fn.")
    if not batch:
        return None, None

    imgs, labels = zip(*batch)
    try:
        imgs = torch.stack(imgs, 0)
        labels = torch.tensor(labels, dtype=torch.long)   
    except Exception as e:
        logging.error(f"Error stacking images or labels in collate_fn: {e}", exc_info=True)
        return None, None
    return imgs, labels


  
def load_encoder(ckpt_path, device='cpu'):
    if not os.path.exists(ckpt_path):
        logging.error(f"Encoder checkpoint file not found: {ckpt_path}")
        raise FileNotFoundError(f"Encoder checkpoint file not found: {ckpt_path}")

    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        logging.error(f"Failed to load encoder checkpoint file {ckpt_path}: {e}", exc_info=True)
        raise

    loaded_args = checkpoint.get('args')
    if loaded_args is None:
         logging.error("Encoder checkpoint does not contain 'args' dictionary.")
         raise ValueError("Encoder checkpoint missing training arguments.")
    logging.info(f"Loaded training arguments from encoder checkpoint: {loaded_args}") 

      
    backbone_type = loaded_args.get('backbone')
    if backbone_type is None:
        logging.error("Encoder checkpoint args missing 'backbone' information.")
        raise ValueError("Encoder checkpoint args missing 'backbone' information.")

      
    proj_out_dim = loaded_args.get('out_dim', 128)

    try:
        model = ECL(out_dim=proj_out_dim, backbone_type=backbone_type)
        logging.info(f"Instantiated full ECL model with backbone: {backbone_type}")
    except Exception as e:
        logging.error(f"Failed to instantiate ECL model for backbone {backbone_type}: {e}", exc_info=True)
        raise

    model_state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint.get('model_state_dict')))
    if model_state_dict is None or not isinstance(model_state_dict, (dict, OrderedDict)):
         raise ValueError("Could not find a suitable state_dict in the encoder checkpoint.")

    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
      
    try:
        logging.info("Attempting to load encoder state_dict with strict=True...")
        model.load_state_dict(new_state_dict, strict=True)
        logging.info("Successfully loaded encoder model state_dict with strict=True.")
    except RuntimeError as e:
        logging.warning(f"Strict load failed for encoder: {e}. Attempting load with strict=False.")
        try:
            model.load_state_dict(new_state_dict, strict=False)
            logging.info("Successfully loaded encoder model state_dict with strict=False.")
            model_keys = set(model.state_dict().keys())
            checkpoint_keys = set(new_state_dict.keys())
            missing_keys = model_keys - checkpoint_keys
            unexpected_keys = checkpoint_keys - model_keys
            if missing_keys:
                 logging.warning(f"Missing keys in encoder when loading strict=False: {missing_keys}")
            if unexpected_keys:
                 logging.warning(f"Unexpected keys in encoder when loading strict=False: {unexpected_keys}")
        except Exception as e_false:
            logging.error(f"Failed to load encoder state_dict even with strict=False: {e_false}", exc_info=True)
            raise e_false
    except Exception as e:
        logging.error(f"An unexpected error occurred during encoder state_dict loading: {e}", exc_info=True)
        raise
      

    model.eval().to(device)
    return model, backbone_type

  
class FeatureClassifier(nn.Module):
    def __init__(self, feature_dimension, num_classes=2, hidden_dim1=512, hidden_dim2=128):
        super(FeatureClassifier, self).__init__()
        self.fc1 = nn.Linear(feature_dimension, hidden_dim1)
        self.relu1 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU(inplace=False)
        self.fc3 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        if x.ndim == 1:
             x = x.unsqueeze(0)
          
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

  
def train_classifier(image_dir, pseudo_label_file, ckpt_path_encoder, args, device):
    logging.info(f"Starting Classifier Training.")
    logging.info(f"Image directory: {image_dir}, Pseudo-label file: {pseudo_label_file}")
    logging.info(f"Encoder Checkpoint: {ckpt_path_encoder}")
    
      
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = Path(args.output_dir_classifier) / current_time_str
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Classifier outputs will be saved to: {run_output_dir}")

      
    try:
        with open(run_output_dir / "training_args_classifier.json", 'w') as f:
            json.dump(vars(args), f, indent=4)
        logging.info(f"Saved classifier training arguments to {run_output_dir / 'training_args_classifier.json'}")
    except Exception as e:
        logging.warning(f"Could not save training arguments: {e}", exc_info=True)
      

    if not os.path.exists(image_dir):
        logging.error(f"Image directory not found: {image_dir}. Exiting.")
        return
    if not os.path.exists(pseudo_label_file):
        logging.error(f"Pseudo-label file not found: {pseudo_label_file}. Exiting.")
        return

    try:
        with open(pseudo_label_file, 'r') as f:
            pseudo_labels = json.load(f)
        logging.info(f"Loaded {len(pseudo_labels)} pseudo-labels.")
    except Exception as e:
        logging.error(f"Failed to load pseudo-label file {pseudo_label_file}: {e}. Exiting.", exc_info=True)
        return

    try:
        ecl_model, backbone_type = load_encoder(ckpt_path_encoder, device)
        logging.info(f"Loaded ECL model with backbone {backbone_type} from {ckpt_path_encoder}.")
    except Exception as e:
        logging.error(f"Failed to load ECL model for classifier training: {e}. Exiting.", exc_info=True)
        return

    logging.info("Freezing ECL model parameters.")
    for param in ecl_model.parameters():
        param.requires_grad = False

    try:
        dummy_img = torch.randn(1, 3, args.frame_size[0], args.frame_size[1]).to(device)
        with torch.no_grad():
            dummy_output = ecl_model(dummy_img)
            if isinstance(dummy_output, tuple) and len(dummy_output) > 0 and isinstance(dummy_output[0], torch.Tensor):
                feature_dimension = dummy_output[0].shape[1]
                logging.info(f"Automatically determined feature dimension from ECL output[0]: {feature_dimension}")
            else:   
                feature_dimension = ecl_model.encoder.feature_dim   
                logging.info(f"Using feature dimension from ecl_model.encoder.feature_dim: {feature_dimension}")

    except AttributeError:   
         logging.error(f"Could not automatically determine feature dimension. Tried model output and direct attribute.")
         logging.error(f"ECL output type: {type(dummy_output)}. Expected tuple with tensor or direct .feature_dim.")
         return
    except Exception as e:
        logging.error(f"Error determining feature dimension: {e}. Exiting.", exc_info=True)
        return


    num_classes = 2
    classifier = FeatureClassifier(feature_dimension, num_classes, 
                                   hidden_dim1=args.classifier_hidden_dim1, 
                                   hidden_dim2=args.classifier_hidden_dim2).to(device)
    logging.info(f"Initialized MLP Classifier with input_dim={feature_dimension}, hidden_dims=[{args.classifier_hidden_dim1}, {args.classifier_hidden_dim2}], output_classes={num_classes}.")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.classifier_lr)

    start_epoch = 0
      
    if args.resume_classifier_ckpt:
        if os.path.isfile(args.resume_classifier_ckpt):
            logging.info(f"Resuming classifier training from checkpoint: {args.resume_classifier_ckpt}")
            try:
                ckpt = torch.load(args.resume_classifier_ckpt, map_location=device)
                classifier.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                start_epoch = ckpt['epoch']
                  
                logging.info(f"Resumed from epoch {start_epoch}. Classifier and optimizer states loaded.")
            except Exception as e:
                logging.error(f"Failed to load classifier checkpoint: {e}. Starting from scratch.", exc_info=True)
                start_epoch = 0   
        else:
            logging.warning(f"Resume checkpoint not found: {args.resume_classifier_ckpt}. Starting from scratch.")
      

    train_transform = get_transforms(name="train", size=args.frame_size[0], backbone=backbone_type, norm="imagenet")   
    train_dataset = ClassifierDataset(image_dir, pseudo_labels, transform=train_transform)

    if len(train_dataset) == 0:
        logging.error("No images found for classifier training after dataset initialization. Exiting.")
        return
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_none)
    logging.info(f"Created DataLoader for classifier training with {len(train_dataset)} images.")

    logging.info(f"Starting training of the MLP Classifier from epoch {start_epoch + 1}...")
    num_linear_epochs = args.classifier_epochs

    for epoch in range(start_epoch, num_linear_epochs):
        classifier.train()
        total_loss = 0
        batches_processed = 0
        correct_predictions = 0
        total_samples = 0

        for i, data in enumerate(tqdm(train_loader, desc=f"Classifier Epoch {epoch+1}/{num_linear_epochs}")):
            if data is None or data[0] is None or data[1] is None:
                 logging.warning(f"Skipping batch {i} in epoch {epoch+1} due to data loading error or empty batch.")
                 continue
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                 try:
                     features, _ = ecl_model(imgs)   
                 except Exception as e:
                     logging.error(f"Error during forward pass of ECL model in batch {i}: {e}", exc_info=True)
                     continue

            try:
                 logits = classifier(features)
                 loss = criterion(logits, labels)
                 
                 _, predicted = torch.max(logits.data, 1)
                 total_samples += labels.size(0)
                 correct_predictions += (predicted == labels).sum().item()

            except Exception as e:
                 logging.error(f"Error during forward/loss/accuracy of classifier in batch {i}: {e}", exc_info=True)
                 continue

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches_processed += 1

        if batches_processed > 0:
            avg_loss = total_loss / batches_processed
            accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
            logging.info(f"Classifier Epoch {epoch+1}: Average Training Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")
        else:
            logging.warning(f"Classifier Epoch {epoch+1}: No batches processed.")

          
        if args.classifier_checkpoint_freq > 0 and (epoch + 1) % args.classifier_checkpoint_freq == 0 or (epoch + 1) == num_linear_epochs:
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args)   
            }
            checkpoint_filename = run_output_dir / f"classifier_{backbone_type}_{args.dataset_name}_epoch_{epoch+1:03d}.pth"
            try:
                torch.save(checkpoint_data, checkpoint_filename)
                logging.info(f"Saved classifier checkpoint to {checkpoint_filename}")
            except Exception as e:
                logging.error(f"Failed to save classifier checkpoint: {e}", exc_info=True)
      
    logging.info("Finished training the MLP Classifier.")
      

  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an MLP classifier on frozen encoder features.')
    parser.add_argument('--image_dir', type=str, default="/home/teaching/deepfake/FF_frames", help='Path to image folder for classifier training')
    parser.add_argument('--pseudo_label_file', type=str, default="/home/teaching/deepfake/project/files/Unsupervised_DF_Detection/output_gpu/FF_real_labels/image_pseudo_labels_gpu.json", help='Path to JSON file with pseudo-labels')
    parser.add_argument('--ckpt_path_encoder', type=str, default="/home/teaching/deepfake/project/files/Unsupervised_DF_Detection/save/ECL_convnext_base/FF_frames_models/20250512_012333/ckpt_epoch_018.pth", help='Checkpoint path of the trained ECL encoder model')
    
      
    parser.add_argument('--output_dir_classifier', type=str, default="/home/teaching/deepfake/project/files/Unsupervised_DF_Detection/trained_classifier_runs", help='Base directory to save classifier training runs (timestamped folders will be created inside)')
      
    parser.add_argument('--dataset_name', type=str, default="FF", help="Name of the dataset (e.g., CelebDFv2, FFPP) for filenaming")

    parser.add_argument('--classifier_epochs', type=int, default=20, help='Number of epochs for MLP classifier')
    parser.add_argument('--classifier_lr', type=float, default=1e-4, help='Learning rate for MLP classifier')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for classifier training')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers (adjust based on CPU cores)')   
    parser.add_argument('--classifier_hidden_dim1', type=int, default=512, help='Dimension of the first hidden layer')
    parser.add_argument('--classifier_hidden_dim2', type=int, default=128, help='Dimension of the second hidden layer')
    parser.add_argument('--frame_size', type=int, nargs=2, default=[299, 299], help='Image size (height width) for preprocessing')

      
    parser.add_argument('--classifier_checkpoint_freq', type=int, default=2, help='Frequency (in epochs) to save classifier checkpoints (0 to disable, always saves last)')
    parser.add_argument('--resume_classifier_ckpt', type=str, default=None, help='Path to classifier checkpoint .pth file to resume training')

    args = parser.parse_args()

      
    log_level = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
      
    Path(args.output_dir_classifier).mkdir(parents=True, exist_ok=True)


    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {DEVICE}")

    args.frame_size = tuple(args.frame_size)

    train_classifier(
        args.image_dir,
        args.pseudo_label_file,
        args.ckpt_path_encoder,
        args,
        DEVICE
    )