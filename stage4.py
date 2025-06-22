import os
import logging
import torch
import cv2
import numpy as np
import glob
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
from PIL import Image

 
from model import ECL
 
from data.transform import get_transforms

 
import torch.nn as nn

 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

 
def load_encoder(ckpt_path, device='cpu'):
    """
    Loads the full ECL model from a checkpoint, dynamically building the
    architecture based on the backbone type saved in the checkpoint args.
    Returns the loaded ECL model and its backbone type.
    """
    if not os.path.exists(ckpt_path):
        logging.error(f"Checkpoint file not found: {ckpt_path}")
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    try:
         
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        logging.error(f"Failed to load checkpoint file {ckpt_path}: {e}")
        raise

    loaded_args = checkpoint.get('args')
    if loaded_args is None:
         logging.error("Checkpoint does not contain 'args' dictionary.")
         raise ValueError("Checkpoint missing training arguments.")

    logging.info(f"Loaded training arguments from checkpoint: {loaded_args}")

    backbone_type = loaded_args.get('backbone')
    if backbone_type is None:
        logging.error("Checkpoint args missing 'backbone' information.")
        raise ValueError("Checkpoint args missing 'backbone' information.")

     
     
     
    proj_out_dim = loaded_args.get('out_dim', 128)

    try:
         
         
        model = ECL(out_dim=proj_out_dim, backbone_type=backbone_type)
        logging.info(f"Instantiated full ECL model with backbone: {backbone_type}")
    except Exception as e:
        logging.error(f"Failed to instantiate ECL model for backbone {backbone_type}: {e}")
        raise

     
    model_state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint.get('model_state_dict')))

    if model_state_dict is None or not isinstance(model_state_dict, (dict, OrderedDict)):
         raise ValueError("Could not find a suitable state_dict in the checkpoint.")


     
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    try:
         
        logging.info("Attempting to load state_dict with strict=True...")
        model.load_state_dict(new_state_dict, strict=True)
        logging.info("Successfully loaded model state_dict with strict=True.")
    except RuntimeError as e:
        logging.warning(f"Strict load failed: {e}. Attempting load with strict=False.")
        try:
            model.load_state_dict(new_state_dict, strict=False)
            logging.info("Successfully loaded model state_dict with strict=False.")
             
            model_keys = set(model.state_dict().keys())
            checkpoint_keys = set(new_state_dict.keys())
            missing_keys = model_keys - checkpoint_keys
            unexpected_keys = checkpoint_keys - model_keys
            if missing_keys:
                 logging.warning(f"Missing keys when loading strict=False: {missing_keys}")
            if unexpected_keys:
                 logging.warning(f"Unexpected keys when loading strict=False: {unexpected_keys}")

        except Exception as e_false:
            logging.error(f"Failed to load state_dict into model even with strict=False: {e_false}")
             
            model_keys = set(model.state_dict().keys())
            checkpoint_keys = set(new_state_dict.keys())
            missing_keys = model_keys - checkpoint_keys
            unexpected_keys = checkpoint_keys - model_keys
            logging.error(f"Missing keys: {missing_keys}")
            logging.error(f"Unexpected keys: {unexpected_keys}")
            raise e_false  
    except Exception as e:
        logging.error(f"An unexpected error occurred during state_dict loading: {e}")
        raise

     
    model.eval().to(device)
     
    return model, backbone_type

 
class FeatureClassifier(nn.Module):
    """
    A small MLP classifier to be trained on top of frozen features.
    This must match the architecture used in train_classifier.py.
    """
    def __init__(self, feature_dimension, num_classes=2, hidden_dim1=512, hidden_dim2=128):  
        super(FeatureClassifier, self).__init__()
        self.fc1 = nn.Linear(feature_dimension, hidden_dim1)  
        self.relu1 = nn.ReLU()  
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  
        self.relu2 = nn.ReLU()  
        self.fc3 = nn.Linear(hidden_dim2, num_classes)  

    def forward(self, x):
         
        if x.ndim == 1:
             x = x.unsqueeze(0)  
        elif x.ndim > 2:
              
              
             logging.warning(f"FeatureClassifier received input with unexpected dimensions: {x.shape}. Assuming [batch_size, feature_dimension].")

        x = self.fc1(x)  
        x = self.relu1(x)  
        x = self.fc2(x)  
        x = self.relu2(x)  
        x = self.fc3(x)

        return x  

 
def extract_frame_predictions(video_path, ecl_model, classifier, transform, device, frame_limit=32):
    """
    Extracts frame-level classification probabilities for a limited number of frames
    from a single video using the loaded ECL model and classifier.
    """
    cap = cv2.VideoCapture(video_path)
    frame_predictions = []  

    if not cap.isOpened():
        logging.warning(f"Could not open video file: {video_path}")
        return []

    processed_count = 0
     
    ecl_model.eval()
    classifier.eval()
    with torch.no_grad():
        while cap.isOpened() and processed_count < frame_limit:
            ret, frame = cap.read()
            if not ret:
                break  

            try:
                 
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                 
                 
                img_tensor = transform(pil_img).unsqueeze(0).to(device)

                 
                if img_tensor.shape[2] != args.frame_size[0] or img_tensor.shape[3] != args.frame_size[1]:
                      
                     img_tensor = torch.nn.functional.interpolate(
                         img_tensor, size=args.frame_size, mode='bilinear', align_corners=False
                     )
                      

                 
                 
                features, _ = ecl_model(img_tensor)

                 
                logits = classifier(features)

                 
                probabilities = torch.softmax(logits, dim=1)

                 
                frame_predictions.append(probabilities.squeeze(0).cpu().numpy())

                processed_count += 1

            except Exception as e:
                 
                logging.error(f"Error processing frame for video {os.path.basename(video_path)}: {e}")
                 
                continue  


    cap.release()

    return frame_predictions  

 
def aggregate_video_prediction(frame_predictions, aggregation_method='majority_vote'):
    """
    Aggregates frame-level classification probabilities to a single video-level prediction.

    Args:
        frame_predictions (list): List of numpy arrays, each representing frame probabilities [P(class0), P(class1)].
        aggregation_method (str): 'majority_vote' (on predicted classes per frame) or 'average_prob' (average probabilities then argmax).
        Assumes class index 0 is REAL and class index 1 is FAKE based on your ground truth structure.

    Returns:
        int: Predicted label for the video (0 for REAL, 1 for FAKE). Returns -1 if no valid frame predictions.
    """
    if not frame_predictions:
        logging.warning("No frame predictions provided for aggregation.")
        return -1  

     
    try:
        frame_probs_array = np.stack(frame_predictions, axis=0)  
    except ValueError as e:
        logging.error(f"Error stacking frame predictions: {e}. Ensure all frames have consistent output shape.")
        return -1  

    if aggregation_method == 'majority_vote':
         
        frame_classes = np.argmax(frame_probs_array, axis=1)

         
         
         
        class_counts = np.bincount(frame_classes, minlength=2)

         
        predicted_class_index = np.argmax(class_counts)

        return predicted_class_index  

    elif aggregation_method == 'average_prob':
         
        average_probs = np.mean(frame_probs_array, axis=0)  

         
        predicted_class_index = np.argmax(average_probs)

        return predicted_class_index  

    else:
        logging.error(f"Unknown aggregation method: {aggregation_method}")
        return -1  

 
def plot_confusion_matrix(cm, class_names, title, encoder_type):
    """
    Plots the confusion matrix and displays performance metrics.
    Args:
        cm (np.ndarray): Confusion matrix (2x2).
        class_names (list): List of class names (e.g., ["REAL", "FAKE"]).
        title (str): Title for the plot.
        encoder_type (str): Identifier for saving the file.
    """
     
     
     
    tn, fp, fn, tp = 0, 0, 0, 0  
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.size == 1:  
        val = cm[0,0]
         
         
        if class_names[0] == "FAKE":  
              
              
              
              
              
             logging.warning("1x1 CM - Assuming it corresponds to class 0 ('REAL'). Metrics might be inaccurate.")
             tn = val  
        elif class_names[0] == "REAL":  
               
               
              logging.warning("1x1 CM - Assuming it corresponds to class 0 ('REAL'). Metrics might be inaccurate.")
              tn = val  
         
        elif len(class_names) > 1 and class_names[1] == "FAKE":
             logging.warning("1x1 CM - Assuming it corresponds to class 1 ('FAKE'). Metrics might be inaccurate.")
             tp = val  
        else:
             logging.error(f"Could not infer class for single value CM. Defaulting metrics to 0.")


    else:  
        logging.error(f"Unexpected confusion matrix shape: {cm.shape}. Metrics will be calculated based on available data but might be incorrect.")
         

    total_samples = tp + tn + fp + fn
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
     
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0  
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0   
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0  

     
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': [2, 1]})

     
     
    if cm.ndim == 2 and cm.size > 0:  
         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
     yticklabels=class_names, ax=ax1, annot_kws={"size": 16}, cbar=False)
    else:
         ax1.text(0.5, 0.5, "No data to plot CM", horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
         ax1.set_xticks([])
         ax1.set_yticks([])


    ax1.set_ylabel('True Label', fontsize=14)
    ax1.set_xlabel('Predicted Label', fontsize=14)
    ax1.set_title(f"Deepfake Detection Confusion Matrix", fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=12)

     
    metrics_data = [
        ["True Negatives (REAL)", tn],  
        ["False Positives (FAKE as REAL)", fp],
        ["False Negatives (REAL as FAKE)", fn],
        ["True Positives (FAKE)", tp],  
        ["", ""],  
        ["Overall Accuracy", f"{accuracy:.4f}"],
        ["FAKE Precision", f"{precision:.4f}"],  
        ["FAKE Recall (Sensitivity)", f"{recall:.4f}"],  
        ["REAL Recall (Specificity)", f"{specificity:.4f}"],  
        ["FAKE F1 Score", f"{f1:.4f}"]
    ]

     
    ax2.axis('off')  

    table = ax2.table(cellText=metrics_data, loc='center', cellLoc='left',
     colWidths=[0.6, 0.4], edges='open')  
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)  

    ax2.set_title("Performance Metrics", fontsize=16)

    plt.suptitle(f"{title}", fontsize=18, y=0.98)  

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  
    plt.savefig(f"confusion_matrix_{encoder_type}.png", dpi=150, bbox_inches='tight')
    plt.show()


 
def run_video_evaluation(video_dir, ckpt_path_encoder, ckpt_path_classifier, args, device):
    """
    Runs video-level evaluation by extracting frame features, classifying frames,
    and aggregating frame-level predictions using a trained encoder and classifier.
    Ground truth is determined from video filenames (_fake suffix).
    """
    logging.info(f"Starting Video-level Evaluation.")
    logging.info(f"Video directory: {video_dir}")
    logging.info(f"Encoder Checkpoint: {ckpt_path_encoder}")
    logging.info(f"Classifier Checkpoint: {ckpt_path_classifier}")


    if not os.path.exists(video_dir):
        logging.error(f"Video directory not found: {video_dir}. Exiting.")
        return

     
    try:
        ecl_model, backbone_type = load_encoder(ckpt_path_encoder, device)
        logging.info(f"Loaded ECL model with backbone {backbone_type} from {ckpt_path_encoder}.")
    except Exception as e:
        logging.error(f"Failed to load ECL model for video evaluation: {e}. Exiting.")
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
                logging.error(f"Could not automatically determine feature dimension from model output type: {type(dummy_output)}. Expected tuple with a tensor as first element.")
                raise ValueError("Could not determine feature dimension from model output output[0].")

    except Exception as e:
        logging.error(f"Error determining feature dimension: {e}. Check model forward method and input size ({args.frame_size}). Exiting.")
        return


     
     
    try:
        num_classes = 2  
         
         
         
         
        classifier = FeatureClassifier(feature_dimension, num_classes, hidden_dim1=args.classifier_hidden_dim1, hidden_dim2=args.classifier_hidden_dim2).to(device)
        logging.info(f"Instantiated MLP Classifier with input dimension {feature_dimension}, hidden dimension {args.classifier_hidden_dim1}, hidden dimension {args.classifier_hidden_dim2}, and {num_classes} output classes.")


         
         
        checkpoint = torch.load(ckpt_path_classifier, map_location=device)

         
        if 'model_state_dict' not in checkpoint:
             raise ValueError(f"Classifier checkpoint '{ckpt_path_classifier}' does not contain 'model_state_dict' key.")

        classifier_state_dict = checkpoint['model_state_dict']  

         
        classifier_state_dict_cleaned = OrderedDict()
        for k, v in classifier_state_dict.items():  
             name = k[7:] if k.startswith('module.') else k
             classifier_state_dict_cleaned[name] = v

         
        classifier.load_state_dict(classifier_state_dict_cleaned)
        logging.info(f"Successfully loaded Classifier from {ckpt_path_classifier}")
         

    except FileNotFoundError:
         logging.error(f"Classifier checkpoint not found at {ckpt_path_classifier}. Exiting.")
         return
    except ValueError as ve:  
         logging.error(f"Classifier checkpoint format error: {ve}. Exiting.")
         return
    except Exception as e:
        logging.error(f"Failed to load Classifier checkpoint: {e}. Exiting.")
         
        try:
             
             
            loaded_obj = torch.load(ckpt_path_classifier, map_location=device)
            loaded_obj_type = type(loaded_obj)
            logging.error(f"Type of object loaded from checkpoint: {loaded_obj_type}")
            if loaded_obj_type == str:
                 logging.error(f"Content of loaded string: {loaded_obj[:200]}...")  
        except Exception:
            logging.error("Could not determine type or content of object loaded from checkpoint after initial load failure.")
        logging.error(f"Error details: {e}. Exiting.")
        raise  


     
    transform = get_transforms(name="eval", size=args.frame_size[0], backbone=backbone_type, norm="imagenet")
    logging.info(f"Using eval transform for backbone {backbone_type} with size {args.frame_size}.")


    video_paths = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))

    if not video_paths:
        logging.warning(f"No .mp4 videos found in {video_dir}")
        return

    all_video_names = []
    all_video_true_labels = []
    all_video_preds = []  

    print("\nStarting Video-level Classification and Evaluation...")

    for path in tqdm(video_paths, desc="Processing Videos"):
        video_name = os.path.basename(path)

         
         
        true_label = 0 if "_fake" in video_name.lower() else 1

         
        frame_predictions = extract_frame_predictions(path, ecl_model, classifier, transform, device, args.frame_limit)

        if not frame_predictions:
            logging.warning(f"No valid frame predictions extracted for video {video_name}. Skipping video.")
            continue

         
         
        predicted_label = aggregate_video_prediction(frame_predictions, aggregation_method=args.aggregation_method)

        if predicted_label == -1:  
             logging.warning(f"Video-level prediction failed for {video_name} after aggregation. Skipping video.")
             continue

        all_video_names.append(video_name)
        all_video_true_labels.append(true_label)
        all_video_preds.append(predicted_label)


    if not all_video_preds:
        logging.error("No video predictions collected after processing. Cannot compute metrics.")
         
        cm = np.array([[0,0],[0,0]])
        class_names = ["FAKE", "REAL"]
        all_video_true_labels = []  
        all_video_preds = []
    else:
         
        logging.info("\nComputing Video-level Metrics...")
         
        cm = confusion_matrix(all_video_true_labels, all_video_preds, labels=[0, 1])
         
        class_names = ["FAKE", "REAL"]  


     
     
    plot_confusion_matrix(cm, class_names, f"Video-level Detection Results ({backbone_type})", backbone_type + "_video")

    if all_video_true_labels and all_video_preds:  
        print("\nVideo-level Classification Report:")
         
        print(classification_report(all_video_true_labels, all_video_preds, target_names=class_names, zero_division=0))

        overall_accuracy = accuracy_score(all_video_true_labels, all_video_preds)
        logging.info(f"Video-level Evaluation Overall Accuracy: {overall_accuracy:.4f}")
    else:
         logging.warning("Skipping classification report and accuracy calculation due to lack of data.")


 
def plot_confusion_matrix(cm, class_names, title, encoder_type):
    """
    Plots the confusion matrix and displays performance metrics.
    Args:
        cm (np.ndarray): Confusion matrix (2x2).
        class_names (list): List of class names (e.g., ["REAL", "FAKE"]).
        title (str): Title for the plot.
        encoder_type (str): Identifier for saving the file.
    """
     
     
     
    tn, fp, fn, tp = 0, 0, 0, 0  
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.size == 1:  
        val = cm[0,0]
         
         
        if class_names[0] == "REAL":  
              
              
              
              
              
             logging.warning("1x1 CM - Assuming it corresponds to class 0 ('REAL'). Metrics might be inaccurate.")
             tn = val  
        elif class_names[0] == "FAKE":  
               
               
              logging.warning("1x1 CM - Assuming it corresponds to class 0 ('REAL'). Metrics might be inaccurate.")
              tn = val  
         
        elif len(class_names) > 1 and class_names[1] == "FAKE":
             logging.warning("1x1 CM - Assuming it corresponds to class 1 ('FAKE'). Metrics might be inaccurate.")
             tp = val  
        else:
             logging.error(f"Could not infer class for single value CM. Defaulting metrics to 0.")


    else:  
        logging.error(f"Unexpected confusion matrix shape: {cm.shape}. Metrics will be calculated based on available data but might be incorrect.")
         

    total_samples = tp + tn + fp + fn
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
     
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0  
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0   
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0  

     
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': [2, 1]})

     
     
    if cm.ndim == 2 and cm.size > 0:  
         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
     yticklabels=class_names, ax=ax1, annot_kws={"size": 16}, cbar=False)
    else:
         ax1.text(0.5, 0.5, "No data to plot CM", horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
         ax1.set_xticks([])
         ax1.set_yticks([])


    ax1.set_ylabel('True Label', fontsize=14)
    ax1.set_xlabel('Predicted Label', fontsize=14)
    ax1.set_title(f"Deepfake Detection Confusion Matrix", fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=12)

     
    metrics_data = [
        ["True Negatives (REAL)", tn],  
        ["False Positives (FAKE as REAL)", fp],
        ["False Negatives (REAL as FAKE)", fn],
        ["True Positives (FAKE)", tp],  
        ["", ""],  
        ["Overall Accuracy", f"{accuracy:.4f}"],
        ["FAKE Precision", f"{precision:.4f}"],  
        ["FAKE Recall (Sensitivity)", f"{recall:.4f}"],  
        ["REAL Recall (Specificity)", f"{specificity:.4f}"],  
        ["FAKE F1 Score", f"{f1:.4f}"]
    ]

     
    ax2.axis('off')  

    table = ax2.table(cellText=metrics_data, loc='center', cellLoc='left',
     colWidths=[0.6, 0.4], edges='open')  
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)  

    ax2.set_title("Performance Metrics", fontsize=16)

    plt.suptitle(f"{title}", fontsize=18, y=0.98)  

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  
    plt.savefig(f"confusion_matrix_{encoder_type}.png", dpi=150, bbox_inches='tight')
    plt.show()


 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video-level evaluation using trained encoder and classifier.')
     
    parser.add_argument('--video_dir', type=str, default="/home/teaching/deepfake/UADFV_videos", help='Path to video folder for evaluation')
     
    parser.add_argument('--ckpt_path_encoder', type=str, default="/home/teaching/deepfake/project/files/Unsupervised_DF_Detection/save/ECL_convnext_base/FF_frames_models/20250512_012333/ckpt_epoch_016.pth", help='Checkpoint path of the trained ECL encoder model (from main.py)')
     
    parser.add_argument('--ckpt_path_classifier', type=str, default="/home/teaching/deepfake/project/files/Unsupervised_DF_Detection/trained_classifier_runs/20250512_175543/classifier_convnext_base_FF_epoch_010.pth", help='Path to the trained linear classifier state_dict (from train_classifier.py)')
     
    parser.add_argument('--classifier_hidden_dim1', type=int, default=512, help='Dimension of the hidden layer in the MLP classifier (must match training)')
    parser.add_argument('--classifier_hidden_dim2', type=int, default=128, help='Dimension of the second hidden layer in the MLP classifier (must match training)')

     
    parser.add_argument('--frame_size', type=int, nargs=2, default=[299, 299], help='Frame size for preprocessing (height width)')
     
    parser.add_argument('--frame_limit', type=int, default=32, help='Maximum number of frames to process per video')
     
    parser.add_argument('--aggregation_method', type=str, default='majority_vote', choices=['majority_vote', 'average_prob'], help='Method for aggregating frame predictions')


    args = parser.parse_args()

     
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

     
    args.frame_size = tuple(args.frame_size)

     
    run_video_evaluation(
        args.video_dir,
        args.ckpt_path_encoder,
        args.ckpt_path_classifier,
        args,
        DEVICE
    )
