import os
import sys
import argparse
import dlib
import cv2
import numpy as np
from tqdm import tqdm
import json
import glob
import re
from sklearn.cluster import KMeans
from lib.vaf_util import get_crops_landmarks, classify_eyes_open, classify_mouth_open, generate_convex_mask, new_size
from lib.vaf_util import generate_law_filters, preprocess_image, filter_image, compute_energy
from lib.vaf_util import MOUTH_LM, LEYE_LM, REYE_LM


def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unlabeled_data_path', type=str, help='The path of unlabeled data.', default="/home/teaching/deepfake/CelebDF_frames_v2")
    parser.add_argument('--face_detector_path', type=str, help='The path of shape_predictor_68_face_landmarks.dat.', default="/home/teaching/deepfake/project/files/Unsupervised_DF_Detection/shape_predictor_68_face_landmarks.dat")
    parser.add_argument('--output_path', type=str, help='The path of output files.', default="/home/teaching/deepfake/project/files/Unsupervised_DF_Detection/output_gpu/CelebDF_v2_labels")
    args = parser.parse_args()
    return args


def load_face_detector(face_detector_path):
    if not os.path.isfile(face_detector_path):
        print("Could not find shape_predictor_68_face_landmarks.dat")
        sys.exit()
    face_detector = dlib.get_frontal_face_detector()
    sp68 = dlib.shape_predictor(face_detector_path)
    return face_detector, sp68


LAW_MASKS = generate_law_filters()


def extract_features_mask(img, mask):
    """Computes law texture features for masked area of image."""
    preprocessed_img = preprocess_image(img, size=15)
    law_images = filter_image(preprocessed_img, LAW_MASKS)
    law_energy = compute_energy(law_images, 10)

    energy_features_list = []
    for _, energy in law_energy.items():
         
        energy_masked = energy[np.where(mask != 0)]
        energy_feature = np.mean(energy_masked, dtype=np.float32)
        energy_features_list.append(energy_feature)

    return energy_features_list


def extract_features_eyes(landmarks, face_crop, scale=256):
     
    l_eye_marks = landmarks[LEYE_LM]
    r_eye_marks = landmarks[REYE_LM]
    l_eye_mask = generate_convex_mask(face_crop[..., 0].shape, l_eye_marks[..., 0], l_eye_marks[..., 1])
    r_eye_mask = generate_convex_mask(face_crop[..., 0].shape, r_eye_marks[..., 0], r_eye_marks[..., 1])
    eye_mask = np.logical_or(l_eye_mask, r_eye_mask)
    eye_mask = eye_mask.astype(dtype=np.uint8)
    if np.sum(eye_mask) < 10:
        return None

    energy_features = extract_features_mask(face_crop, eye_mask)

    return energy_features


def extract_features_mouth(landmarks, face_crop, scale=200):
    mouth_marks = landmarks[MOUTH_LM]
    mouth_mask = generate_convex_mask(face_crop[..., 0].shape, mouth_marks[..., 0], mouth_marks[..., 1])
    mouth_mask = mouth_mask.astype(dtype=np.uint8)

     
    out_size = new_size(face_crop.shape[1], face_crop.shape[0], large_dim=scale)
    mouth_mask = cv2.resize(mouth_mask, (out_size[1], out_size[0]), interpolation=cv2.INTER_NEAREST)
    face_crop = cv2.resize(face_crop, (out_size[1], out_size[0]), interpolation=cv2.INTER_LINEAR)

     
    mouth_idxs = np.where(mouth_mask != 0)
    img_intensity = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
    mouth_values = img_intensity[mouth_idxs]

     
    mouth_values = mouth_values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(mouth_values)
    kmeans_pred = kmeans.predict(mouth_values)

    mouth_idxs_array = np.asarray(mouth_idxs)
    bright_cluster = np.argmax(kmeans.cluster_centers_)
    mouth_idx_select = mouth_idxs_array[..., np.where(kmeans_pred == bright_cluster)[0]]

    teeth_mask = np.zeros_like(mouth_mask)
    teeth_mask[mouth_idx_select[0], mouth_idx_select[1]] = 1
    teeth_mask = np.logical_and(mouth_mask, teeth_mask)

    num_pix = np.sum(teeth_mask)
    total_pix = np.sum(mouth_mask)

     
    if total_pix <= 0:
        return None
    percentage = num_pix / float(total_pix)
    if percentage < 0.01:
        return None

    energy_features = extract_features_mask(face_crop, teeth_mask)

    return energy_features


def extract_vaf(face_crop_list, landmarks_list, scale=256):
    flag = True
    feature_vector = None
    landmarks = landmarks_list[0].copy()
    face_crop = face_crop_list[0]

    mouth_open = classify_mouth_open(landmarks)
    eyes_open = classify_eyes_open(landmarks)
    if eyes_open and mouth_open:
        features_eyes = extract_features_eyes(landmarks, face_crop, scale=scale)
        features_mouth = extract_features_mouth(landmarks, face_crop, scale=scale)
        if features_eyes is not None and features_mouth is not None:
            feature_vector = np.concatenate((features_eyes, features_mouth))
    else:
        flag = False

    mouth_open = classify_mouth_open(landmarks)
    eyes_open = classify_eyes_open(landmarks)
    if not (eyes_open and mouth_open):
        flag = False

    valid_segmentation = feature_vector is not None

    return feature_vector, valid_segmentation, flag


def main():
    args = args_func()
    unlabeled_data_path = args.unlabeled_data_path
    face_detector_path = args.face_detector_path
    output_path = args.output_path

     
    unlabeled_data_list = glob.glob(os.path.join(unlabeled_data_path, '*'))
    image_extensions = ['.jpg', '.jpeg', '.png']
    unlabeled_data_list = [file for file in unlabeled_data_list if
                           os.path.splitext(file.lower())[1] in image_extensions]
    unlabeled_data_name_list = list(map(lambda x: os.path.basename(x), unlabeled_data_list))

    face_detector, sp68 = load_face_detector(face_detector_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
     
    total_images = len(unlabeled_data_list)
    error_counters = {
        "total_images": total_images,
        "image_load_errors": 0,
        "face_detection_errors": 0,
        "closed_eyes_or_mouth": 0,
        "eye_segmentation_failures": 0,
        "mouth_segmentation_failures": 0,
        "successful_extractions": 0
    }
    
     
    error_logs = {
        "image_load_errors": [],
        "face_detection_errors": [], 
        "closed_eyes_or_mouth": [],
        "eye_segmentation_failures": [],
        "mouth_segmentation_failures": []
    }

     
    vaf_list = []
    img_name_list = []
    processed_images = set()   
    
    with tqdm(total=total_images, desc='Extracting VAF', unit='file') as pbar:
        for process_data in unlabeled_data_list:
            img_name = os.path.basename(process_data)
            
             
            img = cv2.imread(process_data)
            if img is None or img is False:
                error_counters["image_load_errors"] += 1
                error_logs["image_load_errors"].append(img_name)
                print(f"Could not open image file: {process_data}")
                pbar.update(1)
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
             
            face_crops, landmarks = get_crops_landmarks(face_detector, sp68, img)
            if len(landmarks) == 0:
                error_counters["face_detection_errors"] += 1
                error_logs["face_detection_errors"].append(img_name)
                print(f"Could not find the face in image file: {process_data}")
                pbar.update(1)
                continue
            
             
            vaf, valid_seg, flag = extract_vaf(face_crops, landmarks, scale=256)
            
             
            if not flag:
                error_counters["closed_eyes_or_mouth"] += 1
                error_logs["closed_eyes_or_mouth"].append(img_name)
                pbar.update(1)
                continue
            
            if not valid_seg:
                 
                 
                mouth_open = classify_mouth_open(landmarks[0])
                eyes_open = classify_eyes_open(landmarks[0])
                
                if eyes_open and mouth_open:
                    features_eyes = extract_features_eyes(landmarks[0], face_crops[0], scale=256)
                    features_mouth = extract_features_mouth(landmarks[0], face_crops[0], scale=256)
                    
                    if features_eyes is None:
                        error_counters["eye_segmentation_failures"] += 1
                        error_logs["eye_segmentation_failures"].append(img_name)
                    
                    if features_mouth is None:
                        error_counters["mouth_segmentation_failures"] += 1
                        error_logs["mouth_segmentation_failures"].append(img_name)
                
                pbar.update(1)
                continue
            
             
            error_counters["successful_extractions"] += 1
            vaf_list.append(vaf)
            img_name_list.append(img_name)
            processed_images.add(img_name)   
            pbar.update(1)

     
    if len(vaf_list) > 0:
        kmeans = KMeans(n_clusters=2, init='k-means++')
        cluster_label = kmeans.fit_predict(vaf_list)

         
        data = list(zip(img_name_list, cluster_label))

        video_label_counts = {}

        for image, label in data:
            video_name = image.split("_")[0]

            if video_name not in video_label_counts:
                video_label_counts[video_name] = {}

            if label not in video_label_counts[video_name]:
                video_label_counts[video_name][label] = 1
            else:
                video_label_counts[video_name][label] += 1

        video_pseudo_labels = {video: max(label_counts, key=label_counts.get) for video, label_counts in
                            video_label_counts.items()}
        pseudo_video = [{"video name": video, "video label": str(label)} for video, label in video_pseudo_labels.items()]

         
        labeled_but_not_processed = 0
        processed_but_not_labeled = 0
        labeled_files = 0
        
        image_pseudo_label_dict = {}

        for filename in os.listdir(unlabeled_data_path):
            if not os.path.isfile(os.path.join(unlabeled_data_path, filename)):
                continue
                
            if not any(filename.lower().endswith(ext) for ext in image_extensions):
                continue
                
            video_name = filename.split('_')[0]
            label = None
            for video in pseudo_video:
                if video['video name'] == video_name:
                    label = video['video label']
                    break
            
            if label is not None:
                image_pseudo_label_dict[filename] = label
                labeled_files += 1
                
                if filename not in processed_images:
                    labeled_but_not_processed += 1
            elif filename in processed_images:
                processed_but_not_labeled += 1

         
        output_file = 'image_pseudo_labels.json'
        with open(os.path.join(output_path, output_file), 'w') as f:
            json.dump(image_pseudo_label_dict, f, indent=4)

         
        error_stats = {
            "error_counters": error_counters,
            "labeled_files": labeled_files,
            "labeled_but_not_processed": labeled_but_not_processed,
            "processed_but_not_labeled": processed_but_not_labeled,
            "percentage_successful": (error_counters["successful_extractions"] / error_counters["total_images"]) * 100
        }
        
        with open(os.path.join(output_path, 'error_statistics.json'), 'w') as f:
            json.dump(error_stats, f, indent=4)
            
         
        with open(os.path.join(output_path, 'error_logs.json'), 'w') as f:
            json.dump(error_logs, f, indent=4)

         
        print("\n--- Processing Summary ---")
        print(f"Total images: {error_counters['total_images']}")
        print(f"Successfully processed: {error_counters['successful_extractions']} ({(error_counters['successful_extractions'] / error_counters['total_images']) * 100:.2f}%)")
        print("\n--- Error Distribution ---")
        print(f"Image load errors: {error_counters['image_load_errors']} ({(error_counters['image_load_errors'] / error_counters['total_images']) * 100:.2f}%)")
        print(f"Face detection errors: {error_counters['face_detection_errors']} ({(error_counters['face_detection_errors'] / error_counters['total_images']) * 100:.2f}%)")
        print(f"Closed eyes or mouth: {error_counters['closed_eyes_or_mouth']} ({(error_counters['closed_eyes_or_mouth'] / error_counters['total_images']) * 100:.2f}%)")
        print(f"Eye segmentation failures: {error_counters['eye_segmentation_failures']} ({(error_counters['eye_segmentation_failures'] / error_counters['total_images']) * 100:.2f}%)")
        print(f"Mouth segmentation failures: {error_counters['mouth_segmentation_failures']} ({(error_counters['mouth_segmentation_failures'] / error_counters['total_images']) * 100:.2f}%)")
        print("\n--- Labeling Statistics ---")
        print(f"Total files labeled: {labeled_files}")
        print(f"Files labeled despite not being processed: {labeled_but_not_processed}")
        print(f"Files processed but not labeled: {processed_but_not_labeled}")
        print(f"The data has successfully been written to the file: {os.path.join(output_path, output_file)}")
    else:
        print("No valid VAF features extracted. Cannot perform clustering.")


if __name__ == '__main__':
    main()