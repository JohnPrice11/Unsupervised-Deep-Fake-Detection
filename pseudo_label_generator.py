# gpu_accelerated_script.py
import os
import sys
import argparse
import dlib # Still needed for CPU face detection
import cv2
import numpy as np
from tqdm import tqdm
import json
import glob
import re
import time
import traceback # Import traceback for detailed error printing

# Import GPU accelerated tools
import cupy as cp
try:
    from cuml.cluster import KMeans as cuKMeans # GPU KMeans
    print("cuML found.")
except ImportError:
    print("cuML KMeans not found. Install RAPIDS cuML for GPU acceleration.")
    print("Falling back to scikit-learn KMeans (CPU).")
    # Ensure fallback is actually sklearn's KMeans if cuML fails
    from sklearn.cluster import KMeans as cuKMeans # Fallback

# Import CPU multiprocessing tools
from multiprocessing import Pool, cpu_count, Manager
from functools import partial

# Import our custom GPU VAF utilities
try:
    # Assuming gpu_vaf_util.py is in the same directory or python path
    from gpu_vaf_util import (get_crops_landmarks_cpu,
                              extract_vaf_gpu_batch)
    print("gpu_vaf_util module loaded.")
except ImportError:
    print("ERROR: Could not import 'gpu_vaf_util.py'. Make sure it's saved correctly.")
    sys.exit(1)

def args_func():
    parser = argparse.ArgumentParser(description="GPU Accelerated Deepfake Detection VAF Extraction")
    parser.add_argument('--unlabeled_data_path', type=str, help='The path of unlabeled data (directory containing images).', default="/home/teaching/deepfake/UADFV_frames_new") # CHANGE HERE FOR DATASET CHANGE
    parser.add_argument('--face_detector_path', type=str, help='The path of shape_predictor_68_face_landmarks.dat.', default="/home/teaching/deepfake/project/files/Unsupervised_DF_Detection/shape_predictor_68_face_landmarks.dat")
    parser.add_argument('--output_path', type=str, help='The path for output files (JSON).', default="/home/teaching/deepfake/project/files/Unsupervised_DF_Detection/output_gpu")
    parser.add_argument('--batch_size', type=int, default=64, help='Number of images to process in GPU VAF batch.') # Increased default
    parser.add_argument('--num_workers', type=int, default=max(1, cpu_count() // 2), help='Number of CPU processes for face detection.')
    parser.add_argument('--vaf_scale', type=int, default=256, help='Scale parameter used in VAF mouth processing.')
    args = parser.parse_args()

    # Validate paths
    if not os.path.isdir(args.unlabeled_data_path):
        print(f"Error: Unlabeled data path not found or not a directory: {args.unlabeled_data_path}")
        sys.exit(1)
    if not os.path.isfile(args.face_detector_path):
        print(f"Error: Face detector model not found: {args.face_detector_path}")
        sys.exit(1)

    return args

# --- Multiprocessing Worker Setup ---

face_detector_global = None
sp68_global = None

def init_worker(detector_path):
    """Initializes dlib models in each worker process."""
    global face_detector_global, sp68_global
    worker_pid = os.getpid()
    print(f"[Worker {worker_pid}] Initializing...") # DEBUG PRINT
    if not os.path.isfile(detector_path):
         print(f"[Worker {worker_pid}] ERROR: Could not find shape_predictor_68_face_landmarks.dat at {detector_path}")
         face_detector_global = None
         sp68_global = None
         return
    try:
        # Load models into worker-specific globals
        print(f"[Worker {worker_pid}] Loading dlib models...") # DEBUG PRINT
        face_detector_global = dlib.get_frontal_face_detector()
        sp68_global = dlib.shape_predictor(detector_path)
        print(f"[Worker {worker_pid}] Dlib models loaded successfully.") # DEBUG PRINT
    except Exception as e:
        print(f"[Worker {worker_pid}] FATAL ERROR during dlib model loading: {e}") # DEBUG PRINT
        print(traceback.format_exc()) # Print full traceback
        face_detector_global = None
        sp68_global = None

# Function to be run by each CPU process pool worker
def process_single_image_cpu(image_path):
    """Reads image, detects face/landmarks using worker's dlib models."""
    worker_pid = os.getpid()
    # print(f"[Worker {worker_pid}] Processing: {os.path.basename(image_path)}") # DEBUG PRINT (can be very verbose)
    global face_detector_global, sp68_global
    if face_detector_global is None or sp68_global is None:
        # print(f"[Worker {worker_pid}] Skipping {os.path.basename(image_path)} - models not loaded.") # DEBUG PRINT
        return os.path.basename(image_path), None, None # Return name, None landmarks, None face_crops

    img_name = os.path.basename(image_path)
    try:
        # Read image using OpenCV
        # print(f"[Worker {worker_pid}] Reading image: {img_name}") # DEBUG PRINT
        img = cv2.imread(image_path)
        if img is None:
            print(f"[Worker {worker_pid}] WARNING: Could not read image file: {image_path}") # DEBUG PRINT
            return img_name, None, None

        # Convert to RGB for dlib
        # print(f"[Worker {worker_pid}] Converting to RGB: {img_name}") # DEBUG PRINT
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Use the imported CPU function with worker's models
        # print(f"[Worker {worker_pid}] Detecting faces/landmarks: {img_name}") # DEBUG PRINT
        face_crops, landmarks = get_crops_landmarks_cpu(face_detector_global, sp68_global, img_rgb)
        # print(f"[Worker {worker_pid}] Detection complete for: {img_name} - Found {len(landmarks)} faces.") # DEBUG PRINT


        if not landmarks: # Check if list is empty
            # print(f"[Worker {worker_pid}] No face found in {img_name}") # DEBUG PRINT
            return img_name, None, None

        # Only return the first detected face's data for simplicity
        # print(f"[Worker {worker_pid}] Returning results for {img_name}") # DEBUG PRINT
        return img_name, landmarks[0], face_crops[0]

    except Exception as e:
         print(f"[Worker {worker_pid}] ERROR processing {img_name}: {e}") # DEBUG PRINT
         print(traceback.format_exc()) # Print full traceback
         return img_name, None, None

# --- Main Execution Logic ---

def main():
    args = args_func()
    start_time = time.time()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # --- 1. Load Image Paths ---
    print("Scanning for images...")
    unlabeled_data_list = glob.glob(os.path.join(args.unlabeled_data_path, '*'))
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'] # Add more if needed
    unlabeled_data_list = [file for file in unlabeled_data_list if
                           os.path.splitext(file.lower())[1] in image_extensions]
    total_files = len(unlabeled_data_list)
    print(f"Found {total_files} image files.")
    if total_files == 0:
        print("No images found in the specified directory. Exiting.")
        return
        
    # --- DEBUG: Process a small subset first ---
    # print("DEBUG: Processing only the first 100 images for testing.")
    # unlabeled_data_list = unlabeled_data_list[:100]
    # total_files = len(unlabeled_data_list)
    # --- END DEBUG ---


    # --- 2. CPU Parallel Processing for Face Detection ---
    print(f"Starting face detection using {args.num_workers} CPU workers...")
    cpu_start_time = time.time()

    # Create a pool of worker processes, initializing dlib in each
    pool_initializer = partial(init_worker, args.face_detector_path)

    cpu_results = [] # Store results: (img_name, landmarks, face_crop)
    skipped_no_face = 0

    # Using imap_unordered for potentially better performance as results come in
    # Chunksize helps reduce overhead for very large lists
    # Reduced chunksize calculation for potentially faster feedback on smaller batches
    chunksize = max(1, min(50, total_files // (args.num_workers * 4) + 1)) 
    print(f"Using chunksize: {chunksize}") # DEBUG PRINT

    try:
        # Set maxtasksperchild=100 to restart workers periodically, potentially freeing resources
        with Pool(processes=args.num_workers, initializer=pool_initializer, maxtasksperchild=100) as pool:
            print("Multiprocessing Pool created. Starting imap_unordered...") # DEBUG PRINT
            with tqdm(total=total_files, desc='Detecting Faces (CPU)', unit='file') as pbar:
                processed_counter = 0 # Counter for debug prints
                for result in pool.imap_unordered(process_single_image_cpu, unlabeled_data_list, chunksize=chunksize):
                    # --- DEBUG PRINT every N results ---
                    processed_counter += 1
                    if processed_counter % (chunksize * args.num_workers) == 0: # Print roughly every full wave of chunks
                         print(f"\n[Main Process] Received result #{processed_counter}/{total_files}")
                    # --- END DEBUG ---
                         
                    if result is not None:
                        img_name, landmarks, face_crop = result
                        if landmarks is not None and face_crop is not None:
                            cpu_results.append(result)
                        else:
                            skipped_no_face +=1
                    else:
                         # This case should ideally not happen if process_single_image_cpu always returns tuple
                         print(f"[Main Process] WARNING: Received None result from worker.")
                         skipped_no_face += 1
                         
                    pbar.update(1)
            print("\n[Main Process] Finished iterating through pool results.") # DEBUG PRINT
    except Exception as e:
        print(f"\n[Main Process] FATAL ERROR during multiprocessing pool execution: {e}") # DEBUG PRINT
        print(traceback.format_exc()) # Print full traceback
        # Decide how to handle: exit, continue with partial results?
        # For now, print error and continue if cpu_results has data.
        if not cpu_results:
             print("Exiting due to error and no successful detections.")
             return


    cpu_end_time = time.time()
    print(f"Face detection finished in {cpu_end_time - cpu_start_time:.2f} seconds.")
    print(f"Found faces in {len(cpu_results)} images.")
    print(f"Skipped {skipped_no_face} images (no face detected or error).")

    if not cpu_results:
        print("No faces found in any images. Cannot proceed to VAF extraction. Exiting.")
        return

    # Filter results - ensure lists are separated correctly
    valid_img_names = [item[0] for item in cpu_results]
    valid_landmarks = [item[1] for item in cpu_results]
    valid_face_crops = [item[2] for item in cpu_results]


    # --- 3. GPU Batch Processing for VAF Extraction ---
    print(f"Starting VAF extraction on GPU with batch size {args.batch_size}...")
    gpu_vaf_start_time = time.time()

    vaf_list_gpu_final = [] # Store final VAF vectors (NumPy arrays from GPU)
    img_name_list_final = [] # Corresponding image names
    processed_count = 0
    skipped_invalid_vaf = 0 # Count skips during VAF extraction step

    # Process in batches
    num_valid_faces = len(valid_img_names)
    with tqdm(total=num_valid_faces, desc='Extracting VAF (GPU)', unit='file') as pbar:
        for i in range(0, num_valid_faces, args.batch_size):
            # Slice the data for the current batch
            batch_end = min(i + args.batch_size, num_valid_faces)
            batch_names_cpu = valid_img_names[i:batch_end]
            batch_landmarks_cpu = valid_landmarks[i:batch_end]
            batch_crops_cpu = valid_face_crops[i:batch_end]

            # Call the GPU batch function
            # It returns lists of NumPy arrays and the original indices within the input *batch*
            # that were successfully processed.
            batch_vafs_cpu, valid_indices_in_batch = extract_vaf_gpu_batch(
                batch_crops_cpu, batch_landmarks_cpu, scale=args.vaf_scale
            )

            # Collect valid results from the batch
            if batch_vafs_cpu: # Check if the list is not empty
                for idx, vaf_vector in enumerate(batch_vafs_cpu):
                    original_batch_index = valid_indices_in_batch[idx]
                    vaf_list_gpu_final.append(vaf_vector) # Append the numpy array
                    img_name_list_final.append(batch_names_cpu[original_batch_index])
                    processed_count += 1

            # Update skip count based on how many in the batch didn't produce a VAF
            skipped_invalid_vaf += len(batch_names_cpu) - len(valid_indices_in_batch)

            pbar.update(len(batch_names_cpu)) # Update progress bar by the number processed in this batch attempt

            # Optional: Explicitly clear GPU memory pool periodically if memory issues arise
            # if i % (args.batch_size * 10) == 0: # e.g., every 10 batches
            #    cp.get_default_memory_pool().free_all_blocks()

    gpu_vaf_end_time = time.time()
    print(f"\nGPU VAF Extraction finished in {gpu_vaf_end_time - gpu_vaf_start_time:.2f} seconds.")
    print(f"Successfully extracted VAF for: {processed_count} images.")
    print(f"Skipped during VAF extraction (invalid state or error): {skipped_invalid_vaf}")

    if not vaf_list_gpu_final:
        print("No valid VAF features were extracted. Cannot perform clustering. Exiting.")
        return

    # --- 4. GPU KMeans Clustering ---
    print("Performing KMeans clustering...")
    cluster_start_time = time.time()

    # Ensure VAF list is a NumPy array first (should already be list of numpy arrays)
    try:
        vaf_array_np = np.array(vaf_list_gpu_final, dtype=np.float32)
        if vaf_array_np.ndim != 2:
             raise ValueError(f"VAF array has unexpected shape: {vaf_array_np.shape}")
        print(f"Shape of VAF features for clustering: {vaf_array_np.shape}")

        # Move data to GPU for cuML (if available and not fallback)
        # Check if cuKMeans is actually from cuML
        is_cuml = hasattr(cuKMeans, '__module__') and 'cuml' in cuKMeans.__module__
        if is_cuml:
             print("Using cuML KMeans on GPU.")
             vaf_array_gpu = cp.asarray(vaf_array_np) # Move data to GPU
             # Use cuML KMeans - use 'k-means||' for potentially better init
             kmeans_gpu = cuKMeans(n_clusters=2, init='k-means||', random_state=42, output_type='numpy', n_init=10)
             cluster_label = kmeans_gpu.fit_predict(vaf_array_gpu)
             # Clear GPU memory after clustering
             del vaf_array_gpu
             # cp.get_default_memory_pool().free_all_blocks()
        else:
             print("Using scikit-learn KMeans on CPU.")
             # Ensure cuKMeans is the sklearn one if fallback happened
             from sklearn.cluster import KMeans as SklearnKMeans
             kmeans_cpu = SklearnKMeans(n_clusters=2, init='k-means++', random_state=42, n_init=10)
             cluster_label = kmeans_cpu.fit_predict(vaf_array_np)

    except cp.cuda.memory.OutOfMemoryError:
        print("Error: Ran out of GPU memory during KMeans. Falling back to CPU KMeans.")
        # Fallback to CPU KMeans here
        try:
            from sklearn.cluster import KMeans as SklearnKMeans
            kmeans_cpu = SklearnKMeans(n_clusters=2, init='k-means++', random_state=42, n_init=10)
            cluster_label = kmeans_cpu.fit_predict(vaf_array_np) # Use the numpy array
        except Exception as e_cpu:
            print(f"CPU KMeans fallback also failed: {e_cpu}")
            print(traceback.format_exc())
            return # Exit if clustering fails completely

    except Exception as e:
        print(f"Error during KMeans clustering: {e}")
        print(traceback.format_exc())
        return # Exit if clustering fails

    cluster_end_time = time.time()
    print(f"Clustering finished in {cluster_end_time - cluster_start_time:.2f} seconds.")

    # --- 5. Assignment Pseudo-labels (Remains on CPU) ---
    print("Assigning pseudo-labels...")
    assign_start_time = time.time()

    # Combine image names (that have VAF) and their cluster labels
    data = list(zip(img_name_list_final, cluster_label))

    # Aggregate labels per video
    video_label_counts = {}
    for image, label in data:
        # Extract video name (handle names like videoID_frameNum.ext)
        # Improved regex to handle more cases, including potential hyphens or periods in video name
        match = re.match(r"([a-zA-Z0-9_.-]+?)(?:_\d+)?\.(?:jpg|jpeg|png|bmp|tiff)$", image, re.IGNORECASE)
        if match:
            video_name = match.group(1)
        else:
            # Fallback: split by first underscore if pattern fails
            video_name = image.split('_')[0]
            # print(f"Warning: Using fallback name extraction for {image} -> {video_name}")

        if video_name not in video_label_counts:
            # Initialize counts for both potential labels (0 and 1)
            video_label_counts[video_name] = {0: 0, 1: 0}

        # Increment count for the assigned label (convert label to int)
        try:
             label_int = int(label)
             if label_int in video_label_counts[video_name]:
                 video_label_counts[video_name][label_int] += 1
             else:
                 print(f"Warning: Unexpected cluster label '{label}' (type: {type(label)}) for image {image}. Skipping.")
        except ValueError:
             print(f"Warning: Cannot convert cluster label '{label}' to int for image {image}. Skipping.")


    # Determine majority label per video
    video_pseudo_labels = {}
    for video, label_counts in video_label_counts.items():
         count0 = label_counts.get(0, 0)
         count1 = label_counts.get(1, 0)

         if count0 == 0 and count1 == 0:
             # print(f"Warning: Video {video} had no valid labeled frames after clustering.")
             continue # Skip this video
         elif count0 >= count1: # Assign 0 in case of tie
              video_pseudo_labels[video] = 0
         else: # count1 > count0
              video_pseudo_labels[video] = 1


    # Map labels back to *all* original image files found initially
    image_pseudo_label_dict = {}
    processed_videos = set(video_pseudo_labels.keys())
    unlabeled_count = 0

    print(f"Mapping labels back to all {len(unlabeled_data_list)} original files...")
    for filename in unlabeled_data_list: # Iterate through original list of all images
        base_filename = os.path.basename(filename)
        # Use the same improved regex for consistency
        match = re.match(r"([a-zA-Z0-9_.-]+?)(?:_\d+)?\.(?:jpg|jpeg|png|bmp|tiff)$", base_filename, re.IGNORECASE)
        if match:
             video_name = match.group(1)
        else:
             video_name = base_filename.split('_')[0] # Fallback

        # Assign the determined video label, default to None (or -1) if video wasn't processed/labeled
        label = video_pseudo_labels.get(video_name, None)

        if label is not None:
            image_pseudo_label_dict[base_filename] = str(label) # Store label as string in JSON
        else:
            # Optional: Assign a default label like '-1' for images belonging to unprocessed videos
            # image_pseudo_label_dict[base_filename] = '-1'
            unlabeled_count +=1


    assign_end_time = time.time()
    print(f"Pseudo-label assignment finished in {assign_end_time - assign_start_time:.2f} seconds.")
    if unlabeled_count > 0:
        print(f"Note: {unlabeled_count} images belong to videos that could not be assigned a pseudo-label (e.g., no valid frames processed).")

    # --- 6. Save Results ---
    output_file = 'image_pseudo_labels_gpu.json'
    output_filepath = os.path.join(args.output_path, output_file)
    try:
        print(f"Saving results to {output_filepath}...")
        with open(output_filepath, 'w') as f:
            json.dump(image_pseudo_label_dict, f, indent=4)
        print(f"\nPseudo-labels successfully saved.")
    except Exception as e:
        print(f"\nError saving JSON output: {e}")
        print(traceback.format_exc())


    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")


if __name__ == '__main__':
    # Optional: Check GPU availability early
    try:
        dev = cp.cuda.runtime.getDevice()
        props = cp.cuda.runtime.getDeviceProperties(dev)
        print(f"CuPy found. Using GPU ID {dev}: {props['name'].decode()}") # Decode name for cleaner print
        # Check cuML again just before running main
        try:
            import cuml
        except ImportError:
            print("Warning: cuML (for KMeans) not found. Clustering will use CPU.")
            # No sys.exit here, allow fallback
    except Exception as e:
        print(f"Error initializing GPU via CuPy: {e}")
        print("Cannot proceed without GPU access for core computations. Exiting.")
        sys.exit(1)

    main()