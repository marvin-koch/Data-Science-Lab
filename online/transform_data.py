import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
# Keep rapidata import if needed elsewhere, but not directly used here
# from rapidata import RapidataClient
from huggingface_hub import HfApi
from datasets import Dataset, Features, Value, Image, concatenate_datasets
from tqdm import tqdm
import math
import os # Added os import
from copy import deepcopy

# Removed Rapidata client init and order fetching - results path will be passed


def load_json(file_path: str) -> dict:
    """Loads JSON data from a file."""
    print(f"Loading JSON from: {file_path}")
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        raise
    except json.JSONDecodeError:
        print(f"❌ Error: Could not decode JSON from {file_path}")
        raise
    except Exception as e:
        print(f"❌ An unexpected error occurred while loading {file_path}: {e}")
        raise

# Simplified map_images_to_prompts - expects mapping file content directly
def create_image_prompt_map(mapping_data: dict) -> dict:
    """
    Creates a mapping between images (relative paths) and prompts from mapping data.

    Args:
        mapping_data: The loaded content of the image_mapping.json file.

    Returns:
        A dictionary where keys are relative image paths (e.g., 'iter_0/image.png')
        and values are prompts.
    """
    mapping = {}
    try:
        for item in mapping_data["data"]:
            prompt = item["prompt"]
            for img_relative_path in item["images"]:
                # Key is the relative path as stored in the mapping file
                mapping[img_relative_path] = prompt
        print(f"Created image-prompt map with {len(mapping)} entries.")
        return mapping
    except (KeyError, TypeError) as e:
        print(f"❌ Error processing mapping data: {e}")
        return {}


# --- MODIFIED FUNCTION ---
# Updated to handle base path and relative image paths correctly
def transform_preference_data(
    rapidata_results: List[dict],
    image_prompt_mapping: Dict[str, str],
    iteration_num: int, # Added iteration_num
    base_image_dir: str = "images" # Base dir *without* iteration
) -> pd.DataFrame:
    """
    Transform list of Rapidata result dictionaries into DataFrame.

    Args:
        rapidata_results (list): List of dictionaries from Rapidata results['results'].
        image_prompt_mapping (dict): Dictionary mapping relative image paths (e.g., 'iter_0/img.png') to prompts.
        iteration_num (int): The current iteration number (e.g., 0, 1, ...).
        base_image_dir (str): The base directory where iteration folders ('iter_N') reside (e.g., "images").

    Returns:
        pd.DataFrame: Transformed data.
    """
    transformed_data = []
    metric_name = "Preference" # Assuming only preference for now
    iter_dir_name = f"iter_{iteration_num}" # Construct iteration directory name

    print("DEBUG: image_prompt_mapping keys sample:", list(image_prompt_mapping.keys())[:5])
    print("DEBUG: Rapidata results sample (keys):", [list(item.get('summedUserScores', {}).keys()) for item in rapidata_results[:2]])
    print("DEBUG: base_image_dir:", base_image_dir)
    print("DEBUG: iter_dir_name:", iter_dir_name)

    # Inner function to get the full, absolute path to the image
    def get_image_full_path(relative_image_path_with_iter: str) -> Path:
        # relative_image_path_with_iter should be like "iter_0/iter0_prompt0_img0_0000.png"
        try:
            # Combine base dir ("images") with the relative path ("iter_0/file.png")
            full_path = Path(base_image_dir) / relative_image_path_with_iter
            if not full_path.exists():
                 print(f"⚠️ Warning: Image file not found at resolved path: {full_path}")
            # Return the Path object directly
            return full_path
        except Exception as e:
            print(f"❌ Error resolving image path for {relative_image_path_with_iter}: {str(e)}")
            return None # Return None if path resolution fails

    print(f"Transforming {len(rapidata_results)} Rapidata result items...")
    for item_idx, item in enumerate(rapidata_results):
        try:
            row = {}

            # Get image FILENAMES from summedUserScores keys (e.g., "iter0_prompt0_img0_0000.png")
            rapidata_image_keys = list(item.get('summedUserScores', {}).keys())

            if not rapidata_image_keys or len(rapidata_image_keys) != 2:
                print(f"⚠️ Warning: Skipping item {item_idx}. Expected 2 image keys in 'summedUserScores', found {len(rapidata_image_keys)}: {rapidata_image_keys}")
                continue

            # CONSTRUCT the relative path keys expected by the mapping (e.g., "iter_0/iter0_prompt0_img0_0000.png")
            relative_path_key1 = f"{iter_dir_name}/{rapidata_image_keys[0]}"
            relative_path_key2 = f"{iter_dir_name}/{rapidata_image_keys[1]}"

            # Use the *constructed* relative path to find the prompt
            if relative_path_key1 not in image_prompt_mapping:
                 print(f"⚠️ Warning: Skipping item {item_idx}. Constructed path '{relative_path_key1}' not found in prompt mapping.")
                 # Optional: Print mapping keys for debugging: print("Available mapping keys:", image_prompt_mapping.keys())
                 continue
            # Check if both images belong to the same prompt (they should)
            if relative_path_key2 not in image_prompt_mapping:
                 print(f"⚠️ Warning: Skipping item {item_idx}. Constructed path '{relative_path_key2}' not found in prompt mapping.")
                 continue
            if image_prompt_mapping[relative_path_key1] != image_prompt_mapping[relative_path_key2]:
                 print(f"⚠️ Warning: Skipping item {item_idx}. Images '{relative_path_key1}' and '{relative_path_key2}' belong to different prompts.")
                 continue

            row['prompt'] = image_prompt_mapping[relative_path_key1]

            # Store constructed relative paths and resolve full paths using them
            row['relative_path1'] = relative_path_key1
            row['relative_path2'] = relative_path_key2
            row['image_full_path1'] = get_image_full_path(relative_path_key1)
            row['image_full_path2'] = get_image_full_path(relative_path_key2)

            # Skip if any image path couldn't be resolved
            if row['image_full_path1'] is None or row['image_full_path2'] is None:
                print(f"⚠️ Warning: Skipping item {item_idx} due to unresolved image paths.")
                continue

            # Extract model/source identifier from the original filename (Rapidata key)
            try:
                # Example: 'iter0_prompt0_img0_0000.png' -> 'iter0_prompt0_img0_0000'
                row['source1'] = Path(rapidata_image_keys[0]).stem
                row['source2'] = Path(rapidata_image_keys[1]).stem
            except Exception:
                 print(f"⚠️ Warning: Could not extract source/model from keys: {rapidata_image_keys[0]}, {rapidata_image_keys[1]}")
                 row['source1'] = 'unknown'
                 row['source2'] = 'unknown'


            # Get scores using the ORIGINAL keys from Rapidata results
            scores = item['summedUserScoresRatios']
            # Ensure the original keys exist in the scores dictionary
            if rapidata_image_keys[0] not in scores or rapidata_image_keys[1] not in scores:
                 print(f"⚠️ Warning: Skipping item {item_idx}. Score keys missing in 'summedUserScoresRatios'. Expected: {rapidata_image_keys}, Found: {scores.keys()}")
                 continue
            row[f'weighted_results1_{metric_name}'] = scores[rapidata_image_keys[0]]
            row[f'weighted_results2_{metric_name}'] = scores[rapidata_image_keys[1]]

            # Add detailed results as JSON string
            row[f'detailedResults_{metric_name}'] = json.dumps(item['detailedResults'])

            transformed_data.append(row)

        except KeyError as e:
            print(f"❌ Error processing item {item_idx}: Missing key {e}.\nItem data: {item}")
            continue
        except Exception as e:
            print(f"❌ Error processing item {item_idx}: {str(e)}\nItem data: {item}")
            continue

    if not transformed_data:
        print("⚠️ Warning: No data was successfully transformed.")
        return pd.DataFrame()

    df = pd.DataFrame(transformed_data)
    print(f"Transformation complete. Created DataFrame with {len(df)} rows.")

    # Define and order columns
    column_order = [
        'prompt',
        'relative_path1', # Keep relative path for info (e.g., iter_0/file.png)
        'relative_path2',
        'image_full_path1', # Use full path for loading image bytes
        'image_full_path2',
        'source1', # Identifier for image 1 source/model
        'source2', # Identifier for image 2 source/model
        f'weighted_results1_{metric_name}',
        f'weighted_results2_{metric_name}',
        f'detailedResults_{metric_name}'
    ]
    # Ensure all expected columns exist before reordering
    # Fill missing columns with None/NaN if necessary, though they should be populated
    for col in column_order:
        if col not in df.columns:
            df[col] = None # Or pd.NA

    df = df.reindex(columns=column_order)

    return df

# Updated to read images from the full path in the dataframe
def process_batch(batch_df):
    """Process a single batch of data, loading images from full local paths"""
    rows = []
    # Use tqdm on the iterable directly
    for _, row in tqdm(batch_df.iterrows(), total=batch_df.shape[0], desc="  Processing images in batch"):
        # Paths should be Path objects from transform function or None
        src_path1 = row['image_full_path1']
        src_path2 = row['image_full_path2']

        # Check existence again just before opening (and check if path is None)
        if not src_path1 or not isinstance(src_path1, Path) or not src_path1.exists():
            print(f"⚠️ Warning: Skipping row. Invalid or missing image path/file for image1: {src_path1}")
            continue
        if not src_path2 or not isinstance(src_path2, Path) or not src_path2.exists():
            print(f"⚠️ Warning: Skipping row. Invalid or missing image path/file for image2: {src_path2}")
            continue

        try:
            # Load image bytes using the full path
            with open(src_path1, 'rb') as f1:
                image1_bytes = f1.read()
            with open(src_path2, 'rb') as f2:
                image2_bytes = f2.read()

            dataset_row = {
                'prompt': row['prompt'],
                 # Pass bytes for Image() feature; path is for info if needed, use full path string
                'image1': {'bytes': image1_bytes, 'path': str(src_path1)},
                'image2': {'bytes': image2_bytes, 'path': str(src_path2)},
                'model1': row['source1'], # Use the extracted source/model identifier
                'model2': row['source2'],
                # Preference scores
                'weighted_results_image1_preference': row.get(f'weighted_results1_Preference'), # Use .get for safety
                'weighted_results_image2_preference': row.get(f'weighted_results2_Preference'),
                'detailed_results_preference': row.get(f'detailedResults_Preference'), # Already a JSON string
            }
            # Ensure float conversion where needed, handle potential None from .get
            if dataset_row['weighted_results_image1_preference'] is None: dataset_row['weighted_results_image1_preference'] = float('nan')
            if dataset_row['weighted_results_image2_preference'] is None: dataset_row['weighted_results_image2_preference'] = float('nan')
            if dataset_row['detailed_results_preference'] is None: dataset_row['detailed_results_preference'] = "{}" # Default empty JSON string

            rows.append(dataset_row)
        except FileNotFoundError as e:
             print(f"❌ Error: File not found during batch processing: {e}")
             continue # Skip this row
        except Exception as e:
            print(f"❌ Error processing row during batching (Prompt: {row.get('prompt', 'N/A')}): {e}")
            continue # Skip this row

    return rows


# Updated upload function signature
def upload_preference_dataset_to_hf(
    df,
    repo_id,
    token,
    batch_size=100 # Smaller batch size might be safer for memory with images
):
    """Upload preference dataset DataFrame to HuggingFace Hub."""

    if df.empty:
        print("❌ DataFrame is empty. Nothing to upload.")
        return

    if not token:
        print("❌ Hugging Face token not provided. Cannot upload.")
        # Optionally raise an error: raise ValueError("HF Token is required for upload")
        return

    print(f"\n--- Uploading Dataset to Hugging Face Hub ---")
    print(f"Target repository: {repo_id}")
    print(f"Processing {len(df)} rows for upload...")

    # Define features matching the process_batch output
    features = Features({
        'prompt': Value('string'),
        'image1': Image(decode=True), # Let datasets handle decoding from bytes
        'image2': Image(decode=True),
        'model1': Value('string'),
        'model2': Value('string'),
        'weighted_results_image1_preference': Value('float'),
        'weighted_results_image2_preference': Value('float'),
        'detailed_results_preference': Value('string'), # Keep as JSON string
    })

    # Initialize HF API and ensure repository exists
    try:
        api = HfApi(token=token)
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=True, # Keep private unless intended otherwise
            exist_ok=True
        )
        print(f"Repository {repo_id} is ready.")
    except Exception as e:
        print(f"❌ Error interacting with Hugging Face Hub API: {e}")
        return # Stop if repo creation fails

    # Process in batches
    num_batches = math.ceil(len(df) / batch_size)
    print(f"Processing in {num_batches} batches of size {batch_size}...")

    all_datasets = []
    upload_successful = True

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(df))

        print(f"\nProcessing batch {batch_idx + 1}/{num_batches} (rows {start_idx}-{end_idx-1})...")

        batch_df = df.iloc[start_idx:end_idx]
        batch_rows = process_batch(batch_df) # Returns list of dicts

        if batch_rows:  # Only create dataset if we have valid rows
            try:
                # Convert list of dicts directly to Dataset
                batch_dataset = Dataset.from_list(batch_rows, features=features)
                all_datasets.append(batch_dataset)
                print(f"  Successfully processed batch {batch_idx + 1}.")
                # Optionally clear memory, though Python's GC should handle it
                del batch_rows
                del batch_dataset
            except Exception as e:
                 print(f"❌ Error creating dataset from batch {batch_idx + 1}: {e}")
                 upload_successful = False
                 # Optionally break here if one batch failure should stop the whole process
                 # break
        else:
             print(f"  Batch {batch_idx + 1} resulted in no valid rows after processing.")


    if not all_datasets:
        print("❌ No valid data batches were processed. Nothing to upload.")
        return

    # Removed redundant check for upload_successful as we handle empty all_datasets
    # if not upload_successful:
    #     print("❌ Dataset creation failed for one or more batches. Aborting upload.")
    #     return

    # Combine all batch datasets if multiple exist
    if len(all_datasets) > 1:
        print("\nCombining all processed batches...")
        try:
            final_dataset = concatenate_datasets(all_datasets)
        except Exception as e:
             print(f"❌ Error concatenating datasets: {e}")
             return
    elif len(all_datasets) == 1:
        final_dataset = all_datasets[0]
    # No else needed, handled by the 'if not all_datasets' check above

    print(f"\nAttempting to upload {len(final_dataset)} rows to {repo_id}...")
    try:
        # Ensure token is passed if required, especially for private repos
        final_dataset.push_to_hub(
            repo_id,
            token=token,
            private=True # Ensure privacy setting matches create_repo
        )
        print("\n✅ Upload complete!")
        print(f"Dataset is available at: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"❌ Error uploading dataset to Hugging Face Hub: {e}")
        print("  Please check your token, network connection, and repository permissions.")


# --- MODIFIED FUNCTION ---
def transform_and_upload_for_iteration(
    rapidata_results_path: str,
    image_mapping_path: str,
    iteration_num: int,
    hf_repo_base_name: str, # e.g., "YourUser/SDXL-Feedback-Loop"
    hf_token: str,
    base_image_dir: str = "images", # Pass the *base* dir, e.g., "images"
    upload_batch_size: int = 50 # Adjusted default batch size
) -> str:
    """
    Loads results and mapping, transforms data, and uploads to HF for an iteration.

    Args:
        rapidata_results_path: Path to the saved Rapidata results JSON.
        image_mapping_path: Path to the image mapping JSON for the iteration.
        iteration_num: The current iteration number (e.g., 0).
        hf_repo_base_name: Base name for the Hugging Face repo.
        hf_token: Hugging Face API token.
        base_image_dir: Base directory where iteration image folders reside (e.g., "images").
        upload_batch_size: Batch size for processing images during upload.

    Returns:
        The Hugging Face repo ID for the uploaded dataset, or None if failed.
    """
    print(f"\n--- Transforming Data and Uploading: Iteration {iteration_num} ---")

    try:
        # Load Rapidata results (only the 'results' part is needed for transformation)
        rapidata_output = load_json(rapidata_results_path)
        results_list = rapidata_output.get("results")
        if results_list is None:
             print(f"❌ Key 'results' not found in Rapidata output file: {rapidata_results_path}")
             return None
        if not isinstance(results_list, list):
             print(f"❌ 'results' key in {rapidata_results_path} is not a list.")
             return None


        # Load image mapping
        image_mapping_content = load_json(image_mapping_path)

        # Create the map from relative image path -> prompt
        # Keys should be like 'iter_0/image.png'
        image_prompt_map = create_image_prompt_map(image_mapping_content)
        if not image_prompt_map:
            print("❌ Failed to create image-prompt map. Aborting.")
            return None

        # Transform the data
        final_df = transform_preference_data(
            rapidata_results=results_list,
            image_prompt_mapping=image_prompt_map,
            iteration_num=iteration_num,      # Pass iteration number
            base_image_dir=base_image_dir     # Pass base image dir ("images")
        )

        if final_df is None or final_df.empty: # Check for None as well
            print("❌ Transformation resulted in an empty or invalid DataFrame. Aborting upload.")
            return None

        print("\nFinal Transformed DataFrame sample:")
        print(final_df.head())

        # Define iteration-specific repo ID
        hf_repo_id = f"{hf_repo_base_name}-iter-{iteration_num}"

        # Upload the dataset
        upload_preference_dataset_to_hf(
            df=final_df,
            repo_id=hf_repo_id,
            token=hf_token,
            batch_size=upload_batch_size
        )
        # Assuming success if no exceptions were raised by upload function
        return hf_repo_id

    except FileNotFoundError:
        print("❌ File not found during transform/upload process. Please check paths.")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred during transform/upload for iteration {iteration_num}: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return None
