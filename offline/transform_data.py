
"""
# Define the path to the JSON file inside the 'order_results' folder
file_path = os.path.join('order_results', 'order_20250401_191556.json')

# Open the JSON file and load its contents
with open(file_path, 'r') as file:
    results = json.load(file)
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
from rapidata import RapidataClient
from huggingface_hub import HfApi
from datasets import Dataset, Features, Value, Image, concatenate_datasets
from tqdm import tqdm
import math
import os
# Define Hugging Face token (replace with your actual token)
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize Rapidata client
rapi = RapidataClient()

#orders = rapi.order.find_orders(name="750 Logos offline training", amount=1)
order = rapi.order.get_order_by_id("680ca4a0e63ceeba46b02aa4")

#order = orders[-1]

order_id = order.order_id

# Map metric types to IDs

# type2id = {
#     "Preference": "preference_order_id",
#     "Coherence": "coherence_order_id",
#     "Alignment": "alignment_order_id"
# }

type2id = {
    "Preference": order_id,
}
# Fetch results for each metric type
results = {
    res_type: rapi.order.get_order_by_id(id_).get_results()["results"] for res_type, id_ in type2id.items()
}

# Load image generation prompts
path_to_image_gen_prompts = "metadata/image_mapping.json"
image_dir = "images"
with open(path_to_image_gen_prompts, 'r') as file:
    image_gen_prompts = json.load(file)

# Convert list of prompts into dictionary
def convert_to_dict(data_list):
    return {item['id']: item['prompt'] for item in data_list}


def map_images_to_prompts(json_data):
    """
    Creates a mapping between images and prompts from the given JSON data.

    Args:
        json_data: A JSON string or a Python dictionary representing the data.

    Returns:
        A dictionary where keys are image filenames and values are the corresponding prompts.
    """

    try:
        if isinstance(json_data, str):
            data = json.loads(json_data)
        elif isinstance(json_data, dict):
            data = json_data
        else:
            raise ValueError("Input must be a JSON string or a Python dictionary.")

        mapping = {}
        for item in data["data"]:
            prompt = item["prompt"]
            for image in item["images"]:
                mapping[image] = prompt

        return mapping

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print(f"Error processing JSON: {e}")
        return {}  # Return an empty dictionary in case of errors

image_prompt_map = map_images_to_prompts(image_gen_prompts)

#id2prompt = convert_to_dict(image_gen_prompts)

# Transform data for a single metric into a DataFrame
def transform_single_metric_data(data_list: List[dict], prompt_mapping: Dict[str, str], metric_name: str):
    """
    Transform list of dictionaries into DataFrame with image bytes for a single metric.
    
    Args:
        data_list (list): List of dictionaries containing image comparison data
        prompt_mapping (dict): Dictionary mapping IDs to prompts
        metric_name (str): Name of the metric being processed (for column names)
        
    Returns:
        pd.DataFrame: Transformed data with specified columns including image bytes
    """
    transformed_data = []

    def get_image_path(file_name: str) -> bytes:
        """Read image file and return bytes."""
        try:
            folder_name = file_name.split('_')[0]  # Extract folder name
            image_path = Path(f"{image_dir}/{file_name}")
            return image_path
        except Exception as e:
            print(f"Error reading image {file_name}: {str(e)}")
            return b''  # Return empty bytes in case of error

    def get_prompt_id(image_name: str) -> int:
        """Extract prompt ID from image filename."""
        try:
            parts = image_name.split('_')
            if not parts or not parts[0]:
                print(f"Warning: Could not extract ID from image name: {image_name}")
                return -1

            id_str = parts[1].lstrip('0')
            if not id_str:  # If the string was all zeros
                return 0
            return int(id_str)
        except Exception as e:
            print(f"Error extracting prompt ID from {image_name}: {str(e)}")
            return -1

    for item in data_list:
        try:
            row = {}

            # Get image names from summedUserScoresRatios
            image_names = list(item['summedUserScores'].keys())
            if not image_names:
                print("Warning: No image names found in item")
                continue

            # # Get prompt from first image's ID
            # prompt_id = get_prompt_id(image_names[0])
            # row['prompt'] = prompt_mapping.get(prompt_id, f"Missing prompt for ID: {prompt_id}")

            row['prompt'] = prompt_mapping[image_names[0]]
            # Set image paths
            row['file_name1'] = image_names[0]
            row['file_name2'] = image_names[1]

            # Read image bytes
            row['image_path1'] = get_image_path(image_names[0])
            row['image_path2'] = get_image_path(image_names[1])

            # Set weighted results matching image paths
            scores = item['summedUserScoresRatios']
            row[f'weighted_results1_{metric_name}'] = scores[image_names[0]]
            row[f'weighted_results2_{metric_name}'] = scores[image_names[1]]

            # Add detailed results
            row[f'detailedResults_{metric_name}'] = item['detailedResults']

            transformed_data.append(row)

        except Exception as e:
            print(f"Error processing item: {str(e)}\nItem data: {item}")
            continue

    if not transformed_data:
        print("Warning: No data was successfully transformed")
        return pd.DataFrame()

    df = pd.DataFrame(transformed_data)

    # Define and order columns
    column_order = [
        'prompt',
        'file_name1',
        'file_name2',
        'image_path1',
        'image_path2',
        f'weighted_results1_{metric_name}',
        f'weighted_results2_{metric_name}',
        f'detailedResults_{metric_name}'
    ]

    return df[column_order]

# Example usage
preference_df = transform_single_metric_data(results["Preference"], image_prompt_map, "Preference")
# coherence_df = transform_single_metric_data(results["Coherence"], id2prompt, "Coherence")
# alignment_df = transform_single_metric_data(results["Alignment"], id2prompt, "Alignment")

# Merge the dataframes using the common columns as keys
common_columns = ['prompt', 'file_name1', 'file_name2', 'image_path1', 'image_path2']

# First merge preference and coherence

final_df = preference_df[common_columns + [
    'weighted_results1_Preference',
    'weighted_results2_Preference',
    'detailedResults_Preference'
]]

# merge_df = pd.merge(
#     preference_df[common_columns + [
#         'weighted_results1_Preference',
#         'weighted_results2_Preference',
#         'detailedResults_Preference'
#     ]],
#     coherence_df[common_columns + [
#         'weighted_results1_Coherence',
#         'weighted_results2_Coherence',
#         'detailedResults_Coherence'
#     ]],
#     on=common_columns,
#     how='outer'
# )

# Then merge with alignment
# final_df = pd.merge(
#     merged_df,
#     alignment_df[common_columns + [
#         'weighted_results1_Alignment',
#         'weighted_results2_Alignment',
#         'detailedResults_Alignment'
#     ]],
#     on=common_columns,
#     how='outer'
# )

# Display final merged DataFrame
print(final_df)

# Upload dataset to Hugging Face
def process_batch(batch_df):
    """Process a single batch of data, loading images from local paths"""
    rows = []
    for _, row in tqdm(batch_df.iterrows(), desc="Processing images in batch"):
        # Convert WindowsPath to regular Path if needed
        src_path1 = Path(row['image_path1'])
        src_path2 = Path(row['image_path2'])

        if not src_path1.exists() or not src_path2.exists():
            print(f"Warning: Missing images for paths: {src_path1} or {src_path2}")
            continue

        try:
            with open(src_path1, 'rb') as f1, open(src_path2, 'rb') as f2:
                image1_bytes = f1.read()
                image2_bytes = f2.read()

            dataset_row = {
                'prompt': row['prompt'],
                'image1': image1_bytes,
                'image2': image2_bytes,
                'model1': row['file_name1'].split('_')[0],
                'model2': row['file_name2'].split('_')[0],
                # Preference scores
                'weighted_results_image1_preference': row['weighted_results1_Preference'],
                'weighted_results_image2_preference': row['weighted_results2_Preference'],
                'detailed_results_preference': json.dumps(row['detailedResults_Preference']),
                # # Coherence scores
                # 'weighted_results_image1_coherence': row['weighted_results1_Coherence'],
                # 'weighted_results_image2_coherence': row['weighted_results2_Coherence'],
                # 'detailed_results_coherence': json.dumps(row['detailedResults_Coherence']),
                # # Alignment scores
                # 'weighted_results_image1_alignment': row['weighted_results1_Alignment'],
                # 'weighted_results_image2_alignment': row['weighted_results2_Alignment'],
                # 'detailed_results_alignment': json.dumps(row['detailedResults_Alignment'])
            }
            rows.append(dataset_row)
        except Exception as e:
            print(f"Error processing row: {e}")
            continue

    return rows

def upload_dataset(
    df,
    repo_id,
    token,
    batch_size=100,
    test_mode=False,
    n_test_samples=20
):
    """Upload dataset to HuggingFace Hub with batching for memory efficiency"""

    if test_mode:
        print(f"\nCreating test subset with {n_test_samples} samples...")
        df = df.sample(n=n_test_samples, random_state=42)

    # Define features
    features = Features({
        'prompt': Value('string'),
        'image1': Image(),
        'image2': Image(),
        'model1': Value('string'),
        'model2': Value('string'),
        'weighted_results_image1_preference': Value('float'),
        'weighted_results_image2_preference': Value('float'),
        'detailed_results_preference': Value('string'),
   
    })

    # Initialize HF API and create repository
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True
    )
    print(f"Repository {repo_id} is ready")

    # Process in batches
    num_batches = math.ceil(len(df) / batch_size)
    print(f"Processing {len(df)} rows in {num_batches} batches...")

    all_batches = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(df))

        print(f"\nProcessing batch {batch_idx + 1}/{num_batches}...")

        batch_df = df.iloc[start_idx:end_idx]
        batch_rows = process_batch(batch_df)

        if batch_rows:  # Only create dataset if we have valid rows
            batch_dataset = Dataset.from_list(batch_rows, features=features)
            all_batches.append(batch_dataset)
            print(f"Successfully processed batch {batch_idx + 1}")

        # Clear memory
        del batch_rows

    if not all_batches:
        print("No valid data to upload!")
        return

    # Combine all batches and upload
    print("\nCombining all batches...")
    final_dataset = concatenate_datasets(all_batches)

    print("\nUploading dataset to HuggingFace Hub...")
    final_dataset.push_to_hub(
        repo_id,
        token=token,
        private=True
    )

    print("\nUpload complete!")
    print(f"Dataset is available at: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    upload_dataset(
        df=final_df,
        repo_id="username/dataset",
        token=HF_TOKEN,
        batch_size=1,
        test_mode=False
    )

