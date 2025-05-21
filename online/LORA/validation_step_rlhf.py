import os
import json
import csv
from datetime import datetime
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
from rapidata import RapidataClient
import random
from typing import List, Dict, Any
import math # For handling potential floating point inaccuracies if needed
from rapidata import RapidataClient
import time

def get_validation_feedback(dataset_path_1: str, dataset_path_2: str, validation_prompts: List[str]) -> List[Dict[str, Any]]:
    """
    Reads filenames and paths from two directories, pairs them based on sorted
    filename order, associates them with corresponding prompts, and creates a
    randomized list of datapoints for comparison.

    Assumes that files in dataset_path_1 and dataset_path_2 correspond to each
    other based on their alphabetical sorting order, and that this order also
    corresponds to the order of prompts in validation_prompts.

    Args:
        dataset_path_1 (str): Path to the first directory containing files
                              (e.g., generated images/videos/audio).
        dataset_path_2 (str): Path to the second directory containing the
                              corresponding files for comparison (e.g., ground
                              truth or another model's output).
        validation_prompts (list[str]): A list of prompts, where the i-th prompt
                                         corresponds to the i-th file pair when
                                         files in both directories are sorted
                                         alphabetically.

    Returns:
        list[dict]: A list of datapoint dictionaries ready for comparison.
                    Each dictionary has the structure:
                    {
                        'file1': str,  # Path to the file from dataset_path_1
                        'file2': str,  # Path to the file from dataset_path_2
                        'context': str # The associated prompt
                    }
                    The order of datapoints in the list is randomized.
                    Returns an empty list if no valid pairs can be formed.

    Raises:
        FileNotFoundError: If either dataset_path_1 or dataset_path_2 does not
                           exist or is not a directory.
        ValueError: If the number of files found in dataset_path_1, the number
                    of files found in dataset_path_2, and the number of prompts
                    do not all match.
    """
    datapoints = []

    # --- 1. Validate Input Paths ---
    if not os.path.isdir(dataset_path_1):
        raise FileNotFoundError(f"Dataset path 1 does not exist or is not a directory: {dataset_path_1}")
    if not os.path.isdir(dataset_path_2):
        raise FileNotFoundError(f"Dataset path 2 does not exist or is not a directory: {dataset_path_2}")

    # --- 2. List and Sort Files ---
    # List only files (not directories) within the dataset paths and sort them
    try:
        files1 = sorted([f for f in os.listdir(dataset_path_1) if os.path.isfile(os.path.join(dataset_path_1, f))])
        files2 = sorted([f for f in os.listdir(dataset_path_2) if os.path.isfile(os.path.join(dataset_path_2, f))])
    except OSError as e:
        raise OSError(f"Could not read files from input directories.") from e

    # --- 3. Check for Count Mismatches ---
    num_prompts = len(validation_prompts)
    num_files1 = len(files1)
    num_files2 = len(files2)

    if not (num_files1 == num_files2 == num_prompts):
        raise ValueError(
            f"Mismatch in counts required for pairing: "
            f"Number of prompts = {num_prompts}, "
            f"Number of files in '{os.path.basename(dataset_path_1)}' = {num_files1}, "
            f"Number of files in '{os.path.basename(dataset_path_2)}' = {num_files2}. "
            f"All three counts must be identical."
        )

    if num_prompts == 0:
        print("No prompts provided (or no files found matching prompt count), returning empty datapoints list.")
        return [] # Return empty list if there's nothing to process

    # --- 4. Create Datapoint Dictionaries ---
    for i in range(num_prompts):
        filename1 = files1[i]
        filename2 = files2[i]
        prompt = validation_prompts[i]

        # Construct the full paths (these will be local file paths)
        # If URLs are needed, further transformation would be required here
        # based on how the files are served (e.g., adding a base URL).
        path1 = os.path.join(dataset_path_1, filename1)
        path2 = os.path.join(dataset_path_2, filename2)

        # Create the datapoint structure as required
        datapoint = [path1, path2]
        datapoints.append(datapoint)


    print(f"Successfully created {len(datapoints)} datapoints.")

    return datapoints





# def extract_comparison_metrics(order_results):
#     """
#     Extracts detailed comparison metrics between two models (A/LoRA vs B/Base)
#     from API results.

#     Args:
#         order_results (dict): The dictionary containing the API call results.

#     Returns:
#         dict: A dictionary containing the 8 extracted metrics:
#               - raw_wins_lora
#               - raw_wins_base
#               - user_score_wins_lora
#               - user_score_wins_base
#               - sum_votes_lora
#               - sum_votes_base
#               - sum_user_score_lora
#               - sum_user_score_base
#               Returns None if the input structure is invalid.
#     """
#     if not isinstance(order_results, dict) or 'results' not in order_results:
#         print("Error: Invalid input format. 'order_results' must be a dict with a 'results' key.")
#         return None

#     # Initialize counters and sums
#     raw_wins_lora = 0
#     raw_wins_base = 0
#     user_score_wins_lora = 0  # Sum of summedUserScores for A across all contexts
#     user_score_wins_base = 0  # Sum of summedUserScores for B across all contexts
#     sum_votes_lora = 0
#     sum_votes_base = 0
#     sum_user_score_lora = 0.0   # Sum of individual userScores for A votes
#     sum_user_score_base = 0.0   # Sum of individual userScores for B votes

#     results_list = order_results.get('results', [])
#     if not isinstance(results_list, list):
#         print("Error: 'results' key does not contain a list.")
#         return None

#     # --- 1 & 2: Raw Wins ---
#     # Calculated based on aggregatedResults (raw vote counts) per context
#     raw_wins_lora = aggregated_results.get("A_wins_total")
#     raw_wins_base = aggregated_results.get("B_wins_total")
#     for result in results_list:
#         if not isinstance(result, dict):
#             print(f"Warning: Skipping invalid item in 'results' list: {result}")
#             continue

#         aggregated_results = result.get('aggregatedResults', {})
#         summed_user_scores = result.get('summedUserScores', {})
#         detailed_results = result.get('detailedResults', [])

#         # Ensure we have exactly two models to compare in this context
#         if len(aggregated_results) != 2 or len(summed_user_scores) != 2:
#             print(f"Warning: Skipping result context due to unexpected number of models in aggregatedResults or summedUserScores: {result.get('context')}")
#             continue

#         # Determine Model A (LoRA) and Model B (Base) identifiers for this context
#         # We rely on the documented assumption that the first key is Model A, second is Model B
#         model_ids = list(aggregated_results.keys())
#         model_a_id = model_ids[0]
#         model_b_id = model_ids[1]
        
#         # --- 3 & 4: User Score Wins (who won based on summed_user_score) ---
#         if summed_user_score[model_a_id] > summed_user_score[model_b_id]:
#             user_score_wins_lora += 1
#         elif summed_user_score[model_a_id] < summed_user_score[model_b_id]:
#             user_score_wins_base += 1

#         # --- 5 & 6: Sum of Raw Votes ---
#         # Accumulated from aggregatedResults across all contexts
#         sum_votes_lora += aggregated_results[model_a_id]
#         sum_votes_base += aggregated_results[model_b_id]

#         # --- 7 & 8: Sum of Individual User Scores per Vote ---
#         # Requires iterating through detailedResults
#         sum_user_score_lora = summed_user_score[model_a_id]
#         sum_user_score_base = summed_user_score[model_b_id]

        

#     # Prepare the final dictionary
#     metrics = {
#         "raw_wins_lora": raw_wins_lora,
#         "raw_wins_base": raw_wins_base,
#         "user_score_wins_lora": user_score_wins_lora,
#         "user_score_wins_base": user_score_wins_base,
#         "sum_votes_lora": sum_votes_lora,
#         "sum_votes_base": sum_votes_base,
#         "sum_user_score_lora": sum_user_score_lora,
#         "sum_user_score_base": sum_user_score_base,
#     }

#     return metrics


def extract_comparison_metrics(order_results):
    """
    Extracts detailed comparison metrics between two models (A/LoRA vs B/Base)
    from API results.

    Args:
        order_results (dict): The dictionary containing the API call results.

    Returns:
        dict: A dictionary containing the 8 extracted metrics.
              Returns None if the input structure is invalid.
    """
    if not isinstance(order_results, dict) or 'results' not in order_results:
        print("Error: Invalid input format. 'order_results' must be a dict with a 'results' key.")
        return None

    # Initialize counters and sums
    raw_wins_lora = order_results.get("summary", {}).get("A_wins_total", 0)
    raw_wins_base = order_results.get("summary", {}).get("B_wins_total", 0)
    user_score_wins_lora = 0
    user_score_wins_base = 0
    sum_votes_lora = 0
    sum_votes_base = 0
    sum_user_score_lora = 0.0
    sum_user_score_base = 0.0

    results_list = order_results.get('results', [])
    if not isinstance(results_list, list):
        print("Error: 'results' key does not contain a list.")
        return None

    for result in results_list:
        if not isinstance(result, dict):
            print(f"Warning: Skipping invalid item in 'results' list: {result}")
            continue

        aggregated_results = result.get('aggregatedResults', {})
        summed_user_scores = result.get('summedUserScores', {})

        # Ensure we have exactly two models to compare in this context
        if len(aggregated_results) != 2 or len(summed_user_scores) != 2:
            print(f"Warning: Skipping result context due to unexpected number of models in aggregatedResults or summedUserScores: {result.get('context')}")
            continue

        model_ids = list(aggregated_results.keys())
        model_a_id = model_ids[0]
        model_b_id = model_ids[1]

        # 3 & 4: User Score Wins
        score_a = summed_user_scores.get(model_a_id, 0.0)
        score_b = summed_user_scores.get(model_b_id, 0.0)
        if score_a > score_b:
            user_score_wins_lora += 1
        elif score_b > score_a:
            user_score_wins_base += 1

        # 5 & 6: Sum of Raw Votes
        sum_votes_lora += aggregated_results.get(model_a_id, 0)
        sum_votes_base += aggregated_results.get(model_b_id, 0)

        # 7 & 8: Sum of User Scores
        sum_user_score_lora += score_a
        sum_user_score_base += score_b

    # Final metrics dictionary
    return {
        "raw_wins_lora": raw_wins_lora,
        "raw_wins_base": raw_wins_base,
        "user_score_wins_lora": user_score_wins_lora,
        "user_score_wins_base": user_score_wins_base,
        "sum_votes_lora": sum_votes_lora,
        "sum_votes_base": sum_votes_base,
        "sum_user_score_lora": sum_user_score_lora,
        "sum_user_score_base": sum_user_score_base,
    }


def run_order(dataset_path_1: str, dataset_path_2: str, validation_prompts: List[str],valid_step: int) -> Dict[str, float]:
    datapoints = get_validation_feedback(dataset_path_1 = dataset_path_1,dataset_path_2 = dataset_path_2,validation_prompts = validation_prompts)
    rapi = RapidataClient()
    order = rapi.order.create_compare_order(
        name="Validation Test at Step {}".format(valid_step),
        instruction="Which logo do you prefer given the description?",
        contexts=validation_prompts,  # Pass prompts as context
        datapoints=datapoints,
        responses_per_datapoint=15,
        validation_set_id="67f002c7075c672e0bfe36e6"
        ).run()
    
    time.sleep(60)

    order_results = order.get_results()
    return extract_comparison_metrics(order_results=order_results)
    
    
    


