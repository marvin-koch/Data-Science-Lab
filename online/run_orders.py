# --- START OF FILE run_orders.py ---

import os
import json
import csv
from datetime import datetime
# Keep diffusers/torch/PIL imports if needed elsewhere, but not directly used here
from rapidata import RapidataClient





def load_image_mapping_for_iteration(mapping_file_path: str):
    """Loads image mapping from a specific JSON file path."""
    if not os.path.exists(mapping_file_path):
        raise FileNotFoundError(f"❌ Mapping file not found: {mapping_file_path}")

    with open(mapping_file_path, "r") as f:
        image_mapping = json.load(f)

    print(f"🟢 Loaded image mapping from {mapping_file_path}")
    return image_mapping

def create_rapidata_order_for_iteration(
    rapi: RapidataClient,
    mapping_file_path: str,
    iteration_num: int,
    base_image_dir: str = "images",
    validation_set_id: str = None # id of validation set
):
    """Reads the JSON mapping for an iteration and submits a batch order."""
    print(f"\n--- Creating Rapidata Order: Iteration {iteration_num} ---")
    image_mapping = load_image_mapping_for_iteration(mapping_file_path)

    phrase = "Single centered logo on a plain background. No duplicates or mockups."

    prompts = [(entry["prompt"]).replace(phrase, "") for entry in image_mapping["data"]]
    
    # IMPORTANT: Ensure datapoints use paths relative to where Rapidata workers can access them
    # OR just filenames if transform_data.py resolves paths locally later.
    # Assuming transform_data.py resolves paths from base_image_dir + relative path in mapping:
    datapoints = [[os.path.join(base_image_dir, img_rel_path) for img_rel_path in entry["images"]]
                  for entry in image_mapping["data"]]

   
    """
    try:
        print(f"Attempting to find validation set with id: '{validation_set_id}'")
        validation_sets = rapi.validation.get_validation_set_by_id(validation_set_id)
        if validation_sets:
            print(f"🟢 Found validation set ID: {validation_set_id}")
        else:
            print(f"🟡 Validation set ID '{validation_set_id}' not found or empty. Proceeding without validation.")
    except Exception as e:
        print(f"⚠️ Error finding validation set: {e}. Proceeding without validation.")
    """


    order_name = f"SDXL Logos Feedback Iteration {iteration_num} - {datetime.now().strftime('%Y%m%d')}"
    print(f"Creating order: '{order_name}'")
    print(f"Contexts (Prompts): {len(prompts)}")
    print(f"Datapoints (Image Pairs): {len(datapoints)}")

    # Check if datapoints list is empty
    if not datapoints:
        raise ValueError("Cannot create order: No datapoints found in mapping file.")
        
    # Verify paths in the first datapoint exist (optional sanity check)
    if datapoints:
        first_dp = datapoints[0]
        if not all(os.path.exists(p) for p in first_dp):
             print(f"⚠️ Warning: Not all image paths in the first datapoint exist locally: {first_dp}")
             print(f"Make sure paths are correct for Rapidata workers if they access storage directly.")


    try:
        order = rapi.order.create_compare_order(
            name=order_name,
            instruction="Which logo do you prefer given the description?", # Customize instruction
            contexts=prompts,
            datapoints=datapoints,
            responses_per_datapoint=9, # Adjust as needed (e.g., 3-5 for cost/speed)
            validation_set_id=validation_set_id # Pass the ID if found
        ).run() # Use .run() to immediately start the order
        print(f"🟢 Order created and started successfully. Order ID: {order.order_id}")
    except Exception as e:
        print(f"❌ Failed to create Rapidata order: {e}")
        # Depending on severity, you might want to raise the exception
        # raise e
        return None # Return None if order creation failed

    return order # Return the order object

def save_order_results_for_iteration(
    order, # Pass the order object directly
    iteration_num: int,
    base_order_results_dir: str = "order_results"
):
    """Saves Rapidata order results in an iteration-specific folder."""
    print(f"\n--- Saving Order Results: Iteration {iteration_num} ---")

    if order is None:
        print("No order object provided, cannot save results.")
        return None

    # Create iteration-specific directory
    iter_results_dir = os.path.join(base_order_results_dir, f"iter_{iteration_num}")
    os.makedirs(iter_results_dir, exist_ok=True)

    try:
        print(f"Waiting for order {order.order_id} to complete...")
        order.display_progress_bar() # Blocks until completion
        print(f"Fetching results for order {order.order_id}...")
        # Check order status after progress bar, just in case

        results = order.get_results() # Fetches results
        print("Results fetched.")

        order_filename = f"order_results_{order.order_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_path = os.path.join(iter_results_dir, order_filename)

        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"🟢 Order results saved at {results_path}")
        return results_path

    except Exception as e:
        print(f"❌ Error getting or saving results for order {order.order_id}: {e}")
        return None # Return None if results couldn't be saved

