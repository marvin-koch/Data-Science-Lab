# Text-to-Image model fine-tuning using real human preference data

ETH Data Science Lab Spring 2025, in collaboration with Rapidata.ai. 

## Setup

1. **Install dependencies**  
    ``` 
   pip install -r requirements.txt
    ```

2. **Define your Huggingface token**
    ```
   export HF_TOKEN="some value"
    ```



## Run
**Offline Training (Dataset generation and model training)**  
    ```
   bash offline/train_offline.sh
    ```

**Online Training (Iterative dataset generation and model training)**  
    ```
   bash online/train_offline.sh
    ```


