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
1. **Offline Training (Dataset generation and model training)**  
    ```
    cd offline
    bash train_offline.sh
    ```

2. **Online Training (Iterative dataset generation and model training)**  
    ```
    cd online
    bash train_online.sh
    ```


