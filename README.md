
## Getting Started
### Prerequisites
- Python 3.12
- Minimum of 20GB VRAM for running evaluations
- Minimum of 40GB VRAM for training (default setting)

### Setup
1. **Set up a virtual environment (tested with Python 3.8-3.12)**  

2. **Install required dependencies**  

    ```sh
    pip install -r requirements.txt
    ```
   
3. **Set up WandB API key**  

    Set up your [WandB](https://wandb.ai/) API key for training and evaluation logging.

    ```sh
    export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ```
    Input the WANDB API key to login when prompted as you run
    ```
    huggingface-cli login
    ```
    
### Dataset
- **Training/testing data**:
The datasets have already been checked into the codebase. To unarchive them, use the following commands:
    ```
    tar -xzvf data/vqa_train_10k.tar.gz -C data/
    tar -xzvf data/vqa_test_1k.tar.gz -C data/
    ```

<!-- - **Re-collect DrivingQA data**:
While the training and evaluation datasets already include pre-collected DrivingQA data, we also offer a script that illustrates how to collect DrivingQA data using the OpenAI ChatGPT API. If you wish to re-collect the DrivingQA data, simplely run the following command with your OpenAI API key:
    ```sh
    python scripts/collect_vqa.py -i data/vqa_test_1k.pkl -o output_folder/ --openai_api xxxxxxxx
    ``` -->
### Finetuning with legible objective

1. **Finetuning on the pretrained model**

    Run the following command:

    ```sh
    python train_legible.py
    ```

2. **Evaluate for DrivingQA**

    Run the following command:

    ```sh
    python eval_legible.py \
        --model_dir ./legible \
        --data_path data/vqa_test_1k.pkl \
        --n_samples 5 \
        --seed 2026
    ```

3. **View Results**  

    The results can be viewed on the WandB project "llm-driver".


### Acknowledgements

This project has drawn inspiration from the [Driving-with-LLMs](https://github.com/wayveai/Driving-with-LLMs) repository. 
