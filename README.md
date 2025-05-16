# Goal Inference Benchmark for Assistive Wearable Agents

This repository is the official implementation of the NeurIPS 2025 submission (#2012) [Benchmarking Egocentric Multimodal Goal Inference for Assistive Wearable Agents]().

This repository contains the code to evaluate the Qwen2.5-VL-72B model on the benchmark, for both the Multi-Choice-Question (MCQ) and Generative-LLM-Judge settings. Also, we provide the raw predictions for all models used in the paper, and the analysis script to help you reproduce the all the results.

## Requirements
This repository requires python3.9.

To install requirements:

```
pip install -r requirements.txt
```

## Prepare Data and Models
First, you need to set up the keys to access the S3 bucket. You can find the keys (`s3_access_keys.sh`) in the zip file submitted for Supplemental Material.
Then, you can either do `sh s3_access_keys.sh`, or set up the key directly by running the following command with the correct values swapped in:
```
export AWS_ACCESS_KEY_ID=FIND_THE_AWS_ACCESS_KEY_ID_IN_SUPPLEMENTAL_MATERIAL
export AWS_SECRET_ACCESS_KEY=FIND_THE_AWS_SECRET_ACCESS_KEY_IN_SUPPLEMENTAL_MATERIAL
```

With the access keys configured, run this command to download the data for the benchmark:
```
aws s3 cp s3://submission-2012/data/ ob2/ --recursive
```
If you don't have aws cli installed, you can install it following the instructions here: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

You can download the Qwen2.5-VL-72B-Instruct model from HuggingFace: https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct

And the LLM Judge model from HuggingFace: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B

With both dataset and model downloaded, you should have a directory structure like this:
```
ROOT_DIR
├── ob2
│   ├── digital_state_v2.4.4.json
│   ├── ob2_v2.4.4_1_mcq.json
│   ├── structured_goals.json
│   ├── ob2_transcriptions_250501.json
│   ├── structured_goals_longitudinal_history_v244.json
│   ├── clipped_videos
│   │   ├── 001a1239-f4f6-4a93-8e1b-533f0901b7f3.mp4
│   │   ├── ...
├── models
│   ├── Qwen2.5-VL-72B-Instruct
│   │── DeepSeek-R1-Distill-Llama-70B
├── assets
│   ├── raw_predictions
│   │   ├── generative_internvl_2b_V.json
│   │   ├── ...
```

## Evaluation

First, edit the `ROOT_DIR` variable in evaluation scripts `eval_qwen_mcq.sh` and `eval_qwen_generative.sh` to point to the correct path.

You need at least 4 GPUs with 80GB memory (e.g. A100-80G, H100) to run the evaluation.

To evaluate Qwen2.5-VL-72B-Instruct model on the Multi-Choice-Question task of the benchmark, run:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash eval_qwen_mcq.sh
```

To evaluate Qwen2.5-VL-72B-Instruct model on the generative task of the benchmark, run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash eval_qwen_generative.sh
```
This will evaluate Qwen2.5-VL-72B-Instruct model using DeepSeek-R1-Distill-Llama-70B as the LLM Judge.

If the benchmark is properly setup, you should see similar results for both runs:
```
MCQ: 0.871299
Generative: 0.507109
```
## Analysis

The analysis script consolidates results from different models, creates multiple data subsets, and evaluates results on them using varying combinations of input modalities.

You need two specific OB2 input files under `ob2` and all the raw prediction files to be present under `assets/raw_predictions` to run subset analysis:
```
ROOT_DIR
├── ob2
│   ├── digital_state_v2.4.4.json
│   ├── structured_goals.json
├── assets
│   ├── raw_predictions
│   │   ├── generative_internvl_2b_V.json
│   │   ├── generative_internvl_2b_VA.json
│   │   ├── ...
```

Run the command:
```
python subset_analysis.py -o ~/temp/
```

This will generate a `temp` directory in your home folder and save all the results and plots in it. If run successfully, you should see results similar to the ones below around script's termination output on the terminal:
```
#### MODEL RESULTS on generative task ####
internvl_2b: 0.2187 (0.2078, 0.2297)
qwen_3b: 0.2534 (0.2418, 0.2649)
llama3v: 0.3235 (0.3119, 0.3351)
internvl_8b: 0.3571 (0.3442, 0.3691)
qwen_7b: 0.4076 (0.3952, 0.42)
llama4v: 0.4484 (0.4346, 0.46)
internvl_78b: 0.4956 (0.4821, 0.5077)
qwen_72b: 0.4988 (0.4851, 0.5115)
```
