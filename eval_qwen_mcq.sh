#!/bin/bash
export PYTHONPATH="$(pwd):$PYTHONPATH";

ROOT_DIR=/PATH/TO/BENCHMARK_ROOT

torchrun --nnodes=1 --nproc-per-node=1 --max-restarts=1 \
  evals/cli/multimodal_eval.py \
  --tasks maestro_ob2 \
  --dataset_dir $ROOT_DIR/ob2/clipped_videos \
  --model_path $ROOT_DIR/models/Qwen2.5-VL-72B-Instruct/ \
  --predictor_name maestro_ob2_qwen \
  --task_args '{
    "maestro_ob2-maestro_ob2_qwen": {
      "jsonl_dataset_path": "'$ROOT_DIR'/ob2/ob2_v2.4.4_1_mcq.json",
      "max_gen_len": 1024,
      "digital_key_only": "0",
      "longitudinal_positive_only": "0",
      "transcription_dataset_path": "'$ROOT_DIR'/ob2/ob2_transcriptions_250501.json",
      "digital_state_path": "'$ROOT_DIR'/ob2/digital_state_v2.4.4.json",
      "longitudinal_history_path": "'$ROOT_DIR'/ob2/structured_goals_longitudinal_history_v244.json"
    }
  }' \
  --dump_dir $ROOT_DIR/evals_runs \
  --number_of_samples 32 \
  --device_map auto
