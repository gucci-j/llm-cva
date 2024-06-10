Tuning (LAPT)
===

## Usage
```bash
$ python src/main.py -h
usage: main.py [-h] --dataset_path DATASET_PATH --tokenizer_name_or_path
               TOKENIZER_NAME_OR_PATH --model_name_or_path MODEL_NAME_OR_PATH
               [--cache_dir CACHE_DIR] --model_type {gpt2,bloom,llama2}
               [--tune_embeddings] [--r R] [--lora_alpha LORA_ALPHA]
               [--lora_dropout LORA_DROPOUT] [--generate_dev_set]

Tune a language model.

options:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        Path to the tokenized dataset.
  --tokenizer_name_or_path TOKENIZER_NAME_OR_PATH
                        Path to the tokenizer.
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to the model.
  --cache_dir CACHE_DIR
                        Path to the cache directory.
  --model_type {gpt2,bloom,llama2}
                        Model type.
  --tune_embeddings     Whether to tune the embeddings.
  --r R                 The r parameter for LoRA.
  --lora_alpha LORA_ALPHA
                        The alpha parameter for LoRA.
  --lora_dropout LORA_DROPOUT
                        The dropout parameter for LoLA.
  --generate_dev_set    Whether to generate a development set.
```

## Reproduction
**Example 1**: The following is an example of training the Swahili LAPT baseline model using BLOOM-1B as source.
```bash
cd /path/to/llm-vocab-adaptation/tuning/src

python main.py \
    --dataset_path /path/to/trimmed/preprocessed/swahili/data/dir/ \
    --output_dir /path/to/output/dir/bloom-1b1-sw-pruned-tuned \
    --logging_dir /path/to/llm-vocab-adaptation/tuning/logs/bloom-1b1-sw-pruned \
    --model_name_or_path /path/to/trimmed/model/dir/bloom-1b1-sw-pruned \
    --tokenizer_name_or_path /path/to/trimmed/model/dir/bloom-1b1-sw-pruned \
    --model_type bloom \
    --seed 42 \
    --evaluation_strategy no \
    --logging_steps 5 \
    --save_steps 1000 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --prediction_loss_only \
    --overwrite_output_dir \
    --optim adamw_bnb_8bit \
    --report_to tensorboard \
    --do_train \
    --tf32 True \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --disable_tqdm True \
    --label_names labels \
    --remove_unused_columns False \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --generate_dev_set
```

**Example 2**: The following is an example of training the Swahili target model adapted with CLP+ using BLOOM-1B as source.
```bash
cd /path/to/llm-vocab-adaptation/tuning/src

python main.py \
    --dataset_path /path/to/preprocessed/swahili/data \
    --output_dir /path/to/output/dir/bloom-1b1-sw-clp-plus-tuned \
    --logging_dir /path/to/llm-vocab-adaptation/tuning/logs/bloom-1b1-sw-clp-plus \
    --model_name_or_path /path/to/adapted/model/dir/bloom-1b1-sw-clp-plus \
    --tokenizer_name_or_path /path/to/adapted/model/dir/bloom-1b1-sw-clp-plus \
    --model_type bloom \
    --tune_embeddings \
    --evaluation_strategy no \
    --logging_steps 5 \
    --save_steps 1000 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --prediction_loss_only \
    --overwrite_output_dir \
    --optim adamw_bnb_8bit \
    --report_to tensorboard \
    --do_train \
    --tf32 True \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --disable_tqdm True \
    --label_names labels \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --generate_dev_set
```
