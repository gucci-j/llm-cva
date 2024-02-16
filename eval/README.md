Evaluation
===

This is for LLM evaluation.

## Prerequisites
Please download the following two datasets from their corresponding websites. We cannot redistribute them due to the license agreements.
* KenSwQuAD - https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OTL0LM
* XCSQA - https://inklab.usc.edu/XCSR/xcsr_datasets

## Usage
```
$ python src/main.py -h
usage: main.py [-h] --model_name_or_path MODEL_NAME_OR_PATH
               [--tokenizer_name_or_path TOKENIZER_NAME_OR_PATH]
               [--model_cache_dir MODEL_CACHE_DIR]
               [--data_cache_dir DATA_CACHE_DIR] [--dataset_path DATASET_PATH]
               --task_name {xlsum,xnli,xquad,xcsqa} --target_lang
               {japanese,english,german,swahili,arabic}
               [--results_dir RESULTS_DIR] [--seed SEED]
               [--num_max_samples NUM_MAX_SAMPLES] [--num_shots NUM_SHOTS]
               [--is_peft] [--lora_only]
               [--adapter_name_or_path ADAPTER_NAME_OR_PATH]
               [--plot_model_name PLOT_MODEL_NAME]
               [--plot_category PLOT_CATEGORY]
               [--max_context_len MAX_CONTEXT_LEN]
               [--prompting_in_target_language] [--use_arabert]
               [--temperature TEMPERATURE]
               [--repetition_penalty REPETITION_PENALTY] [--top_k TOP_K]
               [--top_p TOP_P] [--num_beams NUM_BEAMS]
               [--num_return_sequences NUM_RETURN_SEQUENCES] [--do_sample]
               [--early_stopping]

options:
  -h, --help            show this help message and exit
  --model_name_or_path MODEL_NAME_OR_PATH
                        The model name or path
  --tokenizer_name_or_path TOKENIZER_NAME_OR_PATH
                        The tokenizer name or path. If None, the
                        model_name_or_path is used as tokenizer_name_or_path
  --model_cache_dir MODEL_CACHE_DIR
                        The directory where the model is cached
  --data_cache_dir DATA_CACHE_DIR
                        The directory where the dataset is cached
  --dataset_path DATASET_PATH
                        [xcsqa, kenswquad] The path to the dataset
  --task_name {xlsum,xnli,xquad,xcsqa}
                        The task to evaluate
  --target_lang {japanese,english,german,swahili,arabic}
                        The target language to evaluate
  --results_dir RESULTS_DIR
                        The directory where the results are saved
  --seed SEED           Random seed for evaluation
  --num_max_samples NUM_MAX_SAMPLES
                        The maximum number of samples to evaluate
  --num_shots NUM_SHOTS
                        The number of shots to evaluate
  --is_peft             Whether to use PEFT model or not
  --lora_only           Whether to use LoRA only model or not
  --adapter_name_or_path ADAPTER_NAME_OR_PATH
                        The adapter name or path
  --plot_model_name PLOT_MODEL_NAME
                        The model name for plotting
  --plot_category PLOT_CATEGORY
                        The category for plotting
  --max_context_len MAX_CONTEXT_LEN
                        The maximum length of the context
  --prompting_in_target_language
                        Whether to use prompting in target language or not
  --use_arabert         Whether to use AraBERT or not
  --temperature TEMPERATURE
                        The value used to module the next token probabilities
  --repetition_penalty REPETITION_PENALTY
                        The parameter for repetition penalty
  --top_k TOP_K         The number of highest probability vocabulary tokens to
                        keep for top-k-filtering
  --top_p TOP_P         The cumulative probability of parameter highest
                        probability vocabulary tokens to keep for nucleus
                        sampling
  --num_beams NUM_BEAMS
                        The number of beams for beam search
  --num_return_sequences NUM_RETURN_SEQUENCES
                        The number of independently computed returned
                        sequences for each element in the batch
  --do_sample           Whether to use sampling or not
  --early_stopping      Whether to stop the beam search when at least
                        num_beams sentences are finished per batch or not
```

## Reproduction
### 1. Source
The following shell script is an example of evaluating BLOOM-1B (Source) over the four tasks introduced in the paper. Please modify the paths accordingly.

* [source.sh](./scripts/source.sh)

### 2. LAPT

The following shell script is an example of evaluating the Swahili LAPT baseline model using BLOOM-1B as source over the four tasks introduced in the paper. Please modify the paths accordingly.

* [lapt.sh](./scripts/lapt.sh)

### 3. Cross-lingual Vocabulary Adaptation

The following shell script is an example of evaluating **all** the Swahili adapted model using BLOOM-1B as source over the four tasks introduced in the paper. Please modify the paths accordingly.

* [cva.sh](./scripts/cva.sh)
