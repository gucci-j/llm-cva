Adaptation
===

This is to initialize embeddings using various cross-lingual vocabulary adaptation techniques.

## Implemented Approaches
* LAPT (Trim unused vocabulary)
* Random
* [CLP](https://arxiv.org/abs/2301.09626)
* CLP+
* [Heuristics](https://arxiv.org/abs/2309.04679)
* [FOCUS](https://arxiv.org/abs/2305.14481)

## Usage
```bash
$ python src/main.py -h
usage: main.py [-h] --initialization_method
               {random,clp,clp_plus,heuristics,lapt,focus}
               --source_model_name_or_path SOURCE_MODEL_NAME_OR_PATH
               --source_tokenizer_name_or_path SOURCE_TOKENIZER_NAME_OR_PATH
               [--helper_tokenizer_name_or_path HELPER_TOKENIZER_NAME_OR_PATH]
               [--helper_model_name_or_path HELPER_MODEL_NAME_OR_PATH]
               [--target_tokenizer_name_or_path TARGET_TOKENIZER_NAME_OR_PATH]
               [--cache_dir CACHE_DIR] [--seed SEED] [--copy_special_tokens]
               [--unicode_script_file_path UNICODE_SCRIPT_FILE_PATH]
               --output_dir OUTPUT_DIR [--dataset_path DATASET_PATH]
               [--output_data_dir OUTPUT_DATA_DIR]
               [--fasttext_model_path FASTTEXT_MODEL_PATH]

options:
  -h, --help            show this help message and exit
  --initialization_method {random,clp,clp_plus,heuristics,lapt,focus}
                        The embedding initialization method to use.
  --source_model_name_or_path SOURCE_MODEL_NAME_OR_PATH
                        The source model to initialize the target model with.
  --source_tokenizer_name_or_path SOURCE_TOKENIZER_NAME_OR_PATH
                        The source tokenizer to initialize the target
                        tokenizer with.
  --helper_tokenizer_name_or_path HELPER_TOKENIZER_NAME_OR_PATH
                        [expand_after] The helper tokenizer to help initialize
                        a terget tokenizer.
  --helper_model_name_or_path HELPER_MODEL_NAME_OR_PATH
                        [clp, clp_plus] The helper model to help initialize a
                        terget model.
  --target_tokenizer_name_or_path TARGET_TOKENIZER_NAME_OR_PATH
                        [random, clp, clp_plus, heuristics, focus] The target
                        tokenizer name or path.
  --cache_dir CACHE_DIR
                        The cache directory to save the pretrained models.
  --seed SEED           The random seed.
  --copy_special_tokens
                        [clp, clp_plus] Whether to copy the special tokens'
                        embeddings from the source model to the target model.
  --unicode_script_file_path UNICODE_SCRIPT_FILE_PATH
                        [heuristics] The path to the unicode script file.
  --output_dir OUTPUT_DIR
                        The output directory to save the target model and
                        tokenizer.
  --dataset_path DATASET_PATH
                        [lapt] The path to the dataset.
  --output_data_dir OUTPUT_DATA_DIR
                        [lapt] The output directory to save the pruned
                        dataset.
  --fasttext_model_path FASTTEXT_MODEL_PATH
                        [focus] The path to the FastText model.
```

## Reproduction
**Example 1**: The following is to initialize a BLOOM-1B model for Arabic using CLP.
```bash
#!/bin/bash

cd /path/to/llm-vocab-adaptation/adaptation/src

python main.py \
    --source_model_name_or_path bigscience/bloom-1b1 \
    --source_tokenizer_name_or_path bigscience/bloom-1b1 \
    --helper_model_name_or_path aubmindlab/aragpt2-base \
    --target_tokenizer_name_or_path aubmindlab/aragpt2-base \
    --cache_dir /path/to/cache/dir \
    --seed 42 \
    --initialization_method clp \
    --output_dir /path/to/output/model/dir/bloom-1b1-ar-clp 
```

**Example 2**: The following is to trim the vocabulary of a BLOOM-1B model for Arabic. This will be used as LAPT.
```bash
#!/bin/bash

cd /path/to/llm-vocab-adaptation/adaptation/src

python main.py \
    --source_model_name_or_path bigscience/bloom-1b1 \
    --source_tokenizer_name_or_path bigscience/bloom-1b1 \
    --cache_dir /path/to/cache/dir \
    --initialization_method lapt \
    --output_dir /path/to/output/dir/bloom-1b1-ar-pruned \
    --dataset_path /path/to/preprocessed/arabic/data/dir/shard_0 \
    --output_data_dir /path/to/trimmed/preprocessed/arabic/data/dir/shard_0
```

**Example 3**: The following is to initialize a BLOOM-1B model for Swahili using Heuristics.
```bash
#!/bin/bash

cd /path/to/llm-vocab-adaptation/adaptation/src

python main.py \
    --source_model_name_or_path bigscience/bloom-1b1 \
    --source_tokenizer_name_or_path bigscience/bloom-1b1 \
    --helper_model_name_or_path benjamin/gpt2-wechsel-swahili \
    --target_tokenizer_name_or_path benjamin/gpt2-wechsel-swahili \
    --cache_dir /path/to/cache/dir \
    --seed 42 \
    --initialization_method heuristics \
    --unicode_script_file_path /path/to/llm-vocab-adaptation/adaptation/data/unicode_scripts_for_embeddings_exploration.txt \
    --output_dir /path/to/output/dir/bloom-1b1-sw-heuristics
```

**Example 4**: The following is to initialize a BLOOM-1B model for Swahili using FOCUS.
```bash
#!/bin/bash

cd /path/to/llm-vocab-adaptation/adaptation/src

python main.py \
    --source_model_name_or_path bigscience/bloom-1b1 \
    --source_tokenizer_name_or_path bigscience/bloom-1b1 \
    --target_tokenizer_name_or_path benjamin/gpt2-wechsel-swahili \
    --fasttext_model_path /path/to/fasttext_model_sw_bpe.bin \
    --cache_dir /path/to/cache/dir \
    --seed 42 \
    --initialization_method focus \
    --output_dir /path/to/output/dir/bloom-1b1-sw-focus
```
