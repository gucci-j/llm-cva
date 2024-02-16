Preprocessing
===

## Reproduction
### 1. Download data
#### OSCAR
Download the OSCAR data for German, Japanese, and Arabic. Although we provide the following script to download the data, you need to login to HuggingFace CLI before running to it.
```bash
$ python src/download_oscar.py -h
usage: download_oscar.py [-h] --cache_dir CACHE_DIR --target_lang
                         {ja,de,sw,ar}

options:
  -h, --help            show this help message and exit
  --cache_dir CACHE_DIR
  --target_lang {ja,de,sw,ar}
```

**Example**: The following will download the Arabic OSCAR data under cache_dir.  
```bash
#!/bin/bash

cd /path/to/llm-vocab-adaptation/preprocessing/src
python download_oscar.py \
    --cache_dir /path/to/data/cache/dir \
    --target_lang ar
```

#### CC-100
Download the CC-100 data for German, Japanese, Arabic, and Swahili. It is available at https://data.statmt.org/cc-100/


### 2. Preprocessing
Preprocess each downloaded datasets using the following script.
```bash
$ export HF_DATASETS_CACHE="/path/to/another/directory"
$ python src/preprocess.py -h
usage: preprocess.py [-h] --target_lang {ja,de,sw,ar} --cache_dir CACHE_DIR
                     --output_dir OUTPUT_DIR --tokenizer_name_or_path
                     TOKENIZER_NAME_OR_PATH
                     [--tokenizer_cache_dir TOKENIZER_CACHE_DIR]
                     [--block_size BLOCK_SIZE] [--min_length MIN_LENGTH]
                     [--max_length MAX_LENGTH] [--num_shards NUM_SHARDS]
                     [--shard_index SHARD_INDEX] [--seed SEED]
                     [--cc100_data_path CC100_DATA_PATH] [--use_filter]

Preprocess the OSCAR or CC-100 corpus.

options:
  -h, --help            show this help message and exit
  --target_lang {ja,de,sw,ar}
                        The target language.
  --cache_dir CACHE_DIR
                        The cache directory.
  --output_dir OUTPUT_DIR
                        The output directory.
  --tokenizer_name_or_path TOKENIZER_NAME_OR_PATH
                        The tokenizer name or path.
  --tokenizer_cache_dir TOKENIZER_CACHE_DIR
                        The tokenizer cache directory.
  --block_size BLOCK_SIZE
                        The block size.
  --min_length MIN_LENGTH
                        The minimum length.
  --max_length MAX_LENGTH
                        The maximum length.
  --num_shards NUM_SHARDS
                        The number of shards.
  --shard_index SHARD_INDEX
                        The shard index.
  --seed SEED           The random seed.
  --cc100_data_path CC100_DATA_PATH
                        The path to the CC-100 dataset.
  --use_filter          Whether to use the filter.
```

**Example 1**: The following will preprocess the Arabic OSCAR data using an Arabic tokenizer and save it under output_dir. Please modify only the paths to reproduce our results.
```bash
#!/bin/bash

export HF_DATASETS_CACHE="/path/to/another/directory"

cd /path/to/llm-vocab-adaptation/preprocessing/src
python preprocess.py \
    --target_lang ar \
    --cache_dir /path/to/data/cache/dir  \
    --output_dir /path/to/preprocessed/data/dir \
    --tokenizer_name_or_path aubmindlab/aragpt2-base \
    --block_size 1024 \
    --min_length 5 \
    --max_length 1024 \
    --num_shards 5 \
    --shard_index 0
```

**Example 2**: The following will preprocess the Swahili CC-100 data using a Swahili tokenizer and save it under output_dir.
```bash
#!/bin/bash
export HF_DATASETS_CACHE="/path/to/another/directory"

cd /path/to/llm-vocab-adaptation/preprocessing/src
python preprocess.py \
    --target_lang sw \
    --cache_dir /path/to/data/cache/dir  \
    --output_dir /path/to/preprocessed/data/dir \
    --tokenizer_name_or_path benjamin/gpt2-wechsel-swahili \
    --cc100_data_path /path/to/cc100/sw.txt \
    --block_size 1024 \
    --min_length 5 \
    --max_length 1024 \
    --num_shards 1
```


### 3. Training a fastText model
To use FOCUS, you need to train a fastText model for each language.
```bash
$ python src/train_fasttext.py -h
usage: Train a fasttext model with the CC-100 corpus. [-h]
                                                      --tokenizer_name_or_path
                                                      TOKENIZER_NAME_OR_PATH
                                                      [--tokenizer_cache_dir TOKENIZER_CACHE_DIR]
                                                      --text_path TEXT_PATH
                                                      [--data_cache_dir DATA_CACHE_DIR]
                                                      [--min_length MIN_LENGTH]
                                                      --target_lang
                                                      {ja,de,sw,ar}
                                                      --model_cache_dir
                                                      MODEL_CACHE_DIR
                                                      --subword_type
                                                      {bpe,unigram}
                                                      [--restart_from_cache]

options:
  -h, --help            show this help message and exit
  --tokenizer_name_or_path TOKENIZER_NAME_OR_PATH
                        The tokenizer name or path.
  --tokenizer_cache_dir TOKENIZER_CACHE_DIR
                        The tokenizer cache directory.
  --text_path TEXT_PATH
                        The path to the text file.
  --data_cache_dir DATA_CACHE_DIR
                        The cache directory.
  --min_length MIN_LENGTH
                        The minimum length.
  --target_lang {ja,de,sw,ar}
                        The target language.
  --model_cache_dir MODEL_CACHE_DIR
                        The model cache directory.
  --subword_type {bpe,unigram}
                        The subword type.
  --restart_from_cache  Whether to restart from the cache.
```

**Example**: The following will train an Arabic fastText model and save it under model_cache_dir.
```bash
#!/bin/bash

export TRANSFORMERS_CACHE="/path/to/another/directory"

cd /path/to/llm-vocab-adaptation/preprocessing/src
python train_fasttext.py \
    --tokenizer_name_or_path aubmindlab/aragpt2-base \
    --tokenizer_cache_dir /path/to/cache/dir \
    --text_path /mnt/parscratch/users/acp23ay/private/datasets/ar.txt \
    --data_cache_dir /path/to/data/cache/dir \
    --min_length 5 \
    --target_lang ar \
    --model_cache_dir /path/to/model/cache/dir \
    --subword_type bpe
```
