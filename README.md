An Empirical Study on Cross-lingual Vocabulary Adaptation for Efficient Generative LLM Inference
===

This is the official code for the paper titled "[An Empirical Study on Cross-lingual Vocabulary Adaptation for Efficient Generative LLM Inference](https://arxiv.org/abs/2402.10712)." For reproduction, please refer to [Reproduction](#reproduction).

## Requirements
* Python 3.10 or later
* PyTorch v2.1.0 or later
* transformers==4.35.0.dev0
* peft==0.6.2
* datasets==2.15.0
* evaluate==0.4.1
* bitsandbytes==0.41.2.post2
* scipy==1.11.4
* scikit-learn==1.3.2
* sentencepiece
* seaborn==0.13.0
* fasttext: Please visit https://github.com/facebookresearch/fastText to install this package.
* jupyterlab
* sumeval
* janome
* protobuf==4.25.1
* entmax==1.1
* fastdist==1.1.6
* dynamic_embedding_pruning==0.0.1
* rouge-score==0.1.2
* numba==0.58.1
* tensorboardX==2.6.2.2
* pyarabic==0.6.15
* rouge==1.0.1


## Installation
After manually installing `PyTorch`, `transformers`, and `fasttext`, please run the following.
```
pip install -r requirements.txt
```

## Reproduction
### 1. Preprocessing  
See [Preprocessing](./preprocessing/).

### 2. Target Model Initialization
See [Adaptation](./adaptation/).
 
### 3. LAPT
See [Tuning](./tuning/).

### 4. Evaluation
See [Evaluation](./eval/).


## Adapted Models

All models are available on the Hugging Face Model Hub.

| Approach | BLOOM-1B | BLOOM-7B | TigerBot-7B | Mistral-7B |
| :- | :--: | :--: | :--: | :--: |
| LAPT | [de](https://huggingface.co/atsuki-yamaguchi/bloom-1b1-lapt-de)/[ja](https://huggingface.co/atsuki-yamaguchi/bloom-1b1-lapt-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/bloom-1b1-lapt-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/bloom-1b1-lapt-sw) | [de](https://huggingface.co/atsuki-yamaguchi/bloom-7b1-lapt-de)/[ja](https://huggingface.co/atsuki-yamaguchi/bloom-7b1-lapt-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/bloom-7b1-lapt-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/bloom-7b1-lapt-sw)  | [de](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-lapt-de)/[ja](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-lapt-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-lapt-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-lapt-sw)  | [de](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-lapt-de)/[ja](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-lapt-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-lapt-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-lapt-sw)  |
| Random | [de](https://huggingface.co/atsuki-yamaguchi/bloom-1b1-random-de)/[ja](https://huggingface.co/atsuki-yamaguchi/bloom-1b1-random-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/bloom-1b1-random-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/bloom-1b1-random-sw) | [de](https://huggingface.co/atsuki-yamaguchi/bloom-7b1-random-de)/[ja](https://huggingface.co/atsuki-yamaguchi/bloom-7b1-random-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/bloom-7b1-random-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/bloom-7b1-random-sw) | [de](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-random-de)/[ja](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-random-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-random-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-random-sw) | [de](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-random-de)/[ja](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-random-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-random-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-random-sw) |
| CLP | [de](https://huggingface.co/atsuki-yamaguchi/bloom-1b1-clp-de)/[ja](https://huggingface.co/atsuki-yamaguchi/bloom-1b1-clp-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/bloom-1b1-clp-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/bloom-1b1-clp-sw) | [de](https://huggingface.co/atsuki-yamaguchi/bloom-7b1-clp-de)/[ja](https://huggingface.co/atsuki-yamaguchi/bloom-7b1-clp-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/bloom-7b1-clp-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/bloom-7b1-clp-sw) | [de](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-clp-de)/[ja](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-clp-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-clp-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-clp-sw) | [de](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-clp-de)/[ja](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-clp-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-clp-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-clp-sw) |
| Heuristics | [de](https://huggingface.co/atsuki-yamaguchi/bloom-1b1-heuristics-de)/[ja](https://huggingface.co/atsuki-yamaguchi/bloom-1b1-heuristics-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/bloom-1b1-heuristics-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/bloom-1b1-heuristics-sw) | [de](https://huggingface.co/atsuki-yamaguchi/bloom-7b1-heuristics-de)/[ja](https://huggingface.co/atsuki-yamaguchi/bloom-7b1-heuristics-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/bloom-7b1-heuristics-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/bloom-7b1-heuristics-sw) | [de](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-heuristics-de)/[ja](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-heuristics-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-heuristics-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-heuristics-sw) | [de](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-heuristics-de)/[ja](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-heuristics-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-heuristics-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-heuristics-sw) |
| FOCUS | [de](https://huggingface.co/atsuki-yamaguchi/bloom-1b1-focus-de)/[ja](https://huggingface.co/atsuki-yamaguchi/bloom-1b1-focus-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/bloom-1b1-focus-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/bloom-1b1-focus-sw) | [de](https://huggingface.co/atsuki-yamaguchi/bloom-7b1-focus-de)/[ja](https://huggingface.co/atsuki-yamaguchi/bloom-7b1-focus-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/bloom-7b1-focus-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/bloom-7b1-focus-sw) | [de](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-focus-de)/[ja](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-focus-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-focus-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-focus-sw) | [de](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-focus-de)/[ja](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-focus-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-focus-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-focus-sw) |
| CLP+ | [de](https://huggingface.co/atsuki-yamaguchi/bloom-1b1-clpp-de)/[ja](https://huggingface.co/atsuki-yamaguchi/bloom-1b1-clpp-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/bloom-1b1-clpp-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/bloom-1b1-clpp-sw) | [de](https://huggingface.co/atsuki-yamaguchi/bloom-7b1-clpp-de)/[ja](https://huggingface.co/atsuki-yamaguchi/bloom-7b1-clpp-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/bloom-7b1-clpp-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/bloom-7b1-clpp-sw) | [de](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-clpp-de)/[ja](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-clpp-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-clpp-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-clpp-sw) | [de](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-clpp-de)/[ja](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-clpp-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-clpp-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-clpp-sw) |

### + Output projection layer initialization
We also release some TigerBot-7B and Mistral-7B models whose output layer is initialized according to each corresponding vocabulary initialization method instead of random initialization.

| Approach | TigerBot-7B | Mistral-7B |
| :- | :--: | :--: |
| Heuristics | [de](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-heuristics-untied-de)/[ja](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-heuristics-untied-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-heuristics-untied-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-heuristics-untied-sw) | [de](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-heuristics-untied-de)/[ja](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-heuristics-untied-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-heuristics-untied-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-heuristics-untied-sw) |
| CLP+ | [de](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-clpp-untied-de)/[ja](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-clpp-untied-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-clpp-untied-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/tigerbot-7b-base-clpp-untied-sw) | [de](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-clpp-untied-de)/[ja](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-clpp-untied-ja)/[ar](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-clpp-untied-ar)/[sw](https://huggingface.co/atsuki-yamaguchi/Mistral-7B-v0.1-clpp-untied-sw) |

### fastText weights
Pre-trained fastText weights, used for FOCUS initialization, are uploaded with BLOOM-1B FOCUS models.

## License
[MIT License](./LICENSE)

### Adapted Tokenizer
Note that adapted tokenizers were obtained from the following for each language:
* German: https://huggingface.co/malteos/gpt2-xl-wechsel-german
* Japanese: https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-ppo
* Arabic: https://huggingface.co/aubmindlab/aragpt2-base
* Swahili: https://huggingface.co/benjamin/gpt2-wechsel-swahili 

Due to the license limitation of the Arabic tokenizer, we have excluded the Arabic tokenizer from each corresponding adapted model. To use it, please make sure to download the tokenizer beforehand from the above link.


## Citation
If you find this work useful, please cite the following:
```
@article{yamaguchi2024empirical,
  title={An Empirical Study on Cross-lingual Vocabulary Adaptation for Efficient Generative {LLM} Inference}, 
  author={Atsuki Yamaguchi and Aline Villavicencio and Nikolaos Aletras},
  journal={ArXiv},
  year={2024},
  volume={abs/2402.10712},
  url={https://arxiv.org/abs/2402.10712}
}
```
