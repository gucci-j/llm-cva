An Empirical Study on Cross-lingual Vocabulary Adaptation for Efficient Generative LLM Inference
===

This is the official code for the paper titled "[An Empirical Study on Cross-lingual Vocabulary Adaptation for Efficient Generative LLM Inference](https://arxiv.org/abs/2402.10712)." For reproduction, please refer to [Reproduction](#reproduction). We plan to release adapted models in the future.

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
Adapted models will be available in HF Hub in the future. We also plan to release LAPT models for reference.

| Approach | BLOOM-1B | BLOOM-7B | TigerBot-7B | Mistral-7B |
| :- | :--: | :--: | :--: | :--: |
| LAPT | TBA | TBA | TBA | TBA |
| Random | TBA | - | - | - |
| CLP | TBA | - | - | - |
| Heuristics | TBA | TBA | TBA | TBA |
| FOCUS | TBA | - | - | - |
| CLP+ | TBA | TBA | TBA | TBA |


## License
[MIT License](./LICENSE)


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
