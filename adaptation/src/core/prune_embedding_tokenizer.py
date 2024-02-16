import copy
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import json
from collections import defaultdict

import datasets
import numpy as np
import torch
from dynamic_embedding_pruning.data.token_converter import TokenConverter
from dynamic_embedding_pruning.data.token_counter import TokenCounter
from tokenizers import models
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class UnusedEmbeddingTokenizerPruner:
    """Prune unused embeddings and tokenizer vocab."""
    def __init__(
        self,
        source_model: AutoModelForCausalLM,
        source_tokenizer: AutoTokenizer,
        dataset: datasets.Dataset | datasets.DatasetDict
    ):
        self.source_model = source_model
        self.source_tokenizer = source_tokenizer
        self.dataset = dataset
        self.token_counter = TokenCounter(
            batch_size=1000,
            num_proc=None,
            load_from_cache_file=None,
        )
    
    
    @staticmethod
    def tailor_dataset(
        source_tokenizer: AutoTokenizer,
        pruned_tokenizer: AutoTokenizer,
        dataset: datasets.Dataset | datasets.DatasetDict
    ) -> datasets.Dataset | datasets.DatasetDict:
        """Tailor the dataset to the pruned tokenizer.

        Args:
            source_tokenizer (AutoTokenizer): A source tokenizer.
            pruned_tokenizer (AutoTokenizer): A pruned tokenizer.
            dataset (datasets.Dataset | datasets.DatasetDict): A dataset tokenized with the source tokenizer.

        Returns:
            datasets.Dataset | datasets.DatasetDict: A dataset tokenized with the pruned tokenizer.
        """
        source_vocab = source_tokenizer.get_vocab()
        pruned_vocab = pruned_tokenizer.get_vocab()
        token_map = [-1] * len(source_vocab)
        for token, index in source_vocab.items():
            if pruned_vocab.get(token) is not None:
                pruned_index = pruned_vocab[token]
                token_map[index] = pruned_index
        token_map = np.asarray(token_map)
        token_converter = TokenConverter(
            batch_size=1000,
            num_proc=None,
            load_from_cache_file=None,
        )
        dataset = token_converter.convert(
            dataset=dataset,
            token_map=token_map,
            columns=["input_ids", "labels"],
        )
        return dataset
                

    def __call__(self) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Prune the unused embeddings and tokenizer vocab.

        Returns:
            AutoModelForCausalLM: The pruned model.
            AutoTokenizer: The pruned tokenizer.
        
        References:
            - https://github.com/asahi417/lm-vocab-trimmer
            - https://github.com/mlsw/dynamic-embedding-pruning
        """
        #####
        # Get used tokens id list
        #####
        token_counts = self.token_counter.count(
            dataset=self.dataset,
            vocabulary_size=len(self.source_tokenizer),
            columns=[self.source_model.main_input_name],
        )
        used_token_ids = np.flatnonzero(token_counts).tolist()
        unused_token_ids = np.flatnonzero(token_counts == 0).tolist()
        vocab = self.source_tokenizer.get_vocab()
        id_to_token = {index: token for token, index in vocab.items()}
        new_vocab = {id_to_token[i]: i for i in used_token_ids}
        
        #####
        # Create a new pruned tokenizer
        #####
        vocab = dict(
            zip(self.source_tokenizer.all_special_tokens, 
                self.source_tokenizer.all_special_ids)
        )
        vocab.update(new_vocab)
        new_vocab_id = sorted(vocab.values())
        new_vocab = list(vocab.keys())
        vocab = dict(sorted(vocab.items(), key=lambda item: item[1]))
        
        #####
        # Prune embeddings
        #####
        # copy the model
        pruned_model = copy.deepcopy(self.source_model)

        # set input embedding
        input_embedding = self.source_model.get_input_embeddings()
        pruned_model.set_input_embeddings(
            torch.nn.Embedding.from_pretrained(
                input_embedding.weight[new_vocab_id]
            )
        )

        # set output embedding
        output_embedding = self.source_model.get_output_embeddings()
        if output_embedding is not None:
            new_weight = output_embedding.weight[new_vocab_id]
            new_bias = None
            if output_embedding.bias is not None:
                new_bias = output_embedding.bias[new_vocab_id]
            with torch.no_grad():
                linear = torch.nn.modules.linear.Linear(
                    in_features=new_weight.shape[1], 
                    out_features=new_weight.shape[0], 
                    bias=output_embedding.bias is not None
                )
                linear.weight.copy_(new_weight)
                if new_bias is not None:
                    linear.bias.copy_(new_bias)
            pruned_model.set_output_embeddings(linear)

        #####
        # Resize the embedding layer
        #####
        pruned_model.config.vocab_size = len(new_vocab_id)
        pruned_model.resize_token_embeddings(
            pruned_model.config.vocab_size,
            #pad_to_multiple_of=32 # See https://github.com/huggingface/transformers/issues/26303
        )
        print(pruned_model.get_input_embeddings().weight.data.shape)
        print(pruned_model)

        #####
        # Reassign ids in the tokenizer
        #####
        vocab = {
            token: index for index, token in enumerate(vocab.keys())
        }

        #####
        # Create a new pruned tokenizer
        #####
        pruned_tokenizer = copy.deepcopy(self.source_tokenizer)
        old_vocab = self.source_tokenizer.get_vocab()
        model_state = json.loads(pruned_tokenizer.backend_tokenizer.model.__getstate__())
        
        # Take care of the vocab
        is_dict = False
        if type(model_state['vocab']) is dict:
            is_dict = True
            model_state['vocab'] = list(model_state['vocab'].items())
        new_state = []
        for w, s in tqdm(model_state['vocab']): # token, index
            if w in vocab:
                new_state.append((w, vocab[w]))
        if is_dict:
            new_state = dict(new_state)
        model_state['vocab'] = new_state

        model_class = getattr(models, model_state.pop("type"))
        if model_class == models.BPE:
            # Some workarounds for BPE
            # See https://huggingface.co/docs/tokenizers/main/en/api/models#tokenizers.models.BPE
            # See https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt
            # See https://github.com/hannawong/vocab_prune/blob/main/prune_one_by_one_fast.py
            if model_state.get("merges") is not None:
                # Get mapping for merges
                merges = model_state.pop("merges")
                full_merges_token_to_id = {}
                merges_token_to_id = defaultdict(list)
                for index in range(len(merges)):
                    token_1, token_2 = merges[index].split()
                    full_merges_token_to_id[token_1 + token_2] = index
                    merges_token_to_id[token_1].append((index, 0))
                    merges_token_to_id[token_2].append((index, 1))
                
                # Delete unused merges
                def delete_token(token):
                    if token not in old_vocab:
                        return
                    del old_vocab[token]
                    if full_merges_token_to_id.get(token) is not None:
                        del full_merges_token_to_id[token]
                    token_pos_list = merges_token_to_id[token]
                    for index, pos in token_pos_list:
                        next_token = "".join(merges[index].split())
                        delete_token(next_token)
                for index in tqdm(unused_token_ids):
                    token = id_to_token[index]
                    delete_token(token)

                # Reassign ids in the tokenizer
                new_merges = []
                for token in full_merges_token_to_id:
                    token_1, token_2 = merges[full_merges_token_to_id[token]].split()
                    new_merges.append((token_1, token_2))
                model_state["merges"] = new_merges

            pruned_tokenizer.backend_tokenizer.model = model_class(
                vocab=model_state["vocab"],
                merges=model_state["merges"],
                fuse_unk=False,
            )
        else:
            # Unigram, WordPiece (not tested), WordLevel (not tested), CharLevel (not tested)
            pruned_tokenizer.backend_tokenizer.model = model_class(**model_state)
        
        return pruned_model, pruned_tokenizer
