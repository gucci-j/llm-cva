import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

class CLPEmbeddingInitializer:
    """Initialize the target model with CLP."""
    def __init__(
        self,
        source_model: AutoModelForCausalLM,
        helper_model: AutoModelForCausalLM,
        source_tokenizer: AutoTokenizer,
        target_tokenizer: AutoTokenizer,
        copy_special_tokens: bool = False,
        seed: int = 42
    ):
        self.source_model = source_model
        self.helper_model = helper_model
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.copy_special_tokens = copy_special_tokens
        self.seed = seed


    def __call__(self) -> AutoModelForCausalLM:
        """Initialize the target model with CLP.

        Raises:
            ValueError: No overlapping tokens between source and target model.

        Returns:
            AutoModelForCausalLM: The target model initialized with CLP.
        
        References:
            - https://github.com/malteos/clp-transfer
        """
        #####
        # Get the source and helper embeddings
        #####
        source_embeddings = self.source_model.get_input_embeddings().weight.detach().numpy()
        helper_embeddings = self.helper_model.get_input_embeddings().weight.detach().numpy()

        #####
        # Get the token to index mappings
        #####
        target_token_to_idx = {t: i for t, i in self.target_tokenizer.get_vocab().items()}
        source_token_to_idx = {t: i for t, i in self.source_tokenizer.get_vocab().items()}

        #####
        # Generate random embeddings
        # See https://github.com/malteos/clp-transfer
        #####
        np.random.seed(self.seed)
        target_embeddings = np.random.normal(
            np.mean(source_embeddings, axis=0), 
            np.std(source_embeddings, axis=0), 
            (
                len(self.target_tokenizer.get_vocab()), 
                source_embeddings.shape[1]
            )
        )
        logger.info(source_embeddings.shape)
        logger.info(target_embeddings.shape)
            
        #####
        # Initialize the target embeddings with the source embeddings for overlapping tokens
        #####
        # Get the overlapping tokens
        target_tokens = set(self.target_tokenizer.get_vocab().keys())
        source_tokens = set(self.source_tokenizer.get_vocab().keys())
        overlapping_tokens = target_tokens & source_tokens
        overlapping_tokens_list = list(overlapping_tokens)
        if not overlapping_tokens:
            raise ValueError('No overlapping tokens! Cannot initialize the target model with CLP.')

        # Copy the source embeddings for overlapping tokens
        for token in overlapping_tokens:
            target_embeddings[target_token_to_idx[token]] = \
                source_embeddings[source_token_to_idx[token]]
        
        #####
        # Initialize the missing tokens with the weighted mean of the source embeddings
        #####
        # Get missing tokens
        missing_tokens = target_tokens - source_tokens
        missing_tokens_list = list(missing_tokens)

        if not missing_tokens:
            logger.info('No missing tokens!')
        else:
            # Get the embeddings for the missing tokens and overlapping tokens in the helper model
            helper_missing_token_embeddings = \
                helper_embeddings[[target_token_to_idx[t] for t in missing_tokens_list], :]
            helper_overlapping_token_embeddings = \
                helper_embeddings[[target_token_to_idx[t] for t in overlapping_tokens_list], :]
            
            # Get the embeddings for the overlapping tokens in the source model
            overlapping_tokens_idxs = \
                [source_token_to_idx[t] for t in overlapping_tokens_list]
            overlapping_token_vecs = source_embeddings[overlapping_tokens_idxs, :] # -> (len(overlapping_tokens), embedding_dim)

            # Calculate the cosine similarity between the missing tokens and overlapping tokens in the helper model
            cos_sims = cosine_similarity(
                helper_missing_token_embeddings, 
                helper_overlapping_token_embeddings
            ) # -> (len(missing_tokens), len(overlapping_tokens))

            # Initialize the target embeddings with the weighted mean of the overlapping tokens in the source model
            for index, token in enumerate(tqdm(missing_tokens_list)):
                # Get the cosine similarity scores for the missing token
                token_cos_sim = cos_sims[index] # -> (len(overlapping_tokens),)
                normed_token_cos_sim = token_cos_sim / token_cos_sim.sum() # -> (len(overlapping_tokens),)
                
                # Calculate the weighted mean of the overlapping tokens in the source model
                target_vec = np.average(
                    overlapping_token_vecs, axis=0, weights=normed_token_cos_sim
                ) # -> (embedding_dim,)

                # Set the embedding of the current missing token to the weighted mean
                target_embeddings[target_token_to_idx[token]] = target_vec

        #####
        # Set the embeddings for the special tokens
        # - See https://github.com/cmdowney88/embeddingstructure
        #####
        if self.copy_special_tokens:
            # Get the special tokens
            source_special_tokens_map = self.source_tokenizer.special_tokens_map
            target_special_tokens_map = self.target_tokenizer.special_tokens_map
            
            # Copy the source embeddings for the special tokens
            for special_token_name, target_special_token in target_special_tokens_map.items():
                if special_token_name in source_special_tokens_map:
                    source_special_token = source_special_tokens_map[special_token_name]
                    source_special_token_idx = source_token_to_idx[source_special_token]
                    target_special_token_idx = target_token_to_idx[target_special_token]
                    target_embeddings[target_special_token_idx] = \
                        source_embeddings[source_special_token_idx]

        #####
        # Initialize the target model
        #####
        target_model = self.source_model
        target_model.resize_token_embeddings(
            len(self.target_tokenizer), 
            pad_to_multiple_of=32 # See https://github.com/huggingface/transformers/issues/26303
        )
        target_model.get_input_embeddings().weight.data = torch.from_numpy(target_embeddings)
        target_model.tie_weights()
        logger.info(target_model.get_input_embeddings().weight.data.shape)

        return target_model
