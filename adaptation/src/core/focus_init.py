import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import entmax
import numpy as np
import torch
from fastdist import fastdist
import fasttext
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class FOCUSEmbeddingInitializer:
    """Initialize the target model with FOCUS."""
    def __init__(
        self,
        source_model: AutoModelForCausalLM,
        source_tokenizer: AutoTokenizer,
        target_tokenizer: AutoTokenizer,
        fasttext_model: fasttext.FastText,
        seed: int = 42
    ):
        self.source_model = source_model
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.fasttext_model = fasttext_model
        self.seed = seed
    

    def is_very_rare_token(self, token: str) -> bool:
        return token not in self.fasttext_model or np.absolute(self.fasttext_model[token]).sum() == 0
    

    def focus_missing_token_initialization(
        self,
        missing_tokens_list: list[str],
        overlapping_tokens_list: list[str],
        token_to_auxiliary_embeddings: dict[str, np.ndarray],
        source_embeddings: np.ndarray,
        source_token_to_idx: dict[str, int],
        target_embeddings: np.ndarray,
        target_token_to_idx: dict[str, int],
    ) -> torch.Tensor:
        #####
        # Get the embeddings for the missing tokens and overlapping tokens in FastText
        #####
        missing_auxiliary_embedding_matrix = np.asarray(
            [token_to_auxiliary_embeddings[t] for t in missing_tokens_list],
            dtype="float32"
        ) # -> (len(missing_tokens), fasttext_embedding_dim)
        overlapping_auxiliary_embedding_matrix = np.asarray(
            [token_to_auxiliary_embeddings[t] for t in overlapping_tokens_list],
            dtype="float32"
        ) # -> (len(overlapping_tokens), fasttext_embedding_dim)

        #####
        # Compute the cosine similarity between the missing tokens and overlapping tokens in FastText
        #####
        cos_sims = fastdist.cosine_matrix_to_matrix(
            missing_auxiliary_embedding_matrix,
            overlapping_auxiliary_embedding_matrix,
        ) # -> (len(missing_tokens), len(overlapping_tokens))

        # Not needed anymore, save memory
        del missing_auxiliary_embedding_matrix
        del overlapping_auxiliary_embedding_matrix

        #####
        # Compute the weighted mean of the overlapping tokens in the source model
        #####
        logger.debug("Computing new embeddings...")
        # Get the embeddings for the overlapping tokens in the source model
        overlapping_tokens_idxs = \
            [source_token_to_idx[t] for t in overlapping_tokens_list]
        overlapping_token_vecs = torch.from_numpy(
            source_embeddings[overlapping_tokens_idxs, :]
        ) # -> (len(overlapping_tokens), source_embedding_dim)

        # Initialize the target embeddings with the weighted mean of the overlapping tokens in the source model
        for index, token in enumerate(tqdm(missing_tokens_list)):
            # Get the cosine similarity scores for the missing token
            token_cos_sim = entmax.sparsemax(
                torch.from_numpy(cos_sims[index])
            ) # -> (len(overlapping_tokens),)
            logger.info(f"token_cos_sim: {token_cos_sim.shape}")
            
            # Get the weighted mean of the overlapping tokens in the source model
            mask = token_cos_sim > 0.0
            masked_token_cos_sim = token_cos_sim[mask] # -> (num_token_cos_sim_positive,)
            masked_overlapping_token_vecs = overlapping_token_vecs[mask] # -> (num_token_cos_sim_positive, source_embedding_dim)
            logger.info(f"masked_token_cos_sim: {masked_token_cos_sim.shape}")
            logger.info(f"masked_overlapping_token_vecs: {masked_overlapping_token_vecs.shape}")

            weighted_src_embs = torch.mul(
                masked_overlapping_token_vecs, 
                masked_token_cos_sim.unsqueeze(1)
            ) # -> (num_token_cos_sim_positive, source_embedding_dim)
            logger.info(f"weighted_src_embs: {weighted_src_embs.shape}")
            logger.info("=" * 10)
            weighted_mean = torch.sum(weighted_src_embs, dim=0) # -> (source_embedding_dim,)
            
            # Set the embedding of the current missing token to the weighted mean
            target_embeddings[target_token_to_idx[token]] = weighted_mean.detach().numpy()

        return target_embeddings


    def __call__(self) -> AutoModelForCausalLM:
        """Initialize the target model with FOCUS.

        Raises:
            ValueError: No overlapping tokens between source and target model.

        Returns:
            AutoModelForCausalLM: The target model initialized with FOCUS.
        
        References:
            - https://github.com/malteos/clp-transfer
            - https://github.com/konstantinjdobler/focus
        """
        #####
        # Get the source embeddings
        #####
        source_embeddings = self.source_model.get_input_embeddings().weight.detach().numpy()

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
            raise ValueError('No overlapping tokens! Cannot initialize the target model with FOCUS.')

        # Copy the source embeddings for overlapping tokens
        for token in overlapping_tokens:
            target_embeddings[target_token_to_idx[token]] = \
                source_embeddings[source_token_to_idx[token]]

        #####
        # Generate the auxiliary embeddings for overlapping tokens
        #####
        token_to_auxiliary_embeddings = {}
        temp_overlapping_tokens_list = []
        for token in overlapping_tokens_list:
            if self.is_very_rare_token(token):
                logger.info(f"Token '{token}' is very rare. Skip this token.")
                continue
            else:
                token_to_auxiliary_embeddings[token] = self.fasttext_model[token]
                temp_overlapping_tokens_list.append(token)
        overlapping_tokens_list = temp_overlapping_tokens_list

        #####
        # Initialize the missing tokens with FOCUS
        #####
        # Get missing tokens
        missing_tokens = target_tokens - source_tokens
        missing_tokens_list = list(missing_tokens)

        if not missing_tokens:
            logger.info('No missing tokens!')
        else:
            # Generate the auxiliary embeddings for missing tokens
            temp_missing_tokens_list = []
            for token in missing_tokens:
                if self.is_very_rare_token(token):
                    logger.info(f"Token '{token}' is very rare. Skip this token.")
                    continue
                else:
                    token_to_auxiliary_embeddings[token] = self.fasttext_model[token]
                    temp_missing_tokens_list.append(token)
            missing_tokens_list = temp_missing_tokens_list
        
            # Initialize the missing tokens with FOCUS
            target_embeddings = self.focus_missing_token_initialization(
                missing_tokens_list,
                overlapping_tokens_list,
                token_to_auxiliary_embeddings,
                source_embeddings,
                source_token_to_idx,
                target_embeddings,
                target_token_to_idx,
            )

        #####
        # Initialize the target model
        #####
        target_model = self.source_model
        target_model.resize_token_embeddings(
            len(self.target_tokenizer), 
            pad_to_multiple_of=32 # See https://github.com/huggingface/transformers/issues/26303
        )
        target_model.get_input_embeddings().weight.data = torch.from_numpy(target_embeddings)
        logger.info(target_model.get_input_embeddings().weight.data.shape)

        return target_model
