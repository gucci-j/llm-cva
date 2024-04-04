import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RandomEmbeddingInitializer:
    """Initialize the target model with random embeddings."""
    def __init__(
        self,
        source_model: AutoModelForCausalLM,
        source_tokenizer: AutoTokenizer,
        target_tokenizer: AutoTokenizer,
        seed: int = 42
    ):
        self.source_model = source_model
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.seed = seed


    def __call__(self) -> AutoModelForCausalLM:
        # Get the source embeddings
        source_embeddings = self.source_model.get_input_embeddings().weight.detach().numpy()
        
        # Generate random embeddings
        # See https://github.com/malteos/clp-transfer
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

        # Initialize the target model
        target_model = self.source_model
        target_model.resize_token_embeddings(
            len(self.target_tokenizer), 
            pad_to_multiple_of=32 # See https://github.com/huggingface/transformers/issues/26303
        )
        target_model.get_input_embeddings().weight.data = torch.from_numpy(target_embeddings)
        target_model.tie_weights()
        logger.info(target_model.get_input_embeddings().weight.data.shape)

        return target_model
