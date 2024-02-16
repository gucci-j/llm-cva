import copy
import json
import logging
from collections import defaultdict
import math

import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)


def hex2dec(hex_str: str) -> int:
    """Convert Unicode hexadecimal string to base-10 int."""
    return int(hex_str, 16)


def get_ord2script(scriptfile: str) -> dict[int, str]:
    """Create a dictionary mapping Unicode code points to scripts.

    Args:
        scriptfile (str): Path to the script file.

    Returns:
        dict[int, str]: Dictionary mapping Unicode code points to scripts. 
    
    Description:
        Return dictionary (key: Unicode decimal, val: script of corresponding 
        character according to Unicode documentation).
    
    Reference:
        - https://github.com/cmdowney88/embeddingstructure
    """
    # Read in the script file
    with open(scriptfile, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()

    # Create dictionary
    ord2script = dict()
    for line in lines:
        if line[0] != '#':
            items = line.split()
            if len(items) > 0:
                script = items[2]
                encoding = items[0]
                if '..' in encoding:
                    start_stop = encoding.split('..')
                    start = hex2dec(start_stop[0])
                    stop = hex2dec(start_stop[1])
                    for dec_encoding in range(start, stop + 1):
                        ord2script[dec_encoding] = script
                else:
                    dec_encoding = hex2dec(encoding)
                    ord2script[dec_encoding] = script

    return ord2script


def top_script(
    token: str, 
    ord2script: dict[int, str],
) -> str:
    """Return most-used script within token (str), using ord2script (dict) to retrieve the script of each char in token.

    Args:
        token (str): A target token.
        ord2script (dict[int, str]): A dictionary mapping Unicode code points to scripts.

    Returns:
        str: The most-used script within the token.
    """
    script_counts = defaultdict(lambda: 0)
    for character in token:
        try:
            script = ord2script[ord(character)]
        except KeyError:
            script = 'UNK'
        script_counts[script] += 1

    return max(script_counts, key=lambda x: script_counts[x])


def get_script_to_ids(
    token_to_idx: dict[str, int], 
    special_tokens: list[str],
    ord_to_script: dict[int, str], 
    word_position: bool,
    whitespace: str = 'Ġ'
) -> dict[str, list[int]]:
    """Create a dictionary mapping scripts to token indices.

    Args:
        token_to_idx (dict[str, int]): A dictionary mapping tokens to indices.
        special_tokens (list[str]): A list of special tokens.
        ord_to_script (dict[int, str]): A dictionary mapping Unicode code points to scripts.
        word_position (bool): Whether to include word position in the script.
        whitespace (str, optional): The whitespace character. Defaults to '▁'.

    Returns:
        dict[str, list[int]]: A dictionary mapping scripts to token indices.
    """
    script_to_ids = defaultdict(list)
    
    # get script for each token in XLM-R's vocab
    for token, index in token_to_idx.items():
        # see if token is a special token
        if token in special_tokens:
            script_to_ids['special'].append(index)
            continue
            
        # leave out the preceding whitespace when identifying token script
        token_text = token[1:] if token[0] == whitespace and len(token) > 1 else token
        
        # identify top script for the token based on characters and Unicode mapping
        script = top_script(token_text, ord_to_script)
        if word_position == True:
            if token[0] == whitespace:
                script += '_initial'
            else:
                script += '_medial'
        script_to_ids[script].append(index)
        
    return script_to_ids


def initialize_by_category_means(
    categories: list[str],
    means: torch.Tensor,
    stds: torch.Tensor,
    category_to_indices: dict[str, int],
    matrix: torch.Tensor,
    categories_to_omit: list[str] = ['special']
) -> torch.Tensor:
    # only initialize the categories that are in both the source and target data
    category_intersection = set(categories).intersection(set(category_to_indices.keys()))
    for category in category_intersection:
        if category in categories_to_omit:
            continue
        category_index = categories.index(category)
        try:
            category_distribution = torch.distributions.Normal(
                means[category_index], stds[category_index]
            )
            for index in category_to_indices[category]:
                matrix[index] = category_distribution.sample()
        except ValueError:
            continue
    return matrix


def round_to_nearest_multiple(vocabulary_size, multiple):
    rounded_size = math.ceil(vocabulary_size / multiple) * multiple
    return rounded_size


class HeuristicsEmbeddingInitializer:
    """Initialize the target model with heuristics based methods."""
    def __init__(
        self,
        source_model: AutoModelForCausalLM,
        source_tokenizer: AutoTokenizer,
        target_tokenizer: AutoTokenizer,
        unicode_script_file: str,
        seed: int = 42
    ):
        self.source_model = source_model
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.unicode_script_file = unicode_script_file
        self.seed = seed


    def reinitialize_by_script(
        self,
        source_token_to_idx: dict[str, int],
        target_token_to_idx: dict[str, int],
        source_special_tokens: list[str],
        target_special_tokens: list[str],
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        unicode_script_file: str,
        word_position: bool = True,
        source_whitespace: str = '▁',
        target_whitespace: str = '▁'
    ) -> np.ndarray:
        """Reinitialize the target embeddings by script.

        Args:
            source_token_to_idx (dict[str, int]): A source dictionary mapping tokens to indices.
            target_token_to_idx (dict[str, int]): A target dictionary mapping tokens to indices.
            source_special_tokens (list[str]): A list of source special tokens.
            target_special_tokens (list[str]): A list of target special tokens.
            source_embeddings (np.ndarray): A source embedding matrix.
            target_embeddings (np.ndarray): A target embedding matrix.
            unicode_script_file (str): Path to the script file.
            word_position (bool, optional): Whether to include word position in the script. Defaults to True.

        Returns:
            np.ndarray: A target embedding matrix.
        """
        #####
        # Categorise the tokens by Unicode block
        #####
        ord_to_script = get_ord2script(unicode_script_file)
        
        #####
        # Get the script to ids mappings
        #####
        source_script_to_ids = get_script_to_ids(
            source_token_to_idx,
            source_special_tokens,
            ord_to_script, 
            word_position,
            source_whitespace
        )
        all_source_scripts = list(source_script_to_ids.keys())
        target_script_to_ids = get_script_to_ids(
            target_token_to_idx,
            target_special_tokens,
            ord_to_script, 
            word_position,
            target_whitespace
        )

        #####
        # Calculate the mean and standard deviation of the embeddings for each script 
        #####
        source_script_stds = []
        source_script_means = []
        for script in all_source_scripts:
            script_embed_list = [
                source_embeddings[x] for x in source_script_to_ids[script]
            ]
            script_embeddings = torch.stack(script_embed_list, dim=0)
            std_and_mean = torch.std_mean(script_embeddings, dim=0)
            source_script_stds.append(std_and_mean[0])
            source_script_means.append(std_and_mean[1])
        source_script_stds = torch.stack(source_script_stds, dim=0)
        source_script_means = torch.stack(source_script_means, dim=0)

        #####
        # Initialise the target embeddings according to a Normal distribution with 
        # the corresponding mean and standard deviation
        #####
        target_embeddings = initialize_by_category_means(
            all_source_scripts, 
            source_script_means, 
            source_script_stds, 
            target_script_to_ids, 
            target_embeddings
        )
        return target_embeddings

    
    def reinitialize_by_identity(
        self,
        source_token_to_idx: dict[str, int],
        target_token_to_idx: dict[str, int],
        source_special_tokens: list[str],
        target_special_tokens: list[str],
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray
    ) -> np.ndarray:
        """Reinitialize the target embeddings by identity.

        Args:
            source_token_to_idx (dict[str, int]): A source dictionary mapping tokens to indices.
            target_token_to_idx (dict[str, int]): A target dictionary mapping tokens to indices.
            source_special_tokens (list[str]): A list of source special tokens.
            target_special_tokens (list[str]): A list of target special tokens.
            source_embeddings (np.ndarray): A source embedding matrix.
            target_embeddings (np.ndarray): A target embedding matrix.

        Returns:
            np.ndarray: A target embedding matrix.
        """
        # Copy the source embeddings for the identical tokens
        for token, new_index in target_token_to_idx.items():
            if token in source_token_to_idx and token not in target_special_tokens and token not in source_special_tokens:
                source_index = source_token_to_idx[token]
                target_embeddings[new_index] = source_embeddings[source_index]
        return target_embeddings


    def __call__(self) -> AutoModelForCausalLM:
        #####
        # Get the source embeddings
        #####
        source_embeddings = self.source_model.get_input_embeddings().weight.detach().numpy()

        #####
        # Generate random embeddings
        # - See https://github.com/malteos/clp-transfer
        #####
        np.random.seed(self.seed)
        target_embeddings = np.random.normal(
            np.mean(source_embeddings, axis=0), 
            np.std(source_embeddings, axis=0), 
            (
                round_to_nearest_multiple(len(self.target_tokenizer), 8), 
                source_embeddings.shape[1]
            )
        )
        source_embeddings = torch.from_numpy(source_embeddings)
        target_embeddings = torch.from_numpy(target_embeddings)
        logger.info(source_embeddings.shape)
        logger.info(target_embeddings.shape)

        #####
        # Set the embeddings for the special tokens
        # - See https://github.com/cmdowney88/embeddingstructure
        #####
        # Get the special tokens
        source_special_tokens_map = self.source_tokenizer.special_tokens_map
        target_special_tokens_map = self.target_tokenizer.special_tokens_map
        source_special_tokens = [token for token in source_special_tokens_map.values()]
        target_special_tokens = [token for token in target_special_tokens_map.values()]
        
        # Get the token to index mappings
        target_token_to_idx = {t: i for t, i in self.target_tokenizer.get_vocab().items()}
        source_token_to_idx = {t: i for t, i in self.source_tokenizer.get_vocab().items()}

        # Copy the source embeddings for the special tokens
        for special_token_name, target_special_token in target_special_tokens_map.items():
            if special_token_name in source_special_tokens_map:
                source_special_token = source_special_tokens_map[special_token_name]
                source_special_token_idx = source_token_to_idx[source_special_token]
                target_special_token_idx = target_token_to_idx[target_special_token]
                target_embeddings[target_special_token_idx] = \
                    source_embeddings[source_special_token_idx]
        logger.info("Special tokens copied")
        
        #####
        # Initialise by script + position
        #####
        source_model_state = json.loads(self.source_tokenizer.backend_tokenizer.model.__getstate__())
        source_whitespace = 'Ġ' if source_model_state["type"] == "BPE" else "▁"
        target_model_state = json.loads(self.target_tokenizer.backend_tokenizer.model.__getstate__())
        target_whitespace = 'Ġ' if target_model_state["type"] == "BPE" else "▁"
        
        target_embeddings = self.reinitialize_by_script(
            source_token_to_idx,
            target_token_to_idx,
            source_special_tokens,
            target_special_tokens,
            source_embeddings,
            target_embeddings,
            unicode_script_file=self.unicode_script_file,
            word_position=True,
            source_whitespace=source_whitespace,
            target_whitespace=target_whitespace
        )
        logger.info("Script + position initialised")
        
        #####
        # Initialise by identity
        #####
        target_embeddings = self.reinitialize_by_identity(
            source_token_to_idx,
            target_token_to_idx,
            source_special_tokens,
            target_special_tokens,
            source_embeddings,
            target_embeddings,
        )
        logger.info("Identity initialised")
        
        #####
        # Set the target model's embeddings
        #####
        target_model = self.source_model
        target_model.resize_token_embeddings(
            len(self.target_tokenizer), 
            pad_to_multiple_of=8 # See https://github.com/huggingface/transformers/issues/26303
        )
        target_model.get_input_embeddings().weight.data = target_embeddings
        target_model.tie_weights()
        logger.info(target_model.get_input_embeddings().weight.data.shape)

        return target_model
