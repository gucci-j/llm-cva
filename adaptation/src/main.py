from pathlib import Path

import datasets
import fasttext
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import (CLPEmbeddingInitializer, 
                  CLPPlusEmbeddingInitializer,
                  FOCUSEmbeddingInitializer,
                  HeuristicsEmbeddingInitializer,
                  RandomEmbeddingInitializer,
                  UnusedEmbeddingTokenizerPruner,
                  CLPPlusEmbeddingInitializerUntied,
                  HeuristicsEmbeddingInitializerUntied)

def main(args):
    if args.initialization_method == "random":
        target_tokenizer = AutoTokenizer.from_pretrained(
            args.target_tokenizer_name_or_path,
            cache_dir=args.cache_dir
        )
        initializer = RandomEmbeddingInitializer(
            source_model=AutoModelForCausalLM.from_pretrained(
                args.source_model_name_or_path,
                cache_dir=args.cache_dir
            ),
            source_tokenizer=AutoTokenizer.from_pretrained(
                args.source_tokenizer_name_or_path,
                cache_dir=args.cache_dir
            ),
            target_tokenizer=target_tokenizer,
            seed=args.seed
        )
        target_model = initializer()
        target_model.save_pretrained(args.output_dir)
        target_tokenizer.save_pretrained(args.output_dir)
        
    elif args.initialization_method == "clp":
        target_tokenizer = AutoTokenizer.from_pretrained(
            args.target_tokenizer_name_or_path,
            cache_dir=args.cache_dir
        )
        initializer = CLPEmbeddingInitializer(
            source_model=AutoModelForCausalLM.from_pretrained(
                args.source_model_name_or_path,
                cache_dir=args.cache_dir
            ),
            helper_model=AutoModelForCausalLM.from_pretrained(
                args.helper_model_name_or_path,
                cache_dir=args.cache_dir
            ),
            source_tokenizer=AutoTokenizer.from_pretrained(
                args.source_tokenizer_name_or_path,
                cache_dir=args.cache_dir
            ),
            target_tokenizer=target_tokenizer,
            copy_special_tokens=args.copy_special_tokens,
            seed=args.seed
        )
        target_model = initializer()
        target_model.save_pretrained(args.output_dir)
        target_tokenizer.save_pretrained(args.output_dir)
    
    elif args.initialization_method == "clp_plus":
        target_tokenizer = AutoTokenizer.from_pretrained(
            args.target_tokenizer_name_or_path,
            cache_dir=args.cache_dir
        )
        if args.untied:
            initializer = CLPPlusEmbeddingInitializerUntied(
                source_model=AutoModelForCausalLM.from_pretrained(
                    args.source_model_name_or_path,
                    cache_dir=args.cache_dir
                ),
                helper_model=AutoModelForCausalLM.from_pretrained(
                    args.helper_model_name_or_path,
                    cache_dir=args.cache_dir
                ),
                source_tokenizer=AutoTokenizer.from_pretrained(
                    args.source_tokenizer_name_or_path,
                    cache_dir=args.cache_dir
                ),
                target_tokenizer=target_tokenizer,
                copy_special_tokens=args.copy_special_tokens,
                seed=args.seed
            )
        else:
            initializer = CLPPlusEmbeddingInitializer(
                source_model=AutoModelForCausalLM.from_pretrained(
                    args.source_model_name_or_path,
                    cache_dir=args.cache_dir
                ),
                helper_model=AutoModelForCausalLM.from_pretrained(
                    args.helper_model_name_or_path,
                    cache_dir=args.cache_dir
                ),
                source_tokenizer=AutoTokenizer.from_pretrained(
                    args.source_tokenizer_name_or_path,
                    cache_dir=args.cache_dir
                ),
                target_tokenizer=target_tokenizer,
                copy_special_tokens=args.copy_special_tokens,
                seed=args.seed
            )
        target_model = initializer()
        target_model.save_pretrained(args.output_dir)
        target_tokenizer.save_pretrained(args.output_dir)
        
    elif args.initialization_method == "heuristics":
        target_tokenizer = AutoTokenizer.from_pretrained(
            args.target_tokenizer_name_or_path,
            cache_dir=args.cache_dir
        )
        if args.untied:
            initializer = HeuristicsEmbeddingInitializerUntied(
                source_model=AutoModelForCausalLM.from_pretrained(
                    args.source_model_name_or_path,
                    cache_dir=args.cache_dir
                ),
                source_tokenizer=AutoTokenizer.from_pretrained(
                    args.source_tokenizer_name_or_path,
                    cache_dir=args.cache_dir
                ),
                target_tokenizer=target_tokenizer,
                unicode_script_file=args.unicode_script_file_path,
                seed=args.seed
            )
        else:
            initializer = HeuristicsEmbeddingInitializer(
                source_model=AutoModelForCausalLM.from_pretrained(
                    args.source_model_name_or_path,
                    cache_dir=args.cache_dir
                ),
                source_tokenizer=AutoTokenizer.from_pretrained(
                    args.source_tokenizer_name_or_path,
                    cache_dir=args.cache_dir
                ),
                target_tokenizer=target_tokenizer,
                unicode_script_file=args.unicode_script_file_path,
                seed=args.seed
            )
        target_model = initializer()
        target_model.save_pretrained(args.output_dir)
        target_tokenizer.save_pretrained(args.output_dir)
    
    elif args.initialization_method == "focus":
        target_tokenizer = AutoTokenizer.from_pretrained(
            args.target_tokenizer_name_or_path,
            cache_dir=args.cache_dir
        )
        initializer = FOCUSEmbeddingInitializer(
            source_model=AutoModelForCausalLM.from_pretrained(
                args.source_model_name_or_path,
                cache_dir=args.cache_dir
            ),
            source_tokenizer=AutoTokenizer.from_pretrained(
                args.source_tokenizer_name_or_path,
                cache_dir=args.cache_dir
            ),
            target_tokenizer=target_tokenizer,
            fasttext_model=fasttext.load_model(args.fasttext_model_path),
            seed=args.seed
        )
        target_model = initializer()
        target_model.save_pretrained(args.output_dir)
        target_tokenizer.save_pretrained(args.output_dir)
    
    elif args.initialization_method == "lapt":
        # This is to prune the unused embeddings from the source model and tokenizer
        # **Note that this only works with non-PEFT models!**

        # load tokenized datasets by the source tokenizer
        dataset = datasets.load_from_disk(args.dataset_path)

        # load the source tokenizer
        source_tokenizer = AutoTokenizer.from_pretrained(
            args.source_tokenizer_name_or_path,
            cache_dir=args.cache_dir
        )

        # load the source model
        source_model = AutoModelForCausalLM.from_pretrained(
            args.source_model_name_or_path,
            cache_dir=args.cache_dir
        )

        # load the pruner
        pruner = UnusedEmbeddingTokenizerPruner(
            source_model=source_model,
            source_tokenizer=source_tokenizer,
            dataset=dataset
        )

        # prune the source model and tokenizer
        target_model, target_tokenizer = pruner()

        # save the pruned model and tokenizer
        target_model.save_pretrained(args.output_dir)
        target_tokenizer.save_pretrained(args.output_dir)

        # tailor the dataset to the pruned tokenizer
        dataset = pruner.tailor_dataset(
            source_tokenizer=source_tokenizer,
            pruned_tokenizer=target_tokenizer,
            dataset=dataset
        )
        dataset.save_to_disk(args.output_data_dir)
    
    else:
        raise NotImplementedError


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initialization_method",
        type=str,
        choices=["random", "clp", "clp_plus", "heuristics", 
                 "lapt", "focus"],
        required=True,
        help="The embedding initialization method to use."
    )
    parser.add_argument(
        "--source_model_name_or_path",
        type=str, 
        required=True,
        help="The source model to initialize the target model with."
    )
    parser.add_argument(
        "--source_tokenizer_name_or_path", 
        type=str, 
        required=True,
        help="The source tokenizer to initialize the target tokenizer with."
    )
    parser.add_argument(
        "--helper_tokenizer_name_or_path",
        type=str,
        required=False,
        default=None,
        help="[expand_after] The helper tokenizer to help initialize a terget tokenizer."
    )
    parser.add_argument(
        "--helper_model_name_or_path", 
        type=str,
        required=False,
        default=None,
        help="[clp, clp_plus] The helper model to help initialize a terget model."
    )
    parser.add_argument(
        "--target_tokenizer_name_or_path", 
        type=str, 
        required=False,
        help="[random, clp, clp_plus, heuristics, focus] The target tokenizer name or path."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The cache directory to save the pretrained models."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="The random seed."
    )
    parser.add_argument(
        "--copy_special_tokens", 
        action="store_true",
        help="[clp, clp_plus] Whether to copy the special tokens' embeddings from the source model to the target model."
    )
    parser.add_argument(
        "--unicode_script_file_path",
        type=str,
        default=None,
        help="[heuristics] The path to the unicode script file."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="The output directory to save the target model and tokenizer."
    )
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        required=False,
        default=None,
        help="[lapt] The path to the dataset."
    )
    parser.add_argument(
        "--output_data_dir", 
        type=str, 
        required=False,
        default=None,
        help="[lapt] The output directory to save the pruned dataset."
    )
    parser.add_argument(
        "--fasttext_model_path",
        type=str,
        default=None,
        help="[focus] The path to the FastText model."
    )
    parser.add_argument(
        "--untied",
        action="store_true",
        help="[clp_plus, heuristics] Whether to apply separate initialization for an LM head. Suitable for LLaMA-style models."
    )
    args = parser.parse_args()
    main(args)
    