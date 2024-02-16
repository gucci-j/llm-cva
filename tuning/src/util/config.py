import argparse
from transformers import HfArgumentParser, TrainingArguments

class CustomArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Tune a language model."
        )
        self.hf_parser = HfArgumentParser(TrainingArguments)

        # Define any custom arguments using argparse
        self.parser.add_argument(
            "--dataset_path",
            type=str,
            required=True,
            help="Path to the tokenized dataset."
        )
        self.parser.add_argument(
            "--tokenizer_name_or_path", 
            type=str, 
            required=True,
            help="Path to the tokenizer."
        )
        self.parser.add_argument(
            "--model_name_or_path", 
            type=str, 
            required=True,
            help="Path to the model."
        )
        self.parser.add_argument(
            "--cache_dir", 
            type=str, 
            default=None,
            help="Path to the cache directory."
        )
        self.parser.add_argument(
            "--model_type", 
            type=str, 
            required=True,
            choices=["gpt2", "bloom", "llama2"],
            help="Model type."
        )
        self.parser.add_argument(
            "--tune_embeddings",
            action="store_true",
            help="Whether to tune the embeddings."
        )
        self.parser.add_argument(
            "--r",
            type=int,
            default=16,
            help="The r parameter for LoRA."
        )
        self.parser.add_argument(
            "--lora_alpha",
            type=int,
            default=32,
            help="The alpha parameter for LoRA."
        )
        self.parser.add_argument(
            "--lora_dropout",
            type=float,
            default=0.05,
            help="The dropout parameter for LoLA."
        )
        self.parser.add_argument(
            "--generate_dev_set",
            action="store_true",
            help="Whether to generate a development set."
        )


    def parse_args(self):
        args, extras = self.parser.parse_known_args()
        training_args = self.hf_parser.parse_args_into_dataclasses(extras)[0]
        return args, training_args
