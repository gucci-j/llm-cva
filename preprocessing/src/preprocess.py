import datasets
from transformers import AutoTokenizer
from util import ArabertPreprocessor


def filter_fn(
    example: dict, 
    target_lang: str,
    min_length: int = 5,
    max_length: int = 2048,
    use_filter: bool = False,
):
    texts = []
    if use_filter:
        if example['meta']['quality_warnings'] is not None:
            if "header" in example['meta']['quality_warnings']:
                return {"filtered_text": ""}
            elif "footer" in example['meta']['quality_warnings']:
                return {"filtered_text": ""}
            elif "noisy" in example['meta']['quality_warnings']:
                return {"filtered_text": ""}
    for text, specs in zip(
        example['text'].split('\n'), 
        example['meta']['sentence_identifications'],
    ):
        if specs is not None:
            lang_type = specs.get('label')
            if lang_type == target_lang and min_length <= len(text) <= max_length:
                texts.append(text)
    
    if target_lang == "ar":
        MODEL_NAME='aubmindlab/aragpt2-base'
        arabert_prep = ArabertPreprocessor(model_name=MODEL_NAME)
        texts = ''.join(texts)
        texts = arabert_prep.preprocess(texts)
        return {"filtered_text": texts} # This is fine as we split by '\n'.
    else:
        return {"filtered_text": ''.join(texts)} # This is fine as we split by '\n'.


def group_texts(examples: dict, block_size=128):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()    
    return result


def filter_fn_cc100(
    example: dict,
    min_length: int = 5,
):
    texts = []
    for text in example["text"]:
        if min_length <= len(text.split(' ')):
            texts.append(text)
    return {"filtered_text": [' '.join(texts)]} # This is different from filter_fn as CC-100 is sentence-level.


def main(args):
    if args.target_lang == "sw":
        # Load the CC-100 dataset
        print("Loading the CC-100 dataset...")
        dataset = datasets.load_dataset(
            "text", 
            data_files={"train": args.cc100_data_path},
            cache_dir=args.cache_dir,
            split="train"
        )
        
        # Preprocess the CC-100 dataset
        print("Preprocessing the CC-100 dataset...")
        dataset = dataset.map(
                lambda example: filter_fn_cc100(
                example, 
                args.min_length
            ),
            remove_columns=dataset.column_names,
            batched=True,
            batch_size=100,
        )
        
        # Tokenize the dataset
        print("Tokenizing the dataset...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path, 
            cache_dir=args.tokenizer_cache_dir
        )
        dataset = dataset.map(
            lambda examples: tokenizer(examples["filtered_text"]),
            batched=True,
            num_proc=4,
            remove_columns=["filtered_text"],
        )

        # Group the texts
        print("Grouping the texts...")
        dataset = dataset.map(
            lambda examples: group_texts(examples, args.block_size ),
            batched=True, 
            num_proc=4
        )

        # Save the sharded dataset
        print("Saving the dataset...")
        output_path = args.output_dir
        dataset.save_to_disk(output_path)
        return

    else:
        # Load the dataset
        print("Loading the dataset...")
        dataset = datasets.load_dataset(
            "oscar-corpus/OSCAR-2301", 
            args.target_lang,
            cache_dir=args.cache_dir,
            split="train"
        )

        # Shard the dataset
        if args.num_shards > 1:
            print("Sharding the dataset...")
            dataset = dataset.shard(
                num_shards=args.num_shards, 
                index=args.shard_index
            )

        # Filter the dataset
        print("Filtering the dataset...")
        dataset = dataset.map(
            lambda example: filter_fn(
                example, 
                args.target_lang,
                args.min_length,
                args.max_length,
                args.use_filter,
            ),
            remove_columns=dataset.column_names,
        )
        if args.use_filter:
            print("Removing empty samples...")
            dataset = dataset.filter(
                lambda example: example["filtered_text"] != ""
            )
    
        # Tokenize the dataset
        print("Tokenizing the dataset...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path, 
            cache_dir=args.tokenizer_cache_dir
        )
        dataset = dataset.map(
            lambda examples: tokenizer(examples["filtered_text"]),
            batched=True,
            num_proc=4,
            remove_columns=["filtered_text"],
        )

        # Group the texts
        print("Grouping the texts...")
        dataset = dataset.map(
            lambda examples: group_texts(examples, args.block_size), 
            batched=True, 
            num_proc=4
        )

        # Save the sharded dataset
        print("Saving the dataset...")
        output_path = f"{args.output_dir}/shard_{args.shard_index}"
        dataset.save_to_disk(output_path)
        return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Preprocess the OSCAR or CC-100 corpus."
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        choices=["ja", "de", "sw", "ar"],
        required=True,
        help="The target language."
    )
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        required=True,
        help="The cache directory."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="The output directory."
    )
    parser.add_argument(
        "--tokenizer_name_or_path", 
        type=str, 
        required=True,
        help="The tokenizer name or path."
    )
    parser.add_argument(
        "--tokenizer_cache_dir", 
        type=str, 
        default=None,
        help="The tokenizer cache directory."
    )
    parser.add_argument(
        "--block_size", 
        type=int, 
        default=2048,
        help="The block size."
    )
    parser.add_argument(
        "--min_length", 
        type=int, 
        default=5,
        help="The minimum length."
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=2048,
        help="The maximum length."
    )
    parser.add_argument(
        "--num_shards", 
        type=int, 
        default=1,
        help="The number of shards."
    )
    parser.add_argument(
        "--shard_index", 
        type=int, 
        default=0,
        help="The shard index."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="The random seed."
    )
    parser.add_argument(
        "--cc100_data_path",
        type=str,
        default=None,
        help="The path to the CC-100 dataset."
    )
    parser.add_argument(
        "--use_filter",
        action="store_true",
        help="Whether to use the filter."
    )
    args = parser.parse_args()
    main(args)
