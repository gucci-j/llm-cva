from datasets import load_dataset


def main(args):
    # If the dataset is gated/private, make sure you have run huggingface-cli login
    dataset = load_dataset(
        "oscar-corpus/OSCAR-2301", 
        args.target_lang, 
        cache_dir=args.cache_dir
    )
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        required=True
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        choices=["ja", "de", "sw", "ar"],
        required=True
    )
    args = parser.parse_args()
    main(args)
