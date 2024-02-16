import json
from datetime import datetime
from pathlib import Path

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import logging

from peft import AutoPeftModelForCausalLM, PeftConfig, PeftModelForCausalLM

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from util import (ArabertPreprocessor, DatasetLoader,
                  TextGenerationPipelineForMultipleChoice, compute_metrics,
                  generate_alignment_matrix, get_label_to_token_ids,
                  postprocess_generated_texts)


def main(args):
    #####
    # Model loading
    #####
    if args.tokenizer_name_or_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
            cache_dir=args.model_cache_dir
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.model_cache_dir
        )
    if args.is_peft:
        if args.lora_only:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                device_map="auto", 
                load_in_8bit=True,
                cache_dir=args.model_cache_dir
            )
            if args.model_name_or_path == "TigerResearch/tigerbot-7b-base":
                model.config.pretraining_tp = 1
            model.load_adapter(args.adapter_name_or_path)
        else:
            try:
                model = AutoPeftModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    device_map="auto", 
                    load_in_8bit=True,
                )
            except Exception:
                parent_path = Path(args.model_name_or_path).parent.parent
                peft_config = PeftConfig.from_pretrained(args.model_name_or_path) 
                base_model_name = peft_config.base_model_name_or_path.split("/")[-1]
                model = AutoModelForCausalLM.from_pretrained(
                    str(parent_path / base_model_name),
                    device_map="auto", 
                    load_in_8bit=True,
                    cache_dir=args.model_cache_dir
                )
                model = PeftModelForCausalLM.from_pretrained(
                    model,
                    args.model_name_or_path
                )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="auto", 
            load_in_8bit=True,
            cache_dir=args.model_cache_dir
        )
        if args.model_name_or_path == "TigerResearch/tigerbot-7b-base":
            model.config.pretraining_tp = 1
    if args.use_arabert:
        MODEL_NAME='aubmindlab/aragpt2-base'
        preprocessor = ArabertPreprocessor(model_name=MODEL_NAME)
    
    #####
    # Data loading
    #####
    dataset = DatasetLoader(
        args.task_name, 
        args.target_lang, 
        args.data_cache_dir, 
        args.num_shots,
        dataset_path=args.dataset_path,
        max_context_len=args.max_context_len,
        tokenizer=tokenizer,
        prompting_in_target_language=args.prompting_in_target_language,
    )
    
    #####
    # Pipeline loading
    #####
    if args.task_name == "xlsum":
        max_new_tokens = 128
    elif args.task_name == "xnli":
        max_new_tokens = 4
    elif args.task_name == "xquad":
        max_new_tokens = 10
    elif args.task_name == "xcsqa":
        max_new_tokens = 4
    else:
        raise ValueError(f"Invalid task name: {args.task_name}")
    
    if args.task_name in ("xnli", "xcsqa"):
        label_to_token_ids = get_label_to_token_ids(
            args.task_name,
            args.target_lang,
            tokenizer,
            prompting_in_target_language=args.prompting_in_target_language,
        )
        pipe = pipeline(
            "text-generation",
            pipeline_class=TextGenerationPipelineForMultipleChoice,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens, 
            return_dict_in_generate=True,
            output_scores=True,
            label_to_token_ids=label_to_token_ids
        )
    else:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens, 
            temperature=args.temperature, 
            repetition_penalty=args.repetition_penalty, 
            top_k=args.top_k, 
            top_p=args.top_p,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            do_sample=args.do_sample,
            early_stopping=args.early_stopping,
        )

    #####
    # Evaluation
    #####
    eval_pred_labels = []
    if args.num_shots == 0:
        test_dataset = dataset.get_test_dataset()
    else:
        test_dataset = dataset.get_fewshot_test_dataset()
    for sample in test_dataset.shuffle(seed=args.seed).select(range(args.num_max_samples)):
        # Generate a prediction
        text = sample["prompt"]
        if args.use_arabert:
            text = preprocessor.preprocess(text)

        if args.task_name in ("xnli", "xcsqa"):
            result = pipe(text, return_full_text=False).pop()
            generated_text = result["generated_text"]
            pred_label = result["pred"]
            if args.task_name == "xnli":
                pred_label, _ = postprocess_generated_texts(
                    pred_label, 
                    args.task_name, 
                    args.target_lang, 
                    args.prompting_in_target_language
                )
                eval_pred_labels.append((pred_label, sample["label"]))
            elif args.task_name == "xcsqa":
                pred_label, _ = postprocess_generated_texts(
                    pred_label,
                    args.task_name,
                    args.target_lang
                )
                eval_pred_labels.append((pred_label, sample["label"]))
        else:
            generated_text = pipe(text, return_full_text=False).pop()["generated_text"]
            if args.task_name == "xlsum":
                if args.target_lang == "german":
                    eval_pred_labels.append((generated_text.strip(), sample["target"]))
                else:
                    eval_pred_labels.append((generated_text.strip(), sample["summary"]))
            elif args.task_name == "xquad":
                if args.target_lang == "japanese":
                    context = " ".join(sample["context"].replace(" ", "").rstrip("。"))
                    alignment_list = generate_alignment_matrix(sample["context"], context)
                    answers = {
                        "text": [
                            " ".join(text.replace(" ", "").rstrip("。")) for text in sample["answers"]["text"]
                        ],
                        "answer_start": [
                            alignment_list[index] for index in sample["answers"]["answer_start"]
                        ],
                    }
                    eval_pred_labels.append(
                        ({"prediction_text": " ".join(generated_text.replace(" ", "").rstrip("。")), "id": sample["id"]}, 
                        {"answers": answers, "id": sample["id"]})
                    )
                else:
                    eval_pred_labels.append(
                        ({"prediction_text": generated_text.strip(), "id": sample["id"]}, 
                        {"answers": sample["answers"], "id": sample["id"]})
                    )
                
        # logging
        logger.info(f"Prompt: {text}")
        logger.info(f"Generated text: {generated_text}")
        if args.task_name in ("xnli", "xcsqa"):
            logger.info(f"Prediction: {pred_label}")
        if args.task_name == "xlsum":
            if args.target_lang == "german":
                logger.info(f"Reference: {sample['target']}")
            else:
                logger.info(f"Reference: {sample['summary']}")
        elif args.task_name == "xnli" or args.task_name == "xcsqa":
            logger.info(f"Reference: {sample['label']}")
        elif args.task_name == "xquad":
            logger.info(f"Reference: {sample['answers']['text']}")
        logger.info(f"-" * 100)

    # Compute metrics
    results = compute_metrics(eval_pred_labels, args.task_name, args.target_lang)

    #####
    # Save results in .json
    #####
    results["task_name"] = args.task_name
    results["target_lang"] = args.target_lang
    results["num_max_samples"] = args.num_max_samples
    results["num_shots"] = args.num_shots
    results["seed"] = args.seed
    if args.is_peft and args.lora_only:
        results["model_name_or_path"] = args.adapter_name_or_path
    else:
        results["model_name_or_path"] = args.model_name_or_path
    results["tokenizer_name_or_path"] = args.tokenizer_name_or_path
    results["plot_model_name"] = args.plot_model_name
    results["plot_category"] = args.plot_category
    results["prompt_lang"] = "target" if args.prompting_in_target_language else "source"
    results["generation_config"] = {
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "do_sample": args.do_sample,
        "early_stopping": args.early_stopping,
    }
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = f"{args.results_dir}/{args.task_name}_{args.target_lang}_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        required=True,
        help="The model name or path",
    )
    parser.add_argument(
        "--tokenizer_name_or_path", 
        type=str,
        help="The tokenizer name or path. If None, the model_name_or_path is used as tokenizer_name_or_path",
    )
    parser.add_argument(
        "--model_cache_dir", 
        type=str, 
        default=None,
        help="The directory where the model is cached",
    )
    parser.add_argument(
        "--data_cache_dir", 
        type=str, 
        default=None,
        help="The directory where the dataset is cached",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="[xcsqa, kenswquad] The path to the dataset",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        choices=["xlsum", "xnli", "xquad", "xcsqa"],
        help="The task to evaluate",
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        required=True,
        choices=["japanese", "english", "german", "swahili", "arabic"],
        help="The target language to evaluate",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="logs",
        help="The directory where the results are saved",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for evaluation",
    )
    parser.add_argument(
        "--num_max_samples",
        type=int,
        default=500,
        help="The maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--num_shots",
        type=int,
        default=0,
        help="The number of shots to evaluate",
    )
    parser.add_argument(
        "--is_peft",
        action="store_true",
        help="Whether to use PEFT model or not",
    )
    parser.add_argument(
        "--lora_only",
        action="store_true",
        help="Whether to use LoRA only model or not",
    )
    parser.add_argument(
        "--adapter_name_or_path",
        type=str,
        default=None,
        help="The adapter name or path",
    )
    parser.add_argument(
        "--plot_model_name",
        type=str,
        default=None,
        help="The model name for plotting",
    )
    parser.add_argument(
        "--plot_category",
        type=str,
        default=None,
        help="The category for plotting",
    )
    parser.add_argument(
        "--max_context_len",
        type=int,
        default=None,
        help="The maximum length of the context",
    )
    parser.add_argument(
        "--prompting_in_target_language",
        action="store_true",
        help="Whether to use prompting in target language or not",
    )
    parser.add_argument(
        "--use_arabert",
        action="store_true",
        help="Whether to use AraBERT or not",
    )

    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="The value used to module the next token probabilities",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="The parameter for repetition penalty",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="The number of highest probability vocabulary tokens to keep for top-k-filtering",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="The number of beams for beam search",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="The number of independently computed returned sequences for each element in the batch",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling or not",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Whether to stop the beam search when at least num_beams sentences are finished per batch or not",
    )

    # Parse
    args = parser.parse_args()
    
    main(args)
