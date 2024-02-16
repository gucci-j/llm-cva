import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import datasets
import torch
from peft import (LoraConfig, TaskType, PeftModel, get_peft_model,
                  get_peft_model_state_dict, prepare_model_for_kbit_training)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from util import CustomArgumentParser

def main(args, training_args):
    # Load the dataset
    dataset = datasets.load_from_disk(args.dataset_path)

    # Generate a development set
    if args.generate_dev_set:
        train_dataset, eval_dataset = dataset.train_test_split(test_size=0.05).values()
    else:
        train_dataset = dataset
        eval_dataset = None

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        cache_dir=args.cache_dir
    )

    # Set up the data collator
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto", 
        load_in_8bit=True,
        cache_dir=args.cache_dir
    )
    model = prepare_model_for_kbit_training(model)
    logger.info(f'Before PEFT applied (Memory): {model.get_memory_footprint()}')

    # Set up LoRA
    if args.model_type == "bloom":
        target_modules = ["query_key_value", "dense", 
                          "dense_h_to_4h", "dense_4h_to_h"]
    elif args.model_type == "llama2":
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", 
                          "gate_proj", "down_proj", "up_proj"]
    elif args.model_type == "gpt2":
        target_modules = ["c_proj", "c_attn", "c_fc"]
    else:
        raise ValueError(f"Model type {args.model_type} not supported.")
    if args.tune_embeddings:
        if args.model_type == "bloom":
            modules_to_save = ["lm_head", "word_embeddings"]
        elif args.model_type == "llama2":
            modules_to_save = ["lm_head", "embed_tokens"]
        elif args.model_type == "gpt2":
            modules_to_save = ["lm_head", "wte"]
        else:
            raise ValueError(f"Model type {args.model_type} not supported.")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            target_modules=target_modules,
            inference_mode=False, 
            r=args.r,
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout, 
            modules_to_save=modules_to_save
        )
        model = get_peft_model(model, peft_config)

    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=args.r,
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout, 
        )
        model = get_peft_model(model, peft_config)
    logger.info(f'After PEFT applied (Memory): {model.get_memory_footprint()}')
    logger.info(model)

    # Set up the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    logger.info(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


if __name__ == "__main__":
    parser = CustomArgumentParser()
    args, training_args = parser.parse_args()
    logger.info(args)
    logger.info(training_args)

    main(args, training_args)
