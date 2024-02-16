import enum
import warnings
import numpy as np
import torch

from transformers import is_torch_available, is_tf_available, Pipeline, TextGenerationPipeline

if is_torch_available():
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

if is_tf_available():
    import tensorflow as tf
    from transformers.models.auto.modeling_auto import TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


class ReturnType(enum.Enum):
    TENSORS = 0
    NEW_TEXT = 1
    FULL_TEXT = 2


class TextGenerationPipelineForMultipleChoice(TextGenerationPipeline):
    def _forward(
        self, 
        model_inputs, 
        label_to_token_ids=None,
        **generate_kwargs
    ):
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        # Allow empty prompts
        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]
        prompt_text = model_inputs.pop("prompt_text")

        # If there is a prefix, we may need to adjust the generation length. Do so without permanently modifying
        # generate_kwargs, as some of the parameterization may come from the initialization of the pipeline.
        prefix_length = generate_kwargs.pop("prefix_length", 0)
        if prefix_length > 0:
            has_max_new_tokens = "max_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].max_new_tokens is not None
            )
            if not has_max_new_tokens:
                generate_kwargs["max_length"] = generate_kwargs.get("max_length") or self.model.config.max_length
                generate_kwargs["max_length"] += prefix_length
            has_min_new_tokens = "min_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].min_new_tokens is not None
            )
            if not has_min_new_tokens and "min_length" in generate_kwargs:
                generate_kwargs["min_length"] += prefix_length

        # BS x SL
        generated_sequence = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
        
        # Calculate the score for each label
        if label_to_token_ids is None:
            raise ValueError("label_to_token_ids must be provided")
        scores = {key: -1 for key in label_to_token_ids.keys()}
        for label, token_ids in label_to_token_ids.items():
            for token_id in token_ids:
                token_score = generated_sequence.scores[0][0][token_id]
                if scores[label] < token_score:
                    scores[label] = token_score

        # Get the best label
        best_label = max(scores.items(), key=lambda x: x[1])[0]
        
        generated_sequence = generated_sequence.sequences
        out_b = generated_sequence.shape[0]
        if self.framework == "pt":
            generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
        elif self.framework == "tf":
            generated_sequence = tf.reshape(generated_sequence, (in_b, out_b // in_b, *generated_sequence.shape[1:]))
        return {"generated_sequence": generated_sequence, "input_ids": input_ids, "prompt_text": prompt_text, "pred": best_label}

    
    def postprocess(
        self, 
        model_outputs, 
        return_type=ReturnType.FULL_TEXT, 
        clean_up_tokenization_spaces=True
    ):
        generated_sequence = model_outputs["generated_sequence"][0]
        input_ids = model_outputs["input_ids"]
        prompt_text = model_outputs["prompt_text"]
        pred = model_outputs["pred"]
        generated_sequence = generated_sequence.numpy().tolist()
        records = []
        for sequence in generated_sequence:
            if return_type == ReturnType.TENSORS:
                record = {"generated_token_ids": sequence, "pred": pred}
            elif return_type in {ReturnType.NEW_TEXT, ReturnType.FULL_TEXT}:
                # Decode text
                text = self.tokenizer.decode(
                    sequence,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )

                # Remove PADDING prompt of the sequence if XLNet or Transfo-XL model is used
                if input_ids is None:
                    prompt_length = 0
                else:
                    prompt_length = len(
                        self.tokenizer.decode(
                            input_ids[0],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                        )
                    )

                all_text = text[prompt_length:]
                if return_type == ReturnType.FULL_TEXT:
                    all_text = prompt_text + all_text
                record = {"generated_text": all_text, "pred": pred}
            else:
                # Decode text
                text = self.tokenizer.decode(
                    sequence,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )

                # Remove PADDING prompt of the sequence if XLNet or Transfo-XL model is used
                if input_ids is None:
                    prompt_length = 0
                else:
                    prompt_length = len(
                        self.tokenizer.decode(
                            input_ids[0],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                        )
                    )
                generated_text = text[prompt_length:]
                record = {"generated_text": generated_text, "pred": pred}
            records.append(record)

        return records
