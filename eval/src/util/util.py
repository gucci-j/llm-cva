import numpy as np
import evaluate
from transformers import AutoTokenizer


def get_label_to_token_ids(
    task_name: str,
    lang_name: str,
    tokenizer: AutoTokenizer,
    prompting_in_target_language: bool = False,
):
    if task_name == "xnli":
        if prompting_in_target_language is False:
            label_to_token_ids = {
                "True": tuple(set(tokenizer.encode("True", add_special_tokens=False) + tokenizer.encode(" True", add_special_tokens=False))),
                "False": tuple(set(tokenizer.encode("False", add_special_tokens=False) + tokenizer.encode(" False", add_special_tokens=False))),
                "Neither": tuple(set(tokenizer.encode("Neither", add_special_tokens=False) + tokenizer.encode(" Neither", add_special_tokens=False)))
            }
        else:
            if lang_name == "japanese":
                label_to_token_ids = {
                    "真": tuple(set(tokenizer.encode("真", add_special_tokens=False) + tokenizer.encode(" 真", add_special_tokens=False) + tokenizer.encode("　真", add_special_tokens=False))),
                    "偽": tuple(set(tokenizer.encode("偽", add_special_tokens=False) + tokenizer.encode(" 偽", add_special_tokens=False) + tokenizer.encode("　偽", add_special_tokens=False))),
                    "どちらでもない": tuple(set(tokenizer.encode("どちらでもない", add_special_tokens=False) + tokenizer.encode(" どちらでもない", add_special_tokens=False) + tokenizer.encode("　どちらでもない", add_special_tokens=False)))
                } 
            elif lang_name == "german":
                label_to_token_ids = {
                    "Wahr": tuple(set(tokenizer.encode("Wahr", add_special_tokens=False) + tokenizer.encode(" Wahr", add_special_tokens=False))),
                    "Falsch": tuple(set(tokenizer.encode("Falsch", add_special_tokens=False) + tokenizer.encode(" Falsch", add_special_tokens=False))),
                    "Weder": tuple(set(tokenizer.encode("Weder", add_special_tokens=False) + tokenizer.encode(" Weder", add_special_tokens=False)))
                }
            elif lang_name == "swahili":
                label_to_token_ids = {
                    "Kweli": tuple(set(tokenizer.encode("Kweli", add_special_tokens=False) + tokenizer.encode(" Kweli", add_special_tokens=False))),
                    "Uongo": tuple(set(tokenizer.encode("Uongo", add_special_tokens=False) + tokenizer.encode(" Uongo", add_special_tokens=False))),
                    "Wala": tuple(set(tokenizer.encode("Wala", add_special_tokens=False) + tokenizer.encode(" Wala", add_special_tokens=False)))
                }
            elif lang_name == "arabic":
                label_to_token_ids = {
                    "صحيح": tuple(set(tokenizer.encode("صحيح", add_special_tokens=False) + tokenizer.encode(" صحيح", add_special_tokens=False))),
                    "خطأ": tuple(set(tokenizer.encode("خطأ", add_special_tokens=False) + tokenizer.encode(" خطأ", add_special_tokens=False))),
                    "لا شيء": tuple(set(tokenizer.encode("لا شيء", add_special_tokens=False) + tokenizer.encode(" لا شيء", add_special_tokens=False)))
                }
            else:
                raise NotImplementedError
        return label_to_token_ids
    
    elif task_name == "xcsqa":
        label_to_token_ids = {
            "A": tuple(set(tokenizer.encode("A", add_special_tokens=False) + tokenizer.encode(" A", add_special_tokens=False))),
            "B": tuple(set(tokenizer.encode("B", add_special_tokens=False) + tokenizer.encode(" B", add_special_tokens=False))),
            "C": tuple(set(tokenizer.encode("C", add_special_tokens=False) + tokenizer.encode(" C", add_special_tokens=False))),
            "D": tuple(set(tokenizer.encode("D", add_special_tokens=False) + tokenizer.encode(" D", add_special_tokens=False))),
            "E": tuple(set(tokenizer.encode("E", add_special_tokens=False) + tokenizer.encode(" E", add_special_tokens=False)))
        }
        return label_to_token_ids

    else:
        raise NotImplementedError


def postprocess_generated_texts(
    text: str,
    task_name: str,
    lang_name: str,
    prompting_in_target_language: bool = False
) -> str:
    """Postprocess the generated text.

    Args:
        text (str): A generated text.
        task_name (str): A task name.
        lang_name (str): A language name.
        prompting_in_target_language (bool): Whether to prompt in the target language. Defaults to False.

    Returns:
        str: A postprocessed generated text.
    """
    # Extract the generated text
    generated_text = text.strip() 

    # Convert the generated text to numerical characters if necessary
    if task_name == "xnli":
        if lang_name == "japanese":
            if prompting_in_target_language:
                if "真" == generated_text:
                    return 0, generated_text
                elif "偽" == generated_text:
                    return 1, generated_text
                elif "どちらでもない" == generated_text:
                    return 2, generated_text
                else:
                    raise NotImplementedError
            else:
                if "True" == generated_text:
                    return 0, generated_text
                elif "False" == generated_text:
                    return 1, generated_text
                elif "Neither" == generated_text:
                    return 2, generated_text
                else:
                    raise NotImplementedError
        else:
            if prompting_in_target_language:
                if lang_name == "german":
                    if "Wahr" == generated_text:
                        return 0, generated_text
                    elif "Falsch" == generated_text:
                        return 1, generated_text
                    elif "Weder" == generated_text:
                        return 2, generated_text
                    else:
                        raise NotImplementedError
                elif lang_name == "swahili":
                    if "Kweli" == generated_text:
                        return 0, generated_text
                    elif "Uongo" == generated_text:
                        return 1, generated_text
                    elif "Wala" == generated_text:
                        return 2, generated_text
                    else:
                        raise NotImplementedError
                elif lang_name == "arabic":
                    if "صحيح" == generated_text:
                        return 0, generated_text
                    elif "خطأ" == generated_text:
                        return 1, generated_text
                    elif "لا شيء" == generated_text:
                        return 2, generated_text
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
            else:
                if "True" == generated_text:
                    return 0, generated_text
                elif "False" == generated_text:
                    return 2, generated_text
                elif "Neither" == generated_text:
                    return 1, generated_text
                else:
                    raise NotImplementedError
                
    elif task_name == "xcsqa":
        if "A" == generated_text:
            return 0, generated_text
        elif "B" == generated_text:
            return 1, generated_text
        elif "C" == generated_text:
            return 2, generated_text
        elif "D" == generated_text:
            return 3, generated_text
        elif "E" == generated_text:
            return 4, generated_text
        else:
            raise NotImplementedError
        
    return generated_text


def generate_alignment_matrix(
    src_text: str,
    tgt_text: str,
) -> list[int]:
    """Generate an alignment matrix."""
    alignment = []
    pos = 0
    for char1 in src_text:
        for index in range(pos, len(tgt_text)):
            if char1 == tgt_text[pos]:
                alignment.append(pos)
                break
            pos += 1
    return alignment


def compute_metrics(
    eval_pred_labels: list[tuple[str, str]],
    task_name: str,
    target_lang: str = "english"
) -> dict[str, float]:
    """Compute metrics.

    Args:
        eval_pred_labels (list[tuple[str, str]]): A list of (generated_text, label) pairs.
        task_name (str): A task name.
        target_lang (str): A target language. Defaults to "english".

    Raises:
        NotImplementedError: If the task name is not supported.

    Returns:
        dict[str, float]: A dictionary of metrics.
    """
    if task_name == "xlsum":
        return compute_rouge(eval_pred_labels, target_lang)
    elif task_name == "xnli":
        return compute_accuracy_f1(eval_pred_labels)
    elif task_name == "xquad":
        return compute_qa_metrics(eval_pred_labels)
    elif task_name == "xcsqa":
        return compute_accuracy_f1(eval_pred_labels)
    else:
        raise NotImplementedError


def compute_qa_metrics(
    eval_pred_labels: list[tuple[dict[str, str], dict[str, dict[str, list]]]],
) -> dict[str, float]:
    """Compute QA metrics.

    Args:
        eval_pred_labels (list[tuple[dict[str, str], dict[str, dict[str, list]]]]): A list of (generated_text, label) pairs.

    Returns:
        dict[str, float]: A dictionary of QA metrics.
    """
    qa_metric = evaluate.load("squad")
    eval_preds = [pred for pred, _ in eval_pred_labels]
    eval_labels = [label for _, label in eval_pred_labels]
    result = qa_metric.compute(
        predictions=eval_preds,
        references=eval_labels
    )
    return result


def compute_accuracy_f1(
    eval_pred_labels: list[tuple[str, str]],
) -> dict[str, float]:
    """Compute accuracy and F1.

    Args:
        eval_pred_labels (list[tuple[str, str]]): A list of (generated_text, label) pairs.

    Returns:
        dict[str, float]: A dictionary of accuracy and F1.
    """
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    eval_preds = [pred for pred, _ in eval_pred_labels]
    eval_labels = [label for _, label in eval_pred_labels]
    result_accuracy = accuracy_metric.compute(
        references=eval_labels,
        predictions=eval_preds,
    )
    result_macro_f1 = f1_metric.compute(
        references=eval_labels,
        predictions=eval_preds,
        average="macro"
    )
    result_micro_f1 = f1_metric.compute(
        references=eval_labels,
        predictions=eval_preds,
        average="micro"
    )
    return result_accuracy | {
            "macro_f1": result_macro_f1["f1"],
            "micro_f1": result_micro_f1["f1"]
        }


def compute_rouge(
    eval_pred_labels: list[tuple[str, str]],
    language_type: str = "english"
) -> dict[str, float]:
    """Compute ROUGE scores.

    Args:
        eval_pred_labels (list[tuple[str, str]]): A list of (generated_text, label) pairs.
        language_type (str): A language type. Defaults to "english".

    Returns:
        dict[str, float]: A dictionary of ROUGE scores.
    
    References:
        - https://github.com/chakki-works/sumeval
    """
    if language_type == "japanese":
        from sumeval.metrics.rouge import RougeCalculator
        rouge = RougeCalculator(lang="ja")
        result = {
            "rouge1": [],
            "rouge2": [],
            "rougeL": []
        }
        for pred, label in eval_pred_labels:
            rouge_1 = rouge.rouge_n(
                summary=pred,
                references=label,
                n=1
            )
            rouge_2 = rouge.rouge_n(
                summary=pred,
                references=label,
                n=2
            )
            rouge_l = rouge.rouge_l(
                summary=pred,
                references=label
            )
            result["rouge1"].append(rouge_1)
            result["rouge2"].append(rouge_2)
            result["rougeL"].append(rouge_l)
        result = {
            "rouge1": np.mean(result["rouge1"]),
            "rouge2": np.mean(result["rouge2"]),
            "rougeL": np.mean(result["rougeL"])
        }
    
    elif language_type == "arabic":
        from rouge import Rouge
        rouge = Rouge()
        result = {
            "rouge1": [],
            "rouge2": [],
            "rougeL": []
        }
        for pred, label in eval_pred_labels:
            try:
                scores = rouge.get_scores(pred, label)[0]
                result["rouge1"].append(scores["rouge-1"]["f"])
                result["rouge2"].append(scores["rouge-2"]["f"])
                result["rougeL"].append(scores["rouge-l"]["f"])
            except ValueError:
                result["rouge1"].append(0.0)
                result["rouge2"].append(0.0)
                result["rougeL"].append(0.0)
        result = {
            "rouge1": np.mean(result["rouge1"]),
            "rouge2": np.mean(result["rouge2"]),
            "rougeL": np.mean(result["rougeL"])
        }

    else:
        rouge = evaluate.load("rouge")
        eval_preds = [pred for pred, _ in eval_pred_labels]
        eval_labels = [label for _, label in eval_pred_labels]
        result = rouge.compute(
            predictions=eval_preds, 
            references=eval_labels, 
            use_stemmer=True,
            use_aggregator=True
        )
    return result
