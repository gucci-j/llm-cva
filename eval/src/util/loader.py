from pathlib import Path
import datasets
import pandas as pd
from transformers import AutoTokenizer


jnli_label_to_prompt_label = {
    0: "True",
    1: "False",
    2: "Neither"
}

jnli_label_to_prompt_label_in_target_lang = {
    0: "真",
    1: "偽",
    2: "どちらでもない"
}

xnli_label_to_prompt_label = {
    0: "True",
    1: "Neither",
    2: "False"
}

xnli_label_to_prompt_label_in_german = {
    0: "Wahr",
    1: "Weder",
    2: "Falsch"
}

xnli_label_to_prompt_label_in_swahili = {
    0: "Kweli",
    1: "Wala",
    2: "Uongo"
}

xnli_label_to_prompt_label_in_arabic = {
    0: "صحيح",
    1: "لا هذا ولا ذاك",
    2: "خطأ"
}

xcsqa_prompt_label_to_label = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4
}

langname_to_langcode = {
    "english": "en",
    "japanese": "ja",
    "german": "de",
    "swahili": "sw",
    "arabic": "ar",
}

langname_to_langcode_xcsqa = {
    "english": "en",
    "japanese": "jap",
    "german": "de",
    "swahili": "sw",
    "arabic": "ar"
}

index_to_question_index = {
    0: "01",
    1: "02",
    2: "03",
    3: "04",
    4: "05",
    5: "06",
    6: "07",
    7: "08",
    8: "09",
    9: "10",
    10: "11",
    11: "12",
    12: "13"
}

class DatasetLoader:
    def __init__(
        self,
        task_name: str,
        target_lang: str = None,
        cache_dir: str = None,
        num_shots: int = 1,
        seed: int = 42,
        dataset_path: str = None,
        max_context_len: int = None,
        tokenizer: AutoTokenizer = None,
        prompting_in_target_language: bool = False
    ):
        #####
        # Load dataset
        #####
        if task_name == "xlsum":
            if target_lang == "german":
                task_name = 'GEM/mlsum'
                subset_name = langname_to_langcode[target_lang]
            else:
                task_name = "csebuetnlp/xlsum"
                subset_name = target_lang
        elif task_name == "xnli":
            if target_lang == "japanese":
                task_name = "shunk031/JGLUE"
                subset_name = "JNLI"
            else:
                task_name = "xnli"
                subset_name = langname_to_langcode[target_lang]
        elif task_name == "xquad":
            if target_lang == "japanese":
                task_name = "shunk031/JGLUE"
                subset_name = "JSQuAD"
            elif target_lang == "swahili":
                task_name = "kenswquad"
                subset_name = None
            else:
                task_name = "xquad"
                subset_name = "xquad." + langname_to_langcode[target_lang]
        elif task_name == "xcsqa":
            subset_name = langname_to_langcode_xcsqa[target_lang]
        else:
            raise NotImplementedError
        
        if task_name == "xcsqa":
            train_dataset, test_dataset = self.load_local_dataset(
                dataset_path, subset_name, "dev", task_name
            ).train_test_split(test_size=0.5, seed=seed).values()
            self.dataset = datasets.DatasetDict({
                "train": train_dataset,
                "test": test_dataset
            })
        elif task_name == "kenswquad":
            train_dataset, test_dataset = self.load_local_dataset(
                dataset_path, None, None, task_name, max_context_len, num_shots, tokenizer
            ).train_test_split(test_size=0.4, seed=seed).values()
            self.dataset = datasets.DatasetDict({
                "train": train_dataset,
                "test": test_dataset
            })
        elif task_name == "xquad":
            dataset = datasets.load_dataset(task_name, subset_name, cache_dir=cache_dir)
            train_dataset, test_dataset = dataset["validation"].train_test_split(test_size=0.5, seed=seed).values()
            self.dataset = datasets.DatasetDict({
                "train": train_dataset,
                "test": test_dataset
            })
        else:
            if subset_name is None:
                self.dataset = datasets.load_dataset(task_name, cache_dir=cache_dir)
            else:
                self.dataset = datasets.load_dataset(task_name, subset_name, cache_dir=cache_dir)
        
        #####
        # Set configs
        #####
        self.task_name = task_name
        self.target_lang = target_lang
        self.subset_name = subset_name
        self.num_shots = num_shots
        self.prompting_in_target_language = prompting_in_target_language

        #####
        # Generate prompts
        #####
        if num_shots == 0:
            self.dataset = self.dataset.map(
                lambda sample: self.generate_zeroshot_prompt(sample), 
                batched=False,
            )
        elif num_shots > 0:
            test_dataset = self.get_test_dataset().shuffle(seed=seed)
            ret_dataset = []
            for test_sample_index, test_sample in enumerate(test_dataset):
                ret_sample = self.generate_fewshot_prompt(
                    test_sample, 
                    self.dataset['train'].shuffle(seed=seed).select(
                        range(test_sample_index * num_shots, 
                              (test_sample_index * num_shots) + num_shots) 
                        if test_sample_index * num_shots < len(self.dataset['train']) - num_shots
                        else range(test_sample_index, len(self.dataset['train']))[:num_shots]
                    )
                )
                ret_dataset.append(test_sample | ret_sample)
            self.test_dataset = datasets.Dataset.from_pandas(pd.DataFrame(ret_dataset))
        else:
            raise ValueError("num_shots must be a zero or positive integer.")
        
        #####
        # Check max context length
        #####
        if max_context_len is not None and tokenizer is not None:
            if num_shots == 0:
                self.dataset = self.dataset.map(
                    lambda sample: self.check_max_context_len(
                        sample, tokenizer, max_context_len, truncate=True
                    ), 
                    batched=False,
                )
            else:
                self.test_dataset = self.test_dataset.map(
                    lambda sample: self.check_max_context_len(
                        sample, tokenizer, max_context_len, truncate=False
                    ), 
                    batched=False,
                )

    
    def check_max_context_len(
        self, 
        sample: dict, 
        tokenizer: AutoTokenizer, 
        max_context_len: int,
        truncate: bool
    ) -> dict:
        """Check the maximum context length."""
        try:
            prompt_len = len(tokenizer(sample['prompt'])['input_ids'])
            #print(prompt_len, sample['prompt'])
            if prompt_len > max_context_len:
                raise ValueError("The context length is longer than the maximum context length.")
        except ValueError:
            if truncate and (self.task_name == "csebuetnlp/xlsum" or self.task_name == "GEM/mlsum"):
                article_len = len(tokenizer(sample['text'])['input_ids']) - (prompt_len - max_context_len)
                sample["prompt"] = self.generate_zeroshot_prompt({
                    "text": tokenizer.decode(tokenizer(sample['text'])['input_ids'][:article_len])
                })["prompt"]
            else:
                raise ValueError("The context length is longer than the maximum context length.")
        return sample
    

    def load_local_dataset(
        self, 
        dataset_path: str,
        lang_code: str,
        subset_name: str,
        task_name: str,
        max_context_len: int = None,
        num_shots: int = None,
        tokenizer: AutoTokenizer = None
    ) -> datasets.Dataset:
        """Load a local dataset.

        Args:
            dataset_path (str): A path to the dataset.
            lang_code (str): A language code.
            subset_name (str): A subset name.
            task_name (str): A task name.
            max_context_len (int): A maximum context length.
            tokenizer (AutoTokenizer): A tokenizer.

        Raises:
            NotImplementedError: If the task name is not supported.

        Returns:
            datasets.Dataset: A dataset.
        """
        if task_name == "xcsqa":
            df = pd.read_json(dataset_path + "/{}/{}.jsonl".format(lang_code, subset_name), lines=True)
            rows = [row[1].to_dict() for row in df.iterrows()]
            rows = [
                {
                    "id": row["id"],
                    "lang": row["lang"],
                    "question": row["question"]["stem"],
                    "prompt_label": row["answerKey"],
                    "label": xcsqa_prompt_label_to_label[row["answerKey"]],
                    "choice_1": row["question"]["choices"][0]["text"],
                    "choice_2": row["question"]["choices"][1]["text"],
                    "choice_3": row["question"]["choices"][2]["text"],
                    "choice_4": row["question"]["choices"][3]["text"],
                    "choice_5": row["question"]["choices"][4]["text"],
                } for row in rows
            ]

        elif task_name == "kenswquad":
            df = pd.read_csv(dataset_path)
            rows = [row[1].to_dict() for row in df.iterrows()]

            # Filter out samples with no corresponding text file
            new_rows = []
            for row in rows:
                sample_path = Path(dataset_path).parent / "qatexts" / str(row["Story_ID"] + ".txt")
                try:
                    context = sample_path.read_text().strip().replace("\n", "")
                    new_rows.append(row | {"context": context})
                except Exception:
                    pass
            
            # Convert to SQuAD format
            rows = []
            for row_id, row in enumerate(new_rows):
                for index in range(row["Num_QA_pairs"]):
                    question = row["Q" + index_to_question_index[index]]
                    answer_text = row["A" + index_to_question_index[index]]
                    
                    context_text_lower = row["context"].lower()
                    answer_text_lower = answer_text.lower()
                    start_position = context_text_lower.find(answer_text_lower)

                    if start_position != -1:
                        if max_context_len is not None:
                            if len(tokenizer(row["context"])['input_ids']) <= max_context_len // (num_shots + 2): # TODO: Need to adjust this param.
                                rows.append({
                                    "id": str(row_id),
                                    "context": row["context"],
                                    "question": question,
                                    "answers": {
                                        "text": [answer_text],
                                        "answer_start": [start_position]
                                    }
                                })
                        else:
                            rows.append({
                                    "id": str(row_id),
                                    "context": row["context"],
                                    "question": question,
                                    "answers": {
                                        "text": [answer_text],
                                        "answer_start": [start_position]
                                    }
                                })
        else:
            raise NotImplementedError
        
        return datasets.Dataset.from_pandas(pd.DataFrame(rows))
    

    def generate_zeroshot_prompt(self, sample):
        """Generate a zero-shot prompt.

        Args:
            sample (dict): A sample from the dataset.

        Raises:
            NotImplementedError: If the task name is not supported.

        Returns:
            dict: A dictionary of the prompt.
        """
        if self.task_name == "csebuetnlp/xlsum" or self.task_name == "GEM/mlsum":
            if self.prompting_in_target_language:
                if self.target_lang == "japanese":
                    prompt = f"次の文章の要約を日本語で書きなさい。記事: {sample['text']} 要約:"
                elif self.target_lang == "german":
                    prompt = f"Schreiben Sie eine kurze Zusammenfassung des folgenden Textes auf Deutsch. Artikel: {sample['text']} Zusammenfassung:"
                elif self.target_lang == "swahili":
                    prompt = f"Andika muhtasari mfupi wa maandishi yafuatayo kwa Kiswahili. Makala: {sample['text']} Muhtasari:"
                elif self.target_lang == "arabic":
                    prompt = f"اكتب ملخصًا قصيرًا للنص التالي باللغة العربية. المقالة: {sample['text']} الملخص:"
                else:
                    raise NotImplementedError
            else:
                prompt = f"Write a short summary of the following text in {self.target_lang.capitalize()}. Article: {sample['text']} Summary:"

        elif self.task_name == "shunk031/JGLUE" and self.subset_name == "JNLI":
            if self.prompting_in_target_language:
                prompt = f"{sample['sentence1']} 質問: {sample['sentence2']} 真、偽、どちらでもない？ 答え:"
            else:
                prompt = f"{sample['sentence1']} Question: {sample['sentence2']} True, False, or Neither? Answer:"

        elif self.task_name == "xnli":
            if self.prompting_in_target_language:
                if self.target_lang == "german":
                    prompt = f"{sample['premise']} Frage: {sample['hypothesis']} Wahr, Falsch oder Weder? Antwort:"
                elif self.target_lang == "swahili":
                    prompt = f"{sample['premise']} Swali: {sample['hypothesis']} Kweli, Uongo au Wala? Jibu:"
                elif self.target_lang == "arabic":
                    prompt = f"{sample['premise']} سؤال: {sample['hypothesis']} صحيح ، خطأ أو لا هذا ولا ذاك؟ إجابة:"
                else: 
                    raise NotImplementedError
            else:
                prompt = f"{sample['premise']} Question: {sample['hypothesis']} True, False, or Neither? Answer:"

        elif self.task_name == "xquad" or (self.task_name == "shunk031/JGLUE" and self.subset_name == "JSQuAD") or self.task_name == "kenswquad":
            if self.prompting_in_target_language:
                if self.target_lang == "japanese":
                    prompt = f"次の文章の質問に答えなさい。文章: {sample['context']} 質問: {sample['question']} 答え:"
                elif self.target_lang == "german":
                    prompt = f"Beantworten Sie die folgende Frage. Artikel: {sample['context']} Frage: {sample['question']} Antwort:"
                elif self.target_lang == "swahili":
                    prompt = f"Jibu swali lifuatalo. Makala: {sample['context']} Swali: {sample['question']} Jibu:"
                elif self.target_lang == "arabic":
                    prompt = f"أجب على السؤال التالي. سياق: {sample['context']} السؤال: {sample['question']} الإجابة:"
                else:
                    raise NotImplementedError
            else:
                prompt = f"Answer the following question. Context: {sample['context']} Question: {sample['question']} Answer:"

        elif self.task_name == "xcsqa":
            if self.prompting_in_target_language:
                if self.target_lang == "japanese":
                    prompt = f"{sample['question']} A. {sample['choice_1']}, B. {sample['choice_2']}, C. {sample['choice_3']}, D. {sample['choice_4']}, E. {sample['choice_5']} 答え:"
                elif self.target_lang == "german":
                    prompt = f"{sample['question']} A. {sample['choice_1']}, B. {sample['choice_2']}, C. {sample['choice_3']}, D. {sample['choice_4']}, E. {sample['choice_5']} Antwort:"
                elif self.target_lang == "swahili":
                    prompt = f"{sample['question']} A. {sample['choice_1']}, B. {sample['choice_2']}, C. {sample['choice_3']}, D. {sample['choice_4']}, E. {sample['choice_5']} Jibu:"
                elif self.target_lang == "arabic":
                    prompt = f"{sample['question']} A. {sample['choice_1']}, B. {sample['choice_2']}, C. {sample['choice_3']}, D. {sample['choice_4']}, E. {sample['choice_5']} إجابة:"
                else:
                    raise NotImplementedError
            else:
                prompt = f"{sample['question']} A. {sample['choice_1']}, B. {sample['choice_2']}, C. {sample['choice_3']}, D. {sample['choice_4']}, E. {sample['choice_5']} Answer:"
        
        else:
            raise NotImplementedError
        
        return {"prompt": prompt}
    

    def generate_fewshot_prompt(
        self, sample, demonstration_samples
    ) -> dict:
        if self.task_name == "csebuetnlp/xlsum":
            prompt = ""
            if self.prompting_in_target_language:
                if self.target_lang == "japanese":
                    for text, summary in zip(demonstration_samples['text'], demonstration_samples['summary']):
                        prompt += f"次の文章の要約を日本語で書きなさい。記事: {text} 要約: {summary}\n"
                    prompt += f"次の文章の要約を日本語で書きなさい。記事: {sample['text']} 要約:"
                elif self.target_lang == "swahili":
                    for text, summary in zip(demonstration_samples['text'], demonstration_samples['summary']):
                        prompt += f"Andika muhtasari mfupi wa maandishi yafuatayo kwa Kiswahili. Makala: {text} Muhtasari: {summary}\n"
                    prompt += f"Andika muhtasari mfupi wa maandishi yafuatayo kwa Kiswahili. Makala: {sample['text']} Muhtasari:"
                elif self.target_lang == "arabic":
                    for text, summary in zip(demonstration_samples['text'], demonstration_samples['summary']):
                        prompt += f"اكتب ملخصًا قصيرًا للنص التالي باللغة العربية. المقالة: {text} الملخص: {summary}\n"
                    prompt += f"اكتب ملخصًا قصيرًا للنص التالي باللغة العربية. المقالة: {sample['text']} الملخص:"
                else:
                    raise NotImplementedError
            else:
                for text, summary in zip(demonstration_samples['text'], demonstration_samples['summary']):
                    prompt += f"Write a short summary of the following text in {self.target_lang.capitalize()}. Article: {text} Summary: {summary}\n"
                prompt += f"Write a short summary of the following text in {self.target_lang.capitalize()}. Article: {sample['text']} Summary:"

        elif self.task_name == "GEM/mlsum":
            prompt = ""
            if self.prompting_in_target_language:
                for text, summary in zip(demonstration_samples['text'], demonstration_samples['target']):
                    prompt += f"Schreiben Sie eine kurze Zusammenfassung des folgenden Textes auf Deutsch. Artikel: {text} Zusammenfassung: {summary}\n"
                prompt += f"Schreiben Sie eine kurze Zusammenfassung des folgenden Textes auf Deutsch. Artikel: {sample['text']} Zusammenfassung:"
            else:
                for text, summary in zip(demonstration_samples['text'], demonstration_samples['target']):
                    prompt += f"Write a short summary of the following text in {self.target_lang.capitalize()}. Article: {text} Summary: {summary}\n"
                prompt += f"Write a short summary of the following text in {self.target_lang.capitalize()}. Article: {sample['text']} Summary:"

        elif self.task_name == "shunk031/JGLUE" and self.subset_name == "JNLI":
            prompt = ""
            if self.prompting_in_target_language:
                for sentence1, sentence2, label in zip(
                    demonstration_samples['sentence1'], 
                    demonstration_samples['sentence2'], 
                    demonstration_samples['label']
                ):
                    label = jnli_label_to_prompt_label_in_target_lang[label]
                    prompt += f"{sentence1} 質問: {sentence2} 真、偽、どちらでもない？ 答え: {label}\n"
                prompt += f"{sample['sentence1']} 質問: {sample['sentence2']} 真、偽、どちらでもない？ 答え:"
            else:
                for sentence1, sentence2, label in zip(
                    demonstration_samples['sentence1'], 
                    demonstration_samples['sentence2'], 
                    demonstration_samples['label']
                ):
                    label = jnli_label_to_prompt_label[label]
                    prompt += f"{sentence1} Question: {sentence2} True, False, or Neither? Answer: {label}\n"
                prompt += f"{sample['sentence1']} Question: {sample['sentence2']} True, False, or Neither? Answer:"
            
        elif self.task_name == "xnli":
            prompt = ""
            if self.prompting_in_target_language:
                if self.target_lang == "german":
                    for premise, hypothesis, label in zip(
                        demonstration_samples['premise'], 
                        demonstration_samples['hypothesis'], 
                        demonstration_samples['label']
                    ):
                        label = xnli_label_to_prompt_label_in_german[label]
                        prompt += f"{premise} Frage: {hypothesis} Wahr, Falsch oder Weder? Antwort: {label}\n"
                    prompt += f"{sample['premise']} Frage: {sample['hypothesis']} Wahr, Falsch oder Weder? Antwort:"
                elif self.target_lang == "swahili":
                    for premise, hypothesis, label in zip(
                        demonstration_samples['premise'], 
                        demonstration_samples['hypothesis'], 
                        demonstration_samples['label']
                    ):
                        label = xnli_label_to_prompt_label_in_swahili[label]
                        prompt += f"{premise} Swali: {hypothesis} Kweli, Uongo au Wala? Jibu: {label}\n"
                    prompt += f"{sample['premise']} Swali: {sample['hypothesis']} Kweli, Uongo au Wala? Jibu:"
                elif self.target_lang == "arabic":
                    for premise, hypothesis, label in zip(
                        demonstration_samples['premise'], 
                        demonstration_samples['hypothesis'], 
                        demonstration_samples['label']
                    ):
                        label = xnli_label_to_prompt_label_in_arabic[label]
                        prompt += f"{premise} سؤال: {hypothesis} صحيح ، خطأ أو لا هذا ولا ذاك؟ إجابة: {label}\n"
                    prompt += f"{sample['premise']} سؤال: {sample['hypothesis']} صحيح ، خطأ أو لا هذا ولا ذاك؟ إجابة:"
                else:
                    raise NotImplementedError
            else:
                for premise, hypothesis, label in zip(
                    demonstration_samples['premise'], 
                    demonstration_samples['hypothesis'], 
                    demonstration_samples['label']
                ):
                    label = xnli_label_to_prompt_label[label]
                    prompt += f"{premise} Question: {hypothesis} True, False, or Neither? Answer: {label}\n"
                prompt += f"{sample['premise']} Question: {sample['hypothesis']} True, False, or Neither? Answer:"
            
        elif self.task_name == "xquad" or (self.task_name == "shunk031/JGLUE" and self.subset_name == "JSQuAD") or self.task_name == "kenswquad":
            prompt = ""
            if self.prompting_in_target_language:
                if self.target_lang == "japanese":
                    for context, question, answer in zip(
                        demonstration_samples['context'], 
                        demonstration_samples['question'], 
                        demonstration_samples['answers']
                    ):
                        prompt += f"次の文章の質問に答えなさい。文章: {context} 質問: {question} 答え: {answer['text'][0]}\n"
                    prompt += f"次の文章の質問に答えなさい。文章: {sample['context']} 質問: {sample['question']} 答え:"
                elif self.target_lang == "german":
                    for context, question, answer in zip(
                        demonstration_samples['context'], 
                        demonstration_samples['question'], 
                        demonstration_samples['answers']
                    ):
                        prompt += f"Beantworten Sie die folgende Frage. Artikel: {context} Frage: {question} Antwort: {answer['text'][0]}\n"
                    prompt += f"Beantworten Sie die folgende Frage. Artikel: {sample['context']} Frage: {sample['question']} Antwort:"
                elif self.target_lang == "swahili":
                    for context, question, answer in zip(
                        demonstration_samples['context'], 
                        demonstration_samples['question'], 
                        demonstration_samples['answers']
                    ):
                        prompt += f"Jibu swali lifuatalo. Makala: {context} Swali: {question} Jibu: {answer['text'][0]}\n"
                    prompt += f"Jibu swali lifuatalo. Makala: {sample['context']} Swali: {sample['question']} Jibu:"
                elif self.target_lang == "arabic":
                    for context, question, answer in zip(
                        demonstration_samples['context'], 
                        demonstration_samples['question'], 
                        demonstration_samples['answers']
                    ):
                        prompt += f"أجب على السؤال التالي. سياق: {context} السؤال: {question} الإجابة: {answer['text'][0]}\n"
                    prompt += f"أجب على السؤال التالي. سياق: {sample['context']} السؤال: {sample['question']} الإجابة:"
                else:
                    raise NotImplementedError
            else:
                for context, question, answer in zip(
                    demonstration_samples['context'], 
                    demonstration_samples['question'], 
                    demonstration_samples['answers']
                ):
                    prompt += f"Answer the following question. Context: {context} Question: {question} Answer: {answer['text'][0]}\n"
                prompt += f"Answer the following question. Context: {sample['context']} Question: {sample['question']} Answer:"
            
        elif self.task_name == "xcsqa":
            prompt = ""
            if self.prompting_in_target_language:
                if self.target_lang == "japanese":
                    for question, choice_1, choice_2, choice_3, choice_4, choice_5, label in zip(
                        demonstration_samples['question'], 
                        demonstration_samples['choice_1'], 
                        demonstration_samples['choice_2'], 
                        demonstration_samples['choice_3'], 
                        demonstration_samples['choice_4'], 
                        demonstration_samples['choice_5'], 
                        demonstration_samples['prompt_label']
                    ):
                        prompt += f"{question} A. {choice_1}, B. {choice_2}, C. {choice_3}, D. {choice_4}, E. {choice_5} 答え: {label}\n"
                    prompt += f"{sample['question']} A. {sample['choice_1']}, B. {sample['choice_2']}, C. {sample['choice_3']}, D. {sample['choice_4']}, E. {sample['choice_5']} 答え:"
                elif self.target_lang == "german":
                    for question, choice_1, choice_2, choice_3, choice_4, choice_5, label in zip(
                        demonstration_samples['question'], 
                        demonstration_samples['choice_1'],
                        demonstration_samples['choice_2'],
                        demonstration_samples['choice_3'],
                        demonstration_samples['choice_4'],
                        demonstration_samples['choice_5'],
                        demonstration_samples['prompt_label']
                    ):
                        prompt += f"{question} A. {choice_1}, B. {choice_2}, C. {choice_3}, D. {choice_4}, E. {choice_5} Antwort: {label}\n"
                    prompt += f"{sample['question']} A. {sample['choice_1']}, B. {sample['choice_2']}, C. {sample['choice_3']}, D. {sample['choice_4']}, E. {sample['choice_5']} Antwort:"
                elif self.target_lang == "swahili":
                    for question, choice_1, choice_2, choice_3, choice_4, choice_5, label in zip(
                        demonstration_samples['question'], 
                        demonstration_samples['choice_1'],
                        demonstration_samples['choice_2'],
                        demonstration_samples['choice_3'],
                        demonstration_samples['choice_4'],
                        demonstration_samples['choice_5'],
                        demonstration_samples['prompt_label']
                    ):
                        prompt += f"{question} A. {choice_1}, B. {choice_2}, C. {choice_3}, D. {choice_4}, E. {choice_5} Jibu: {label}\n"
                    prompt += f"{sample['question']} A. {sample['choice_1']}, B. {sample['choice_2']}, C. {sample['choice_3']}, D. {sample['choice_4']}, E. {sample['choice_5']} Jibu:"
                elif self.target_lang == "arabic":
                    for question, choice_1, choice_2, choice_3, choice_4, choice_5, label in zip(
                        demonstration_samples['question'], 
                        demonstration_samples['choice_1'],
                        demonstration_samples['choice_2'],
                        demonstration_samples['choice_3'],
                        demonstration_samples['choice_4'],
                        demonstration_samples['choice_5'],
                        demonstration_samples['prompt_label']
                    ):
                        prompt += f"{question} A. {choice_1}, B. {choice_2}, C. {choice_3}, D. {choice_4}, E. {choice_5} إجابة: {label}\n"
                    prompt += f"{sample['question']} A. {sample['choice_1']}, B. {sample['choice_2']}, C. {sample['choice_3']}, D. {sample['choice_4']}, E. {sample['choice_5']} إجابة:"
                else:
                    raise NotImplementedError
            else:
                for question, choice_1, choice_2, choice_3, choice_4, choice_5, label in zip(
                    demonstration_samples['question'], 
                    demonstration_samples['choice_1'], 
                    demonstration_samples['choice_2'], 
                    demonstration_samples['choice_3'], 
                    demonstration_samples['choice_4'], 
                    demonstration_samples['choice_5'], 
                    demonstration_samples['prompt_label']
                ):
                    prompt += f"{question} A. {choice_1}, B. {choice_2}, C. {choice_3}, D. {choice_4}, E. {choice_5} Answer: {label}\n"
                prompt += f"{sample['question']} A. {sample['choice_1']}, B. {sample['choice_2']}, C. {sample['choice_3']}, D. {sample['choice_4']}, E. {sample['choice_5']} Answer:"
            
        else:
            raise NotImplementedError
        
        return {"prompt": prompt}
    

    def get_test_dataset(self) -> datasets.Dataset:
        """Get the test dataset.

        Returns:
            datasets.Dataset: A test dataset.
        """
        if self.task_name == "shunk031/JGLUE":
            return self.dataset["validation"]
        else:
            return self.dataset["test"]
    

    def get_fewshot_test_dataset(self) -> datasets.Dataset:
        """Get the few-shot test dataset."""
        return self.test_dataset


if __name__ == "__main__":
    dataset_loader = DatasetLoader("xlsum", "japanese")
