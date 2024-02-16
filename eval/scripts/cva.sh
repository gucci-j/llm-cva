#!/bin/bash

cd /path/to/llm-vocab-adaptation/eval/src

#####
# Parameters
#####
lang_code="sw"
num_steps=9000
model_identifier="bloom-1b1"
plot_model_name="BLOOM-1.1B"
results_base_dir="/path/to/llm-vocab-adaptation/eval/logs"
model_base_dir="/path/to/dir/for/tuned/models"
model_cache_dir="/path/to/model/cache/dir"
data_cache_dir="/path/to/data/cache/dir"
xcsqa_dataset_path="/path/to/X-CSQA/dir"
kenswquad_dataset_path="/path/to/kenswquad/dataset/of/KenSwQuAD_final_7526_QA_pairs_csv.csv"

# Please change the base model paths to the correct paths on your system.
models=(
    "${model_base_dir}/${model_identifier}-${lang_code}-clp-tuned/checkpoint-${num_steps}"
    "${model_base_dir}/${model_identifier}-${lang_code}-clp-plus-tuned/checkpoint-${num_steps}"
    "${model_base_dir}/${model_identifier}-${lang_code}-heuristics-tuned/checkpoint-${num_steps}"
    "${model_base_dir}/${model_identifier}-${lang_code}-focus-tuned/checkpoint-${num_steps}"
    "${model_base_dir}/${model_identifier}-${lang_code}-random-tuned/checkpoint-${num_steps}"
)
declare -A tokenizer_paths=(
    ["${model_base_dir}/${model_identifier}-${lang_code}-clp-tuned/checkpoint-${num_steps}"]="${model_base_dir}/${model_identifier}-${lang_code}-clp"
    ["${model_base_dir}/${model_identifier}-${lang_code}-clp-plus-tuned/checkpoint-${num_steps}"]="${model_base_dir}/${model_identifier}-${lang_code}-clp-plus"
    ["${model_base_dir}/${model_identifier}-${lang_code}-heuristics-tuned/checkpoint-${num_steps}"]="${model_base_dir}/${model_identifier}-${lang_code}-heuristics"
    ["${model_base_dir}/${model_identifier}-${lang_code}-focus-tuned/checkpoint-${num_steps}"]="${model_base_dir}/${model_identifier}-${lang_code}-focus"
    ["${model_base_dir}/${model_identifier}-${lang_code}-random-tuned/checkpoint-${num_steps}"]="${model_base_dir}/${model_identifier}-${lang_code}-random"
)
declare -A plot_category_names=(
    ["${model_base_dir}/${model_identifier}-${lang_code}-clp-tuned/checkpoint-${num_steps}"]="CLP Initialization"
    ["${model_base_dir}/${model_identifier}-${lang_code}-clp-plus-tuned/checkpoint-${num_steps}"]="CLP+ Initialization"
    ["${model_base_dir}/${model_identifier}-${lang_code}-heuristics-tuned/checkpoint-${num_steps}"]="Heuristics Initialization"
    ["${model_base_dir}/${model_identifier}-${lang_code}-focus-tuned/checkpoint-${num_steps}"]="FOCUS Initialization"
    ["${model_base_dir}/${model_identifier}-${lang_code}-random-tuned/checkpoint-${num_steps}"]="Random Initialization"
)
declare -A lang_code_to_name=(
    ["ja"]="japanese"
    ["de"]="german"
    ["sw"]="swahili"
    ["ar"]="arabic"
)
tasks=(
    "xnli" 
    "xcsqa"
    "xlsum"
    "xquad"
)

#####
# Eval functions
#####
run_xnli_en() {
    python main.py \
        --model_name_or_path "$1" \
        --tokenizer_name_or_path "$2" \
        --model_cache_dir "$3" \
        --data_cache_dir "$4" \
        --task_name xnli \
        --target_lang "${lang_code_to_name[$5]}" \
        --results_dir ${results_base_dir}/xnli/$5 \
        --seed 42 \
        --num_max_samples 500 \
        --is_peft \
        --num_shots "$6" \
        --do_sample \
        --early_stopping \
        --plot_model_name "$7" \
        --plot_category "$8"
}

run_xnli_target() {
    python main.py \
        --model_name_or_path "$1" \
        --tokenizer_name_or_path "$2" \
        --model_cache_dir "$3" \
        --data_cache_dir "$4" \
        --task_name xnli \
        --target_lang "${lang_code_to_name[$5]}" \
        --results_dir ${results_base_dir}/xnli/$5 \
        --seed 42 \
        --num_max_samples 500 \
        --is_peft \
        --num_shots "$6" \
        --do_sample \
        --early_stopping \
        --plot_model_name "$7" \
        --plot_category "$8" \
        --prompting_in_target_language
}

run_xcsqa_en() {
    python main.py \
        --model_name_or_path "$1" \
        --tokenizer_name_or_path "$2" \
        --model_cache_dir "$3" \
        --data_cache_dir "$4" \
        --dataset_path ${xcsqa_dataset_path} \
        --task_name xcsqa \
        --target_lang "${lang_code_to_name[$5]}" \
        --results_dir ${results_base_dir}/xcsqa/$5 \
        --seed 42 \
        --num_max_samples 500 \
        --is_peft \
        --num_shots "$6" \
        --do_sample \
        --early_stopping \
        --plot_model_name "$7" \
        --plot_category "$8"
}

run_xcsqa_target() {
    python main.py \
        --model_name_or_path "$1" \
        --tokenizer_name_or_path "$2" \
        --model_cache_dir "$3" \
        --data_cache_dir "$4" \
        --dataset_path ${xcsqa_dataset_path} \
        --task_name xcsqa \
        --target_lang "${lang_code_to_name[$5]}" \
        --results_dir ${results_base_dir}/xcsqa/$5 \
        --seed 42 \
        --num_max_samples 500 \
        --is_peft \
        --num_shots "$6" \
        --do_sample \
        --early_stopping \
        --plot_model_name "$7" \
        --plot_category "$8" \
        --prompting_in_target_language
}

run_xlsum_en() {
    python main.py \
        --model_name_or_path "$1" \
        --tokenizer_name_or_path "$2" \
        --model_cache_dir "$3" \
        --data_cache_dir "$4" \
        --task_name xlsum \
        --target_lang "${lang_code_to_name[$5]}" \
        --results_dir ${results_base_dir}/xlsum/$5 \
        --seed 42 \
        --num_max_samples 500 \
        --is_peft \
        --num_shots "$6" \
        --do_sample \
        --early_stopping \
        --plot_model_name "$7" \
        --plot_category "$8" \
        --max_context_len 4096
}

run_xlsum_target() {
    python main.py \
        --model_name_or_path "$1" \
        --tokenizer_name_or_path "$2" \
        --model_cache_dir "$3" \
        --data_cache_dir "$4" \
        --task_name xlsum \
        --target_lang "${lang_code_to_name[$5]}" \
        --results_dir ${results_base_dir}/xlsum/$5 \
        --seed 42 \
        --num_max_samples 500 \
        --is_peft \
        --num_shots "$6" \
        --do_sample \
        --early_stopping \
        --plot_model_name "$7" \
        --plot_category "$8" \
        --max_context_len 4096 \
        --prompting_in_target_language
}

run_xquad_en() {
    python main.py \
        --model_name_or_path "$1" \
        --tokenizer_name_or_path "$2" \
        --model_cache_dir "$3" \
        --data_cache_dir "$4" \
        --dataset_path ${kenswquad_dataset_path} \
        --task_name xquad \
        --target_lang "${lang_code_to_name[$5]}" \
        --results_dir ${results_base_dir}/xquad/$5 \
        --seed 42 \
        --num_max_samples 500 \
        --is_peft \
        --num_shots "$6" \
        --do_sample \
        --early_stopping \
        --plot_model_name "$7" \
        --plot_category "$8"
}

run_xquad_target() {
    python main.py \
        --model_name_or_path "$1" \
        --tokenizer_name_or_path "$2" \
        --model_cache_dir "$3" \
        --data_cache_dir "$4" \
        --dataset_path ${kenswquad_dataset_path} \
        --task_name xquad \
        --target_lang "${lang_code_to_name[$5]}" \
        --results_dir ${results_base_dir}/xquad/$5 \
        --seed 42 \
        --num_max_samples 500 \
        --is_peft \
        --num_shots "$6" \
        --do_sample \
        --early_stopping \
        --plot_model_name "$7" \
        --plot_category "$8" \
        --prompting_in_target_language
}

#####
# Evaluation
#####
# Zero-shot
for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        echo "Running ${model} on zero-shot ${task}..."
        if [ "${task}" == "xnli" ]; then
            run_xnli_en \
                "${model}" \
                "${tokenizer_paths[${model}]}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                0 \
                "${plot_model_name}" \
                "${plot_category_names[${model}]}"
            
            run_xnli_target \
                "${model}" \
                "${tokenizer_paths[${model}]}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                0 \
                "${plot_model_name}" \
                "${plot_category_names[${model}]}"

        elif [ "${task}" == "xcsqa" ]; then
            run_xcsqa_en \
                "${model}" \
                "${tokenizer_paths[${model}]}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                0 \
                "${plot_model_name}" \
                "${plot_category_names[${model}]}"
            
            run_xcsqa_target \
                "${model}" \
                "${tokenizer_paths[${model}]}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                0 \
                "${plot_model_name}" \
                "${plot_category_names[${model}]}"

        elif [ "${task}" == "xlsum" ]; then
            run_xlsum_en \
                "${model}" \
                "${tokenizer_paths[${model}]}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                0 \
                "${plot_model_name}" \
                "${plot_category_names[${model}]}"
            
            run_xlsum_target \
                "${model}" \
                "${tokenizer_paths[${model}]}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                0 \
                "${plot_model_name}" \
                "${plot_category_names[${model}]}"

        elif [ "${task}" == "xquad" ]; then
            run_xquad_en \
                "${model}" \
                "${tokenizer_paths[${model}]}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                0 \
                "${plot_model_name}" \
                "${plot_category_names[${model}]}"
            
            run_xquad_target \
                "${model}" \
                "${tokenizer_paths[${model}]}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                0 \
                "${plot_model_name}" \
                "${plot_category_names[${model}]}"
        fi
    done
done

# Few-shot
for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        echo "Running ${model} on few-shot ${task}..."
        if [ "${task}" == "xnli" ]; then
            run_xnli_en \
                "${model}" \
                "${tokenizer_paths[${model}]}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                5 \
                "${plot_model_name}" \
                "${plot_category_names[${model}]}"
            
            run_xnli_target \
                "${model}" \
                "${tokenizer_paths[${model}]}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                5 \
                "${plot_model_name}" \
                "${plot_category_names[${model}]}"
        
        elif [ "${task}" == "xcsqa" ]; then
            run_xcsqa_en \
                "${model}" \
                "${tokenizer_paths[${model}]}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                5 \
                "${plot_model_name}" \
                "${plot_category_names[${model}]}"
            
            run_xcsqa_target \
                "${model}" \
                "${tokenizer_paths[${model}]}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                5 \
                "${plot_model_name}" \
                "${plot_category_names[${model}]}"
        
        elif [ "${task}" == "xquad" ]; then
            run_xquad_en \
                "${model}" \
                "${tokenizer_paths[${model}]}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                3 \
                "${plot_model_name}" \
                "${plot_category_names[${model}]}"
            
            run_xquad_target \
                "${model}" \
                "${tokenizer_paths[${model}]}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                3 \
                "${plot_model_name}" \
                "${plot_category_names[${model}]}"
        fi
    done
done
