#!/bin/bash

cd /path/to/llm-vocab-adaptation/eval/src

#####
# Parameters
#####
model_identifier="bloom-1b1"
plot_model_name="BLOOM-1.1B"
base_model_name="bigscience/bloom-1b1"
results_base_dir="/path/to/llm-vocab-adaptation/eval/logs"
model_cache_dir="/path/to/model/cache/dir"
data_cache_dir="/path/to/data/cache/dir"
xcsqa_dataset_path="/path/to/X-CSQA/dir"
kenswquad_dataset_path="/path/to/kenswquad/of/KenSwQuAD_final_7526_QA_pairs_csv.csv"

lang_codes=(
    "ja"
    "de"
    "sw"
    "ar"
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
        --model_cache_dir "$2" \
        --data_cache_dir "$3" \
        --task_name xnli \
        --target_lang "${lang_code_to_name[$4]}" \
        --results_dir ${results_base_dir}/xnli/$4 \
        --seed 42 \
        --num_max_samples 500 \
        --num_shots "$5" \
        --do_sample \
        --early_stopping \
        --plot_model_name "$6" \
        --plot_category "$7"
}

run_xnli_target() {
    python main.py \
        --model_name_or_path "$1" \
        --model_cache_dir "$2" \
        --data_cache_dir "$3" \
        --task_name xnli \
        --target_lang "${lang_code_to_name[$4]}" \
        --results_dir ${results_base_dir}/xnli/$4 \
        --seed 42 \
        --num_max_samples 500 \
        --num_shots "$5" \
        --do_sample \
        --early_stopping \
        --plot_model_name "$6" \
        --plot_category "$7" \
        --prompting_in_target_language
}

run_xcsqa_en() {
    python main.py \
        --model_name_or_path "$1" \
        --model_cache_dir "$2" \
        --data_cache_dir "$3" \
        --dataset_path ${xcsqa_dataset_path} \
        --task_name xcsqa \
        --target_lang "${lang_code_to_name[$4]}" \
        --results_dir ${results_base_dir}/xcsqa/$4 \
        --seed 42 \
        --num_max_samples 500 \
        --num_shots "$5" \
        --do_sample \
        --early_stopping \
        --plot_model_name "$6" \
        --plot_category "$7"
}

run_xcsqa_target() {
    python main.py \
        --model_name_or_path "$1" \
        --model_cache_dir "$2" \
        --data_cache_dir "$3" \
        --dataset_path ${xcsqa_dataset_path} \
        --task_name xcsqa \
        --target_lang "${lang_code_to_name[$4]}" \
        --results_dir ${results_base_dir}/xcsqa/$4 \
        --seed 42 \
        --num_max_samples 500 \
        --num_shots "$5" \
        --do_sample \
        --early_stopping \
        --plot_model_name "$6" \
        --plot_category "$7" \
        --prompting_in_target_language
}

run_xlsum_en() {
    python main.py \
        --model_name_or_path "$1" \
        --model_cache_dir "$2" \
        --data_cache_dir "$3" \
        --task_name xlsum \
        --target_lang "${lang_code_to_name[$4]}" \
        --results_dir ${results_base_dir}/xlsum/$4 \
        --seed 42 \
        --num_max_samples 500 \
        --num_shots "$5" \
        --do_sample \
        --early_stopping \
        --plot_model_name "$6" \
        --plot_category "$7" \
        --max_context_len 4096
}

run_xlsum_target() {
    python main.py \
        --model_name_or_path "$1" \
        --model_cache_dir "$2" \
        --data_cache_dir "$3" \
        --task_name xlsum \
        --target_lang "${lang_code_to_name[$4]}" \
        --results_dir ${results_base_dir}/xlsum/$4 \
        --seed 42 \
        --num_max_samples 500 \
        --num_shots "$5" \
        --do_sample \
        --early_stopping \
        --plot_model_name "$6" \
        --plot_category "$7" \
        --max_context_len 4096 \
        --prompting_in_target_language
}

run_xquad_en() {
    python main.py \
        --model_name_or_path "$1" \
        --model_cache_dir "$2" \
        --data_cache_dir "$3" \
        --dataset_path ${kenswquad_dataset_path} \
        --task_name xquad \
        --target_lang "${lang_code_to_name[$4]}" \
        --results_dir ${results_base_dir}/xquad/$4 \
        --seed 42 \
        --num_max_samples 500 \
        --num_shots "$5" \
        --do_sample \
        --early_stopping \
        --plot_model_name "$6" \
        --plot_category "$7"
}

run_xquad_target() {
    python main.py \
        --model_name_or_path "$1" \
        --model_cache_dir "$2" \
        --data_cache_dir "$3" \
        --dataset_path ${kenswquad_dataset_path} \
        --task_name xquad \
        --target_lang "${lang_code_to_name[$4]}" \
        --results_dir ${results_base_dir}/xquad/$4 \
        --seed 42 \
        --num_max_samples 500 \
        --num_shots "$5" \
        --do_sample \
        --early_stopping \
        --plot_model_name "$6" \
        --plot_category "$7" \
        --prompting_in_target_language
}

#####
# Evaluation
#####
# Zero-shot
for task in "${tasks[@]}"; do
    for lang_code in "${lang_codes[@]}"; do
        echo "Running ${base_model_name} on zero-shot ${task} in ${lang_code}..."
        if [ "${task}" == "xnli" ]; then
            run_xnli_en \
                "${base_model_name}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                0 \
                "${plot_model_name}" \
                "Source"
            
            run_xnli_target \
                "${base_model_name}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                0 \
                "${plot_model_name}" \
                "Source"
        
        elif [ "${task}" == "xcsqa" ]; then
            run_xcsqa_en \
                "${base_model_name}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                0 \
                "${plot_model_name}" \
                "Source"
            
            run_xcsqa_target \
                "${base_model_name}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                0 \
                "${plot_model_name}" \
                "Source"
        
        elif [ "${task}" == "xlsum" ]; then
            run_xlsum_en \
                "${base_model_name}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                0 \
                "${plot_model_name}" \
                "Source"
            
            run_xlsum_target \
                "${base_model_name}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                0 \
                "${plot_model_name}" \
                "Source"
        
        elif [ "${task}" == "xquad" ]; then
            run_xquad_en \
                "${base_model_name}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                0 \
                "${plot_model_name}" \
                "Source"
            
            run_xquad_target \
                "${base_model_name}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                0 \
                "${plot_model_name}" \
                "Source"
        fi
    done
done

# Few-shot
for task in "${tasks[@]}"; do
    for lang_code in "${lang_codes[@]}"; do
        echo "Running ${base_model_name} on few-shot ${task} in ${lang_code}..."
        if [ "${task}" == "xnli" ]; then
            run_xnli_en \
                "${base_model_name}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                5 \
                "${plot_model_name}" \
                "Source"
            
            run_xnli_target \
                "${base_model_name}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                5 \
                "${plot_model_name}" \
                "Source"
        
        elif [ "${task}" == "xcsqa" ]; then
            run_xcsqa_en \
                "${base_model_name}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                5 \
                "${plot_model_name}" \
                "Source"
            
            run_xcsqa_target \
                "${base_model_name}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                5 \
                "${plot_model_name}" \
                "Source"
        
        elif [ "${task}" == "xquad" ]; then
            run_xquad_en \
                "${base_model_name}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                3 \
                "${plot_model_name}" \
                "Source"
            
            run_xquad_target \
                "${base_model_name}" \
                "${model_cache_dir}" \
                "${data_cache_dir}" \
                "${lang_code}" \
                3 \
                "${plot_model_name}" \
                "Source"
        fi
    done
done
    