# Adapted from: https://github.com/ruikangliu/Quantized-Reasoning-Models

# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom evaluation tasks for LightEval."""

import random

from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    IndicesExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


latex_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(LatexExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
)

expr_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(ExprExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
)

expr_gsm8k_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(ExprExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
)

gpqa_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=5,
)

mmlu_pro_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=5,
)   


def prompt_fn(line, task_name: str = None):
    """Assumes the model is either prompted to emit \\boxed{answer} or does so automatically"""
    return Doc(
        task_name=task_name,
        query=f"{line['problem']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        choices=[line["solution"]],
        gold_index=0,
    )


def aime24_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line['Problem']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        choices=[str(line["Answer"])],
        gold_index=0,
    )


def aime25_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line['problem']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        choices=[str(line["answer"])],
        gold_index=0,
    )

def gsm8k_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line['question']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        choices=[line["answer"]],
        gold_index=0,
    )


def gpqa_prompt_fn(line, task_name: str = None):
    """Prompt template adapted from simple-evals: https://github.com/openai/simple-evals/blob/83ed7640a7d9cd26849bcb3340125002ef14abbe/common.py#L14"""
    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])
    query_template = "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
    query = query_template.format(A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=line["Question"])

    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
        gold_index=gold_index,
        instruction=query,
    )

def mmlu_pro_prompt_fn(line, task_name: str = None):
    """
    Prompt function for MMLU-Pro, adapted to match GPQA's prompt style:
      - Uses 'Answer: LETTER' format instruction
      - Labels options as A), B), C), ... with variable count (1–26)
      - Shuffles options and tracks gold index
    Input line expected to have:
      - 'question': str
      - 'options': List[str] of length N (1 <= N <= 26)
      - 'answer': str, one of 'A', 'B', ..., corresponding to original option index
    """
    original_options = line["options"]
    original_answer_label = line["answer"]  # e.g., "C"

    n = len(original_options)
    assert 1 <= n <= 26, f"Number of options must be between 1 and 26, got {n}"
    
    # Validate that answer label is within range
    valid_labels = [chr(ord("A") + i) for i in range(n)]
    assert original_answer_label in valid_labels, (
        f"Answer label '{original_answer_label}' not in valid labels {valid_labels}"
    )

    # Get correct answer text
    original_index = ord(original_answer_label) - ord("A")
    correct_answer_text = original_options[original_index]

    # Shuffle options
    shuffled_options = original_options.copy()
    random.shuffle(shuffled_options)

    # Find new index of the correct answer
    gold_index = shuffled_options.index(correct_answer_text)

    # Generate labels A, B, C, ..., up to N options, formatted as "A)", "B)", etc.
    labels = [chr(ord("A") + i) for i in range(n)]
    options_str = "\n".join(f"{label}) {opt}" for label, opt in zip(labels, shuffled_options))

    # Use GPQA-style instruction: require "Answer: LETTER" at the end
    query_template = (
        "Answer the following multiple choice question. Think step by step before answering. "
        "The last line of your response should be of the following format: 'Answer: $LETTER' "
        "(without quotes) where LETTER is one of {valid_letters}.\n\n"
        "{Question}\n\n{Options}"
    )
    valid_letters_str = "".join(labels)  # e.g., "ABCD" or "ABCDE"
    query = query_template.format(
        Question=line["question"],
        Options=options_str,
        valid_letters=valid_letters_str
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=labels,
        gold_index=gold_index,
        instruction=query,
    )

# Define tasks
aime24 = LightevalTaskConfig(
    name="aime24",
    suite=["custom"],
    prompt_function=aime24_prompt_fn,
    hf_repo="Maxwell-Jia/AIME_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[expr_gold_metric],
    version=1,
)
aime25 = LightevalTaskConfig(
    name="aime25",
    suite=["custom"],
    prompt_function=aime25_prompt_fn,
    hf_repo="yentinglin/aime_2025",  
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[expr_gold_metric],
    version=1,
)
aime90 = LightevalTaskConfig(
    name="aime90",
    suite=["custom"],
    prompt_function=aime25_prompt_fn,
    hf_repo="xiaoyuanliu/AIME90",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[expr_gold_metric],
    version=1,
)
math_500 = LightevalTaskConfig(
    name="math_500",
    suite=["custom"],
    prompt_function=prompt_fn,
    hf_repo="HuggingFaceH4/MATH-500",  
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric],
    version=1,
)
numina_math = LightevalTaskConfig(
    name="numina_math",
    suite=["custom"],
    prompt_function=aime25_prompt_fn,
    hf_repo="AI-MO/NuminaMath-1.5",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[expr_gsm8k_metric],
    version=1,
)
gsm8k = LightevalTaskConfig(
    name="gsm8k",
    suite=["custom"],
    prompt_function=gsm8k_prompt_fn,
    hf_repo="openai/gsm8k",   
    hf_subset="main",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[expr_gsm8k_metric],
    version=1,
)
gpqa_diamond = LightevalTaskConfig(
    name="gpqa:diamond",
    suite=["custom"],
    prompt_function=gpqa_prompt_fn,
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_diamond",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,  # needed for reasoning models like R1
    metric=[gpqa_metric],
    stop_sequence=[],  # no stop sequence, will use eos token
    trust_dataset=True,
    version=1,
)

mmlu_pro = LightevalTaskConfig(
    name="mmlu_pro",
    suite=["custom"],
    prompt_function=mmlu_pro_prompt_fn,
    hf_repo="TIGER-Lab/MMLU-Pro",
    hf_subset=None,
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,     
    few_shots_select=None,  
    generation_size=32768,    
    metric=[mmlu_pro_metric],   
    stop_sequence=[],       
    version=1,
)



# Add tasks to the table
TASKS_TABLE = []
TASKS_TABLE.append(aime24)
TASKS_TABLE.append(aime25)
TASKS_TABLE.append(aime90)
TASKS_TABLE.append(math_500)
TASKS_TABLE.append(numina_math)
TASKS_TABLE.append(gsm8k)
TASKS_TABLE.append(gpqa_diamond)
TASKS_TABLE.append(mmlu_pro)

# MODULE LOGIC
if __name__ == "__main__":
    print([t["name"] for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
