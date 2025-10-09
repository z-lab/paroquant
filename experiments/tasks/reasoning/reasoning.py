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

# Copyright (c) 2025 ruikangliu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Custom evaluation tasks for LightEval."""

import random

from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    MultilingualExtractiveMatchMetric,
)
from lighteval.metrics.metrics import SampleLevelMetric
from lighteval.metrics.utils.extractive_match_utils import (
    IndicesExtractionConfig,
    ExtractionTarget,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.language import Language

from typing import Callable, Literal, Sequence
import numpy as np


def multilingual_extractive_match_metric(
    language: Language = Language.ENGLISH,
    gold_extraction_target: Sequence[ExtractionTarget] = (ExprExtractionConfig(),),
    pred_extraction_target: Sequence[ExtractionTarget] = (ExprExtractionConfig(),),
    aggregation_function: Callable[[list[float]], float] = max,
    fallback_mode: Literal["no_fallback", "first_match"] = "first_match",
    extraction_mode: Literal["first_match", "any_match"] = "any_match",
    precision: int = 6,
    timeout_seconds: int = 5,
) -> SampleLevelMetric:

    metric = MultilingualExtractiveMatchMetric(
        language=language,
        gold_extraction_target=gold_extraction_target,
        pred_extraction_target=pred_extraction_target,
        aggregation_function=aggregation_function,
        fallback_mode=fallback_mode,
        extraction_mode=extraction_mode,
        precision=precision,
        timeout_seconds=timeout_seconds,
    )
    return SampleLevelMetric(
        metric_name="extractive_match",
        sample_level_fn=metric,
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )


latex_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(LatexExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(
        ExprExtractionConfig(),
        LatexExtractionConfig(boxed_match_priority=0),
    ),
    aggregation_function=max,
)

expr_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(ExprExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(
        ExprExtractionConfig(),
        LatexExtractionConfig(boxed_match_priority=0),
    ),
    aggregation_function=max,
)

expr_gsm8k_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(ExprExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(
        ExprExtractionConfig(),
        LatexExtractionConfig(boxed_match_priority=0),
    ),
    aggregation_function=max,
)

gpqa_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=[],
    pred_extraction_target=[LatexExtractionConfig(boxed_match_priority=0)],
    precision=5,
)


def prompt_fn(line, task_name: str = None):
    """Assumes the model is either prompted to emit \\boxed{answer} or does so automatically"""
    return Doc(
        task_name=task_name,
        query=f"{line["problem"]}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        choices=[line["solution"]],
        gold_index=0,
    )


def aime24_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line["Problem"]}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        choices=[line["Answer"]],
        gold_index=0,
    )


def aime25_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line["problem"]}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        choices=[line["answer"]],
        gold_index=0,
    )


def gsm8k_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line["question"]}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        choices=[line["answer"]],
        gold_index=0,
    )


def gpqa_prompt_fn(line, task_name: str = None):
    """Prompt template adapted from simple-evals: https://github.com/openai/simple-evals/blob/83ed7640a7d9cd26849bcb3340125002ef14abbe/common.py#L14"""
    gold_index = random.randint(0, 3)
    choices = [
        line["Incorrect Answer 1"],
        line["Incorrect Answer 2"],
        line["Incorrect Answer 3"],
    ]
    choices.insert(gold_index, line["Correct Answer"])
    query_template = "Answer the following multiple choice question. Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{Question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}"
    query = query_template.format(
        A=choices[0],
        B=choices[1],
        C=choices[2],
        D=choices[3],
        Question=line["Question"],
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
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
    metrics=[expr_gold_metric],
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
    metrics=[expr_gold_metric],
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
    metrics=[expr_gold_metric],
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
    metrics=[latex_gold_metric],
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
    metrics=[expr_gsm8k_metric],
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
    metrics=[expr_gsm8k_metric],
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
    metrics=[gpqa_metric],
    stop_sequence=[],  # no stop sequence, will use eos token
    version=1,
)


# Add tasks to the table
TASKS_TABLE: list[LightevalTaskConfig] = []
TASKS_TABLE.append(aime24)
TASKS_TABLE.append(aime25)
TASKS_TABLE.append(aime90)
TASKS_TABLE.append(math_500)
TASKS_TABLE.append(numina_math)
TASKS_TABLE.append(gsm8k)
TASKS_TABLE.append(gpqa_diamond)

# MODULE LOGIC
if __name__ == "__main__":
    print([t.name for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
