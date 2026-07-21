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

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import re
import re
import torch
import torch.distributed as dist
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image
from torch.utils.data import Dataset
# from transformers import Qwen2VLForConditionalGeneration

# from math_verify import parse, verify
from open_r1_multimodal.open_r1.trainer import VLMGRPOTrainer, GRPOConfig
from open_r1_multimodal.open_r1.vlm_modules import *
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
# from transformers import TrainingArguments
import yaml
import json
import random
import math
import numpy as np

from open_r1_multimodal.open_r1.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward
monkey_patch_qwen2_5vl_flash_attn()


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    image_root: Optional[str] = field(
        default='/mnt/data/yeq/Datasets/EMIT/',
        metadata={"help": "Root directory of the image"},
    )
    dataset_name: Optional[str] = field(
        default='/mnt/data/yeq/MLLM/IAD_MMAD/data_config/IAD.yaml',
        metadata={"help": "Dataset_path"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = True
    model_name_or_path: Optional[str] = field(
        default='/mnt/data/yeq/2025_6_6_3090_project/MLLM/Model_Huggface_Download/Qwen2.5-VL-3B-Instruct/',
        # default='/mnt/data/yeq/2025_6_6_3090_project/MLLM/Model_Huggface_Download/Qwen2.5-VL-7B-Instruct/',
        metadata={"help": "Model checkpoint for weights initialization."},
    )
    attn_implementation: Optional[str] = field(
        default='flash_attention_2',
        metadata={
            "help": "Which attention implementation to use. You can run `--attn_implementation=flash_attention_2`, in "
                    "which case you must install this manually by running `pip install flash-attn --no-build-isolation`."
        },
    )
    torch_dtype: Optional[str] = field(
        default='bfloat16',
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def extract_choice(text):
    """从文本中智能提取选择题答案（A, B, C, D）"""
    if not text:
        return None

    # 1. 清理和标准化文本
    text = re.sub(r'\s+', ' ', text)  # 规范化空格

    if len(text) < 3:
        text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号
        return text[0] if text else None

    # 2. 选项不应该前后有字母
    choices = re.findall(r'(?<![A-Z])([A-D])(?![A-Z])', text)
    choices = re.findall(r'(?<![a-z])([A-D])(?![a-z])', text)

    if not choices:
        return None

    # 3. 如果只有一个选项，直接返回
    if len(choices) == 1:
        return choices[0]

    # 4. 如果有多个选项，使用启发式规则
    choice_scores = {choice: 0 for choice in choices}

    # 4.1 关键词周围的选项加分
    keywords = [
        '答案', '选择', '正确', '是', '对',
        'answer', 'correct', 'choose', 'select', 'right',
        '认为', '应该', '觉得', 'think', 'believe', 'should'
    ]

    # 获取每个选项的上下文（前后20个字符）
    for choice in choices:
        pos = text.rfind(choice)
        context = text[max(0, pos - 20):min(len(text), pos + 20)]

        # 关键词加分
        for keyword in keywords:
            if keyword.upper() in context:
                choice_scores[choice] += 1

        # 如果选项靠近文本末尾则加分（通常是最终答案）
        if pos > len(text) * 0.7:  # 在文本后30%
            choice_scores[choice] += 2

        # 如果后面跟着标点符号则加分
        if pos < len(text) - 1 and text[pos + 1] in '。.!！,，':
            choice_scores[choice] += 1

    # 返回得分最高的选项
    return max(choice_scores.items(), key=lambda x: x[1])[0]

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    # pattern = r"<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [re.match(pattern, content) for content in completion_contents]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]
def roam_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion has the correct answer (A, B, C, or D)
    and if the thinking process supports the correct answer."""
    # 仅提取模型回答部分，忽略用户输入
    contents = [completion[-1]["content"] if isinstance(completion, list) else completion[0]["content"] for completion
                in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    think_tag_pattern = r'<think>(.*?)</think>'

    for content, sol in zip(contents, solution):
        reward = 0.0

        try:
            # First try to find if there's a proper format with <think> and <answer> tags
            match = re.search(r'<think>(.*?)</think>\s*<answer>(.*?)</answer>', content, re.DOTALL)
            if match:

                # 查找所有<answer>标签对
                content_answer_matches = re.findall(answer_tag_pattern, content, re.DOTALL)
                content_think_matches = re.findall(think_tag_pattern, content, re.DOTALL)

                # 获取答案部分
                if content_answer_matches:
                    content_answer = content_answer_matches[-1].strip()
                    answer_choice = extract_choice(content_answer)
                else:
                    answer_choice = None

                # 分析思维链部分
                if content_think_matches:
                    thinking = content_think_matches[-1].strip()
                    thinking_choice = extract_choice(thinking)
                else:
                    thinking_choice = None

                if thinking_choice == None and answer_choice == None:
                    reward = 0.0
                elif thinking_choice == None and answer_choice == sol:
                    reward = 0.8
                elif thinking_choice == None and answer_choice != sol:
                    reward = 0.05
                elif thinking_choice == sol and answer_choice == sol:
                    reward = 1.0
                elif thinking_choice != sol and answer_choice != sol and thinking_choice == answer_choice:
                    reward = 0.1
                elif thinking_choice != sol and answer_choice != sol and thinking_choice != answer_choice:
                    reward = 0.0
                else:
                    reward = 0.0
            else:
                reward = 0.0
                thinking_choice = None
                answer_choice = None
                content_answer_matches = None
                content_think_matches = None


        except Exception as e:
            content_answer_matches = f"Answer extract error: {str(e)}"

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Choice reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Extracted answer: {answer_choice}\n")
                f.write(f"Thinking conclusion: {thinking_choice}\n")
                f.write(f"GPT Answer: {content_answer_matches}\n")
                f.write(f"GPT Thinking: {content_think_matches}\n")
                f.write(f"Correct Answer: {sol}\n")

        rewards.append(reward)
    return rewards
def choice_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion has the correct answer (A, B, C, or D)
    and if the thinking process supports the correct answer."""
    # 仅提取模型回答部分，忽略用户输入
    contents = [completion[-1]["content"] if isinstance(completion, list) else completion[0]["content"] for completion
                in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    think_tag_pattern = r'<think>(.*?)</think>'

    for content, sol in zip(contents, solution):
        reward = 0.0

        try:
            # First try to find if there's a proper format with <think> and <answer> tags
            match = re.search(r'<think>(.*?)</think>\s*<answer>(.*?)</answer>', content, re.DOTALL)
            if match:

                # 查找所有<answer>标签对
                content_answer_matches = re.findall(answer_tag_pattern, content, re.DOTALL)
                content_think_matches = re.findall(think_tag_pattern, content, re.DOTALL)

                # 获取答案部分
                if content_answer_matches:
                    content_answer = content_answer_matches[-1].strip()
                    answer_choice = extract_choice(content_answer)
                else:
                    content_answer = None
                    answer_choice = None

                if answer_choice == sol:
                    reward = 1.0

            else:
                reward = 0.0
                thinking_choice = None
                answer_choice = None
                content_answer_matches = None
                content_think_matches = None


        except Exception as e:
            content_answer_matches = f"Answer extract error: {str(e)}"

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Choice easy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Extracted answer: {answer_choice}\n")
                f.write(f"GPT Answer: {content_answer_matches}\n")
                f.write(f"GPT Thinking: {content_think_matches}\n")
                f.write(f"Correct Answer: {sol}\n")

        rewards.append(reward)
    return rewards

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments, question_template: str):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []
        self.question_template = question_template

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        # Format into conversation
        example = self.list_data_dict[i]
        image_root = self.script_args.image_root
        def make_conversation(example):
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example['question']},
                ],
            }
        SYSTEM_PROMPT = (
            "A conversation between User and Assistant. The user asks a choice question, and the Assistant solves it. The assistant "
            "first thinks about the reasoning process in the mind and then provides the user with the answer."
            "Respond with your reasoning in <think> </think> tags "
            "followed by a single letter answer in <answer> </answer> tags."
        )
        # QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
        def make_conversation_image(example):
            return {
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {"type": "text", "text": "You are an industrial inspector who checks products by images. "},
                        ] + [
                        {"type": "text", "text": f"Following is the query image: "},
                        *({'type': 'image', 'text': None} for _ in range(len(example['image']))),
                        {"type": "text", "text": f"Following is the question list: "},
                        ] + [
                        {'type': 'text', 'text': example['question']},
                    ] + [
                        {"type": "text", "text": SYSTEM_PROMPT}
                    ]
                }]


            }

        if 'image' in example:
            image_path = os.path.join(image_root, example['image'][0])
            # In case the image is not found
            while not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, randomly selecting another image")
                new_index = random.randint(0, len(self.list_data_dict) - 1)
                example = self.list_data_dict[new_index]
                image_path = os.path.join(image_root, example['image'][0])

            # 加载图像
            image = Image.open(image_path).convert("RGB")

            # 检查图像尺寸并在必要时压缩
            max_width, max_height = 1000, 800  # 设置最大尺寸
            if image.width > max_width or image.height > max_height:
                # 计算宽高比
                old_width, old_height = image.width, image.height
                ratio = min(max_width / image.width, max_height / image.height)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                # 调整图像大小
                image = image.resize(new_size, Image.LANCZOS)
                # if os.getenv("DEBUG_MODE") == "true":
                #     print(f"Resized image {image_path} from {old_width}x{old_height} to {new_size[0]}x{new_size[1]}")
        else:
            image = None

        return {
            'image': image,
            'image_root': image_root,
            'problem': example['question'],
            'solution': example['solution'],
            'prompt': make_conversation_image(example)['prompt'] if 'image' in example else make_conversation(example)['prompt'],
        }


def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    # elif "internvl" in model_name_or_path.lower():
    #     return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(script_args, training_args, model_args):
    set_global_seed(training_args.seed)

    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)

    # Load the reward functions
    reward_funcs_registry = {
        "accuracy": choice_reward,
        "format": format_reward,
    }
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)



    # Load the dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args, question_template=vlm_module_cls.get_question_template(task_type="rec"))

    trainer_cls = VLMGRPOTrainer
    # Initialize the GRPO trainer
    training_args.gradient_checkpointing_kwargs = {
        "use_reentrant": False
    }
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        torch_dtype=model_args.torch_dtype,
    )



    # Train and push the model to the Hub
    trainer.train()
    trainer.save_model(training_args.output_dir)

    # Save and push to hub
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
