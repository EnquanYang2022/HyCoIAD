import trl
from typing import Optional,Union
from dataclasses import dataclass, field
from trl import ScriptArguments,ModelConfig
# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """
    per_device_train_batch_size: int = field(
        default=2, metadata={"help": "Batch size per device accelerator core/CPU for training."}
    )
    gradient_accumulation_steps: int = field(
        default=2,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    use_vllm: bool = field(
        default=True,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for "
                    "generation instead of the default model.generate(). Requires `vllm` to be installed."
        },
    )
    output_dir: Optional[str] = field(
        default='/mnt/data/yeq/MLLM/IAD-RL2/checkpoints/tet',
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written. Defaults to 'trainer_output' if not provided."
        },
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    logging_steps: float = field(
        default=1,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    num_train_epochs: float = field(default=1, metadata={"help": "Total number of training epochs to perform."})
    bf16: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
            )
        },
    )
    run_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional descriptor for the run. Notably used for wandb, mlflow comet and swanlab logging."
        },
    )
    data_seed: Optional[int] = field(default=42, metadata={"help": "Random seed to be used with data samplers."})
    save_steps: float = field(
        default=1,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    num_generations: Optional[int] = field(
        default=2,
        metadata={
            "help": "Number of generations to sample. The effective batch size (num_processes * per_device_batch_size "
                    "* gradient_accumulation_steps) must be evenly divisible by this value."
        },
    )
    max_completion_length: Optional[int] = field(
        default=2048,
        metadata={"help": "Maximum length of the generated completion."},
    )
    beta: float = field(
        default=0.04,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
                    "training speed, but may be numerically unstable for long training runs."
        },
    )

    # deepspeed: Optional[Union[dict, str]] = field(
    #     default='run_scripts/deepspeed_config/zero2.json',
    #     metadata={
    #         "help": (
    #             "Enable deepspeed and pass the path to deepspeed json config file (e.g. `ds_config.json`) or an already"
    #             " loaded json file as a dict"
    #         )
    #     },
    # )
    # benchmarks: list[str] = field(
    #     default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    # )
    # callbacks: list[str] = field(
    #     default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    # )
    # system_prompt: Optional[str] = field(
    #     default=None, metadata={"help": "The optional system prompt to use for benchmarking."}
    # )
    # hub_model_revision: Optional[str] = field(
    #     default="main", metadata={"help": "The Hub model branch to push the model to."}
    # )
    # overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    # push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    # wandb_entity: Optional[str] = field(
    #     default=None,
    #     metadata={"help": ("The entity to store runs under.")},
    # )
    # wandb_project: Optional[str] = field(
    #     default=None,
    #     metadata={"help": ("The project to store runs under.")},
    # )

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    data_file_paths: Optional[str] = field(
        default='data_config/train_R1.jsonl',
        metadata={"help": "Dataset_path"},
    )
    image_folders: str = field(
        default='/mnt/data/yeq/Datasets/Version2/3CAD/',
        metadata={"help": "Paths to image folders, separated by ':'"},
    )

    is_reward_customized_from_vlm_module: bool = field(
        default=False,
        metadata={"help": "Whether to use a customized reward from vlm module"},
    )

    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
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
    reward_method: Optional[str] = field(
        default='iou',
        metadata={
            "help": "Choose reward method: 'default', 'mcp', ..."
        },
    )
    task_type: Optional[str] = field(
        default='rec',
        metadata={"help": "Choose task type: 'default', 'gui', 'rec','ic','odLength',..."},
    )


    dataset_name: Optional[str] = field(
        default='this_is_not_used',
        metadata={"help": "Dataset_path"},
    )



@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False
    only_training_visual: bool=True
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
    # use_peft: bool = field(  #当这个设置为True时，后面对应的lora参数才会起作用
    #     default=True,
    #     metadata={"help": "Whether to use PEFT for training."},
    # )
    # lora_r: int = field(
    #     default=8,
    #     metadata={"help": "LoRA R value."},
    # )
    # lora_alpha: int = field(
    #     default=16,
    #     metadata={"help": "LoRA alpha."},
    # )
    # lora_dropout: float = field(
    #     default=0.05,
    #     metadata={"help": "LoRA dropout."},
    # )
    # lora_target_modules: Optional[list[str]] = field(
    #     default_factory=lambda: ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    #     metadata={"help": "LoRA target modules."},
    # )



