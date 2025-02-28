from dataclasses import dataclass, field
from typing import Optional, List
import transformers



@dataclass
class SNUModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-12b")
    checkpoint_dir: Optional[str] = field(default="EleutherAI/pythia-12b")
    vision_tower: Optional[str] = field(default=None)
    resume_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the checkpoint to resume."},
    )
    
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
    )
    
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default="openai/clip-vit-large-patch14-336")
    mm_vision_select_layer: Optional[int] = field(
        default=-1
    )  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")

@dataclass
class SNUEvaluationArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    output_dir: str = field(
        default="./output", metadata={"help": "The output dir for logs and metrics"}
    )
    
    full_finetune: bool = field(
        default=False, metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(default=False, metadata={"help": "Use 8-bit adam."})
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=4, metadata={"help": "How many bits to use."})
    lora_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Which modules to use LoRA on. If None, will use all linear layers."
        },
    )
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    report_to: str = field(
        default="none",
        metadata={"help": "To use wandb or something else for reporting."},
    )

@dataclass
class SNUDataArguments:
    eval_type: str = field(default="OID", metadata={"help": "The type of evaluation to perform."
                                                    "choices=['OID', 'Polaris']"})
    eval_dataset_path: str = field(default="/projects/eunki/OID/OID-rated-image-captions.v2.test.alignment.tsv")
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    reward_prompt_file: Optional[str] = field(default=None)
    image_to_caption_file: Optional[str] = field(default=None)