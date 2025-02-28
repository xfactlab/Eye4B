import os
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from typing import Optional
import transformers
from transformers.utils import ModelOutput
from transformers import PreTrainedModel, AutoModel, TrainingArguments, Qwen2VLForConditionalGeneration, PretrainedConfig, LlavaForConditionalGeneration, AutoModelForSequenceClassification
from dataclasses import dataclass, field
from peft import PeftModel, LoraModel, LoraConfig, get_peft_model
from modules import initialize_reward_model_head


class VLMRewardConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
@dataclass
class RewardArgs(TrainingArguments):
     
    vision_tower: Optional[str] = field(
        default=None,
        metadata={"help": ("The vision tower to use.")},
    )
    max_length: Optional[int] = field(
        default=4096,
        metadata={"help": ("The maximum length of the input.")},
    )


@dataclass
class VLMRewardModelOutput(ModelOutput):
    loss: Optional[Tensor] = None
    logits: Tensor = None
    rewards: Tensor = None

class VLMRewardModel(PreTrainedModel):
    def __init__(self, args, config: VLMRewardConfig):
        super(VLMRewardModel, self).__init__(config)
        self.config = config
        self.args = args
        self.model_name_or_path = args.model_name_or_path
        self.use_peft = args.use_peft
        self.peft_checkpoint_dir = args.peft_checkpoint_dir
        self.additional_reward_head = args.additional_reward_head
        self.checkpoint_dir = args.checkpoint_dir

        # Load the backbone model
        if "qwen" in self.model_name_or_path.lower():
            self.backbone_model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_name_or_path)
        elif "llava" in self.model_name_or_path.lower():
            self.backbone_model = LlavaForConditionalGeneration.from_pretrained(self.model_name_or_path)

        # Initialize or load LoRA model if applicable
        if self.use_peft:
            self.lora_config = LoraConfig(
                target_modules=args.lora_target_modules,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout
            )
            self.backbone_model = (
                PeftModel.from_pretrained(
                    self.backbone_model,
                    config=self.lora_config,
                    checkpoint_dir=self.peft_checkpoint_dir
                ) if self.peft_checkpoint_dir else get_peft_model(self.backbone_model, self.lora_config)
            )

        # Initialize the reward model head
        self.hidden_size = getattr(
            self.backbone_model.config, 
            "hidden_size",
            getattr(
                self.backbone_model.config,
                "d_model",
                getattr(
                    self.backbone_model.config,
                    "dim",
                    768
                )
            )
        )
        if self.additional_reward_head:
            self.reward_head = nn.Linear(self.hidden_size, 1)
            nn.init.normal_(self.reward_head.weight, mean=0.0, std=1/np.sqrt(self.hidden_size+1))
            self.reward_head.requires_grad_(True)
        else:
            self.backbone_model.lm_head = nn.Linear(self.backbone_model.config.hidden_size, 1, bias=False)
            nn.init.normal_(self.backbone_model.lm_head.weight, mean=0.0, std=1/np.sqrt(self.backbone_model.config.hidden_size+1))        
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw=None, labels=None, return_dict=True):
        """
        Forward pass for the RewardModel. Computes rewards based on the hidden states
        of the backbone model.

        Args:
            input_ids (Tensor): Tokenized input IDs.
            attention_mask (Tensor): Attention masks for input IDs.
            pixel_values (Tensor): Preprocessed images as input.
            image_grid_thw (Tensor): Image grid metadata.
            labels (Tensor, optional): True rewards for supervised learning.
            return_dict (bool): Whether to return a dictionary or tuple.

        Returns:
            RewardModelOutput: Output of the RewardModel.
        """
        # Ensure the backbone model does not cache during training
        self.backbone_model.config.use_cache = False

        # Forward pass through the backbone model
        # if image_grid_thw != None:
        outputs = self.backbone_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
            output_hidden_states=True,
        )   
            
        if self.additional_reward_head:
            last_hidden_state = outputs.hidden_states[-1]
            logits = outputs.logits
            last_hidden_state = last_hidden_state + 0.0 * torch.mean(logits)
            last_hidden_state_at_the_end = last_hidden_state[:, -1, :]
            last_hidden_state_at_the_end = last_hidden_state_at_the_end.type_as(
            self.reward_head.weight)
            rewards = self.reward_head(last_hidden_state_at_the_end).squeeze(-1)
            return VLMRewardModelOutput(rewards=rewards, logits=logits)
        else:
            logits = outputs.logits
            rewards = self.backbone_model.lm_head(outputs.hidden_states[-1]).squeeze(-1)
            return VLMRewardModelOutput(rewards=rewards, logits=logits)
            


    def _set_gradient_checkpointing(self, module, value=False):
        module.gradient_checkpointing = value
