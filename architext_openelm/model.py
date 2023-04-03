import os
import random
from getpass import getpass
from typing import Optional

import numpy as np
import torch as torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM
from openelm.codegen.codegen_utilities import set_seed
from openelm.mutation_model import PromptModel


class ArchitextPromptMutation(PromptModel):
    """
    Generating hf outputs on the local machine.
    """

    room_labels = ['bedroom1', 'kitchen', 'living_room', 'corridor', 'bathroom1']

    def __init__(self, cfg, prompts: list[str]):
        """
        Args:
            cfg: the config dict.
            prompts: a list of default prompts
        """
        if isinstance(cfg, str):
            self.cfg = OmegaConf.load(cfg)
        elif isinstance(cfg, (dict, DictConfig)):
            self.cfg = DictConfig(cfg)
        else:
            raise ValueError

        set_seed(self.cfg.seed)
        # Use RNG to rotate random seeds during inference.
        self.rng = np.random.default_rng(seed=self.cfg.seed)
        self.prompts = prompts

        # Put huggingface token in env variable or enter with masked input field.
        if 'HF_TOKEN' not in os.environ:
            self.token = getpass('Enter your HF token:')
        else:
            self.token = os.environ['HF_TOKEN']

        self.batch_size = self.cfg.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.cuda else "cpu")

        # Set up the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model, use_auth_token=self.token)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.cfg.pad_token

        # Set up the model
        # TODO: fix data parallel
        if self.cfg.gpus > 1:
            self.model = torch.nn.DataParallel(
                AutoModelForCausalLM.from_pretrained(self.cfg.model, use_auth_token=self.token),
                device_ids=list(range(self.cfg.gpus))
            ).to(self.device)
            self.model.generate = self.model.module.generate
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model,
                                                              use_auth_token=self.token).to(self.device)

    def generate_programs(self, prompt_dicts: list[dict[str, str]], **kwargs) -> list[str]:
        """
        This class does not use codes as intermediate representation. To fit into the genotype format, we output
        a dict with `program_str` (== `result_obj`) being the string describing a floor plan using coordinates.

        Args:
            prompt_dicts: in this special case, prompt_dicts = [{'prompt': prompt_str1}, {'prompt': prompt_str2}, ...]
        Returns:
            a list of mutated prompts.
        """
        prompts = []
        for pd in prompt_dicts:
            prompt_str = pd["prompt"]
            if prompt_str is None:
                # Random generation
                mutated_prompt = random.choice(self.prompts)
            else:
                # Mutate the given string
                lines = prompt_str.split(', ')
                random_prompt = random.choice(self.prompts)
                cut_off = np.random.randint(1, 3, size=1)[0]
                cut_off = min(cut_off, len(lines) - 1)
                mutated_prompt = random_prompt + ' ' + ', '.join(lines[1:cut_off + 1]) + ", " + random.choice(
                    self.room_labels) + ":"
            prompts.append(mutated_prompt)

        completion = self.model.generate(**self.tokenizer(prompts,
                                                          return_tensors="pt",
                                                          padding=True,
                                                          truncation=True).to(self.device),
                                         num_beams=self.cfg.num_generation,
                                         num_return_sequences=self.cfg.num_generation,
                                         max_length=self.cfg.gen_max_len,
                                         pad_token_id=50256,
                                         **kwargs)

        return self.tokenizer.batch_decode(completion)


class ArchitextChatGPTMutation(PromptModel):
    """
    This prompt mutation calls GPT-3.5 API to generate designs
    """
