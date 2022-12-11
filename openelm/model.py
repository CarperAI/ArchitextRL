import os
import random
from getpass import getpass
from typing import Optional

import numpy as np
import torch as torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM
from elm.codegen.codegen_utilities import set_seed
from elm.diff_model import Model


class ArchitextPromptMutation(Model):
    """
    Generating hf outputs in the local machine.
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
        if self.cfg.gpus > 1:
            self.model = torch.nn.DataParallel(
                AutoModelForCausalLM.from_pretrained(self.cfg.model, use_auth_token=self.token),
                device_ids=list(range(self.cfg.gpus))
            ).to(self.device)
            self.model.generate = self.model.module.generate
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model,
                                                              use_auth_token=self.token).to(self.device)

    def __call__(self, prompt, **kwargs):
        config = {'return_tensors': 'pt',
                  'padding': True}

        output = self.model.generate(**self.tokenizer(prompt, **config).to(self.device),
                                     num_beams=self.cfg.num_generation,
                                     num_return_sequences=self.cfg.num_generation,
                                     max_length=self.cfg.gen_max_len,
                                     pad_token_id=50256,
                                     **kwargs)

        return self.tokenizer.batch_decode(output)

    def generate_program(self,
                         seed_str: Optional[str],
                         show_prompts: bool = False) -> list[dict]:
        """
        This class does not use codes as intermediate representation. To fit into the genotype format, we output
        a dict with `program_str` (== `result_obj`) being the string describing a floor plan using coordinates.

        Args:
            seed_str: the original prompt. If None, randomly choose a prompt from `self.prompts`.
            show_prompts: (Optional) this is a test mechanics that dumps the first 2 prompts to stdout in every call.
        Returns:
            a list of completed prompts.
        """
        if seed_str is None:
            # Random generation
            prompts = random.choices(self.prompts, k=self.batch_size)
        else:
            # Mutate the given string
            lines = seed_str.split(', ')
            prompts = []
            for i in range(self.batch_size):
                random_prompt = random.choice(self.prompts)
                cut_off = np.random.randint(1, 3, size=1)[0]
                cut_off = min(cut_off, len(lines) - 1)
                prompts.append(random_prompt + ' ' + ', '.join(lines[1:cut_off + 1]) + ", " + random.choice(
                    self.room_labels) + ":")

        if show_prompts:
            for i in range(min(2, self.batch_size)):
                print(prompts[i])

        return [{'program_str': st,
                 'result_obj': st,
                 "error_code": 0
                 } for st in self.__call__(prompts)]
