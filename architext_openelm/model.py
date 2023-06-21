import ast
import datetime
import json
import os
import random
import time
import re
from getpass import getpass
import openai

import numpy as np
import streamlit
import torch as torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM
from openelm.codegen.codegen_utilities import set_seed
from openelm.mutation_model import PromptModel

from architext_genotype import ArchitextGenotype
from concurrent.futures import ThreadPoolExecutor
from util import base_folder

max_json = re.compile(r"\{[\d\D]*\}")


class ArchitextPromptMutation(PromptModel):
    """
    Generating hf outputs on the local machine.
    """

    room_labels = ['bedroom1', 'kitchen', 'living_room', 'corridor', 'bathroom1']

    def __init__(self, cfg, prompts: list[str], default_height=2.0):
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
        self.default_height = default_height

        self.batch_size = self.cfg.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.cuda else "cpu")

        # Set up the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.cfg.pad_token

        # Set up the model
        # TODO: fix data parallel
        if self.cfg.gpus > 1:
            self.model = torch.nn.DataParallel(
                AutoModelForCausalLM.from_pretrained(self.cfg.model),
                device_ids=list(range(self.cfg.gpus))
            ).to(self.device)
            self.model.generate = self.model.module.generate
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model).to(self.device)

    def mutate_genotypes(self, inputs: list[ArchitextGenotype], **kwargs) -> list[ArchitextGenotype]:
        """
        This class does not use codes as intermediate representation. To fit into the genotype format, we output
        a dict with `program_str` (== `result_obj`) being the string describing a floor plan using coordinates.

        Args:
            inputs: the list of input genotypes.
        Returns:
            a list of mutated prompts.
        """
        prompts = []
        for pd in inputs:
            if pd is None:
                # Random generation
                mutated_prompt = random.choice(self.prompts)
            else:
                # Mutate the design string
                lines = pd.to_design_string().split(', ')
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
                                         num_beams=self.cfg.num_beams,
                                         num_return_sequences=1,
                                         max_length=self.cfg.gen_max_len,
                                         pad_token_id=50256,
                                         do_sample=True,
                                         **kwargs)

        mutated_design_str = self.tokenizer.batch_decode(completion)
        mutated_designs = [
            ArchitextGenotype(design_string=elem,
                              height=self.default_height if parent is None else parent.height,
                              parent=parent) for elem, parent in zip(mutated_design_str, inputs)]
        return mutated_designs


class ArchitextChatGPTMutation(PromptModel):
    """
    This prompt mutation calls GPT-3.5 API to generate designs
    """

    def __init__(self, cfg, prompts: list[str],
                 seed_designs: list[dict] | None = None, default_height=2.0,
                 count_tokens: bool = False, **kwargs):
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

        self.prompts = prompts
        self.default_height = default_height
        if seed_designs is None:
            try:
                with open("seed_designs.json", "r") as f:
                    seed_designs = ast.literal_eval(f.read())
            except:
                raise ValueError("Please provide a list of seed designs.")
        self.seed_designs = seed_designs

        if 'OPENAI_API_KEY' in os.environ:
            openai.api_key = os.environ['OPENAI_API_KEY']
        else:
            openai.api_key = getpass('Enter your OpenAI API key:')

        self.count_tokens = count_tokens
        if self.count_tokens:
            import streamlit

    def mutate_genotypes(self, genotypes: list[ArchitextGenotype], num_threads=10, **kwargs) -> list[ArchitextGenotype]:
        args = []
        for x in genotypes:
            example = random.choice(self.seed_designs) if x is None else x.to_dict()
            prompt = random.choice(self.prompts)
            args.append((example, prompt))

        with ThreadPoolExecutor(num_threads) as pool:
            results = list(pool.map(self._get_completion, args))
        mutated_json, usages = [r[0] for r in results], [r[1] for r in results]
        if self.count_tokens:
            for usage in usages:
                streamlit.session_state["prompt_tokens"] = \
                    streamlit.session_state.get("prompt_tokens", 0) + usage.get("prompt_tokens", 0)
                streamlit.session_state["tokens"] = \
                    streamlit.session_state.get("tokens", 0) + usage.get("total_tokens", 0)

        self._log_returns(mutated_json)

        mutated_genotypes = []
        for x, parent in zip(mutated_json, genotypes):
            if x is None:
                mutated_genotypes.append(None)
            else:
                assert isinstance(x, str)
                try:
                    new_dict = max_json.search(x).group(0)
                    new_dict = ast.literal_eval(new_dict)
                    mutated_genotypes.append(
                        ArchitextGenotype.from_dict(new_dict,
                                                    height=self.default_height if parent is None else parent.height,
                                                    parent=parent)
                    )
                except Exception as e:
                    with open("logs.txt", "a") as f:
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"[{timestamp} ERROR]: Parsing error in the return of GPT-3.5.\n"
                                f"Returned string: {str(x)}\n"
                                f"Exception: {str(e)}\n"
                                )
                    mutated_genotypes.append(None)

        return mutated_genotypes

    @staticmethod
    def _get_completion(args) -> tuple[str | None, dict]:
        """
        The inner loop of getting new design json from GPT-3.5 API.
        args: a tuple of (example, prompt)
            where example: dict, prompt: str
        """
        example, prompt = args

        requirements = "1. The prompt is all the info you have. The design detailed in `layout` follows the prompt as "\
                       "closely as possible.\n" \
                       "2. Different rooms cannot overlap.\n" \
                       "3. The room names should start with one of 'living_room', 'kitchen', 'bedroom', 'bathroom', " \
                       "'corridor'.\n" \
                       "4. Number of bathroom <= Number of bedroom <= 4.\n" \
                       "5. The return must be a valid JSON document.\n"
        sys_msg = "You are a REST API server receiving prompts describing a floor plan. You only return JSON " \
                  "documents describing your design. The format is the following:\n"\
                  "1. `prompt`: the original input prompt, \n"\
                  "2. `layout`: the room-by-room details of the floor plan in terms of the coordinates.\n"\
                  "An example is the following:\n" \
                  "```\n" + str(example) + "\n```\n" \
                  "In the `layout` field, each room is represented as a list of coordinates defining a polygon.\n" \
                  "The returned JSON document must satisfy " \
                  "the following requirements:\n" + requirements
        for i in range(5):
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}]
                )

                return completion["choices"][0]["message"]["content"], completion["usage"]
            except:
                time.sleep(60)
                if i == 4:
                    print("Warning: Calling OpenAI API failed for 5 consecutive times. Returning None. "
                          "See `logs.txt` for details.")
                    with open("logs.txt", "a") as f:
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"[{timestamp} ERROR]: Calling OAI API failed for 5 consecutive times.\n"
                                "Prompt: " + prompt + "\n"
                                                      "Example: " + str(example) + "\n")
                return None, {}

    @staticmethod
    def _log_returns(mutated_json):
        """
        Log the returned jsons to a file.
        """
        with open("returned.txt", "a") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp} INFO]: Returned JSONs: " + str(mutated_json) + "\n")


def build_default_mutation_model(key: str, cfg: DictConfig, count_tokens: bool = True):
    """
    A utility function that builds a default version of each mutation model class.

    Args:
        key (str): A key that defines the mutation model.
        cfg (DictConfig): A configuration object.
        count_tokens (bool): Whether to count the number of tokens in the generated text.
            Note: if True, it will import streamlit and count OpenAI API tokens using
            streamlit.session_state['tokens'].

    Returns:
        MutationModel: A mutation model.
    """
    with open(str(base_folder / 'prompts.txt'), 'r') as f:
        prompts = [p.strip() for p in f.read().split('\n') if p.strip()]
    aug_prompts = ['[prompt] ' + prompt.rstrip() + ' [layout]' for prompt in prompts]

    with open(str(base_folder / "seed_designs.json"), "r") as f:
        seed_designs = json.load(f)

    if key == "gpt-j":
        return ArchitextPromptMutation(cfg,
                                       prompts=aug_prompts,
                                       default_height=cfg.height)

    elif key == "chatgpt":
        return ArchitextChatGPTMutation(cfg,
                                        prompts=prompts,
                                        seed_designs=seed_designs,
                                        default_height=cfg.height,
                                        count_tokens=count_tokens)
