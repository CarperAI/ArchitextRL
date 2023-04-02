from typing import Optional, Union
import numpy as np
from omegaconf import DictConfig, OmegaConf
from typing import List
from architext_genotype import ArchitextGenotype
from model import ArchitextPromptMutation
from openelm.mutation_model import PromptModel
from openelm.environments import ENVS_DICT


architext_init_args = {"config": "architext_cfg.yaml",
                       "prompts": None}


class BaseEnvironment:
    pass


class Architext(BaseEnvironment):
    """
    This will try to mutate layouts using architext-FIM models.

    The Heat Loss Form Factor will be used as the quality metric, defined as:
    heat loss form factor = heat loss area / treated floor area, assuming that all area is treated.
    Numerically, this is calculated as: hllff = sum(surface_area) / floor_area

    The behavioral descriptors will be layout typology (measured by number of bedrooms and bathrooms) and the entropy
    of the floor area distribution across different spaces in the layout.
    """
    # Record different definitions of behaviour spaces in a dict. Feel free to add.
    behavior_mode_spec = {'hlff_and_fae': {'genotype_ndim': 2,
                                           'genotype_space': np.array([[0.5, 5.5], [0, 2000]]).T
                                           }
                          }
    model_param = {'do_sample': True,
                   'num_beams': 1,
                   'max_length': 500}

    def __init__(self,
                 config: Union[str, dict, DictConfig],
                 prompts: Optional[list] = None,
                 mutation_model: Optional[PromptModel] = None,
                 behavior_mode='hlff_and_fae'
                 ):
        """
        Args:
            config: the config file or dict.
            prompts: list of different prompts that can be attached to selected layouts.
            model: (Optional) the model used to perform generation and mutation.
            behavior_mode: (Optional) the choice of behavior spaces (defined by diversity metrics)
        """
        self.np_rng = np.random.RandomState(seed=np.random.randint(1, 1e8))

        if isinstance(config, str):
            self.config = OmegaConf.load(config)
        elif isinstance(config, (dict, DictConfig)):
            self.config = DictConfig(config)
        else:
            raise ValueError

        if prompts is not None:
            self.prompts = prompts
        else:
            with open('../prompts.txt', 'r') as f:
                prompts = [p.strip() for p in f.read().split('\n') if p.strip()]
            self.prompts = ['[prompt] ' + prompt.rstrip() + ' [layout]' for prompt in prompts]

        # Use RNG to rotate random seeds during inference.
        self.rng = np.random.default_rng(seed=self.config.seed)

        self.behaviour_mode = behavior_mode
        self.genotype_ndim = self.behavior_mode_spec[self.behaviour_mode]['genotype_ndim']
        self.genotype_space = self.behavior_mode_spec[self.behaviour_mode]['genotype_space']

        self.model = ArchitextPromptMutation(self.config, self.prompts) if mutation_model is None else mutation_model

    def random(self) -> List[ArchitextGenotype]:
        """
        Sample layouts from the model by randomly selecting prompts.
        Returns:
            the generated layouts as a list of ArchitextGenotype.
        """
        return self._get_layout(None, parent=None)

    def mutate(self, x: ArchitextGenotype) -> List[ArchitextGenotype]:
        """
        Mutate layouts from a given Architext design.
        Args:
            x: the given Architext design.

        Returns:
            the generated layout as a list of ArchitextGenotype.
        """
        return self._get_layout(x.layout, parent=x)

    @staticmethod
    def fitness(x: ArchitextGenotype) -> float:
        if x.valid:
            return x.hlff()
        else:
            return -np.inf

    @staticmethod
    def to_string(x: ArchitextGenotype) -> str:
        return str(x)

    def _get_layout(self, full_prompt, parent: Optional[ArchitextGenotype]) -> list[ArchitextGenotype]:
        return [ArchitextGenotype(design_string=elem['result_obj'],
                                  height=self.config.height,
                                  parent=parent) for elem in
                self.model.generate_programs(full_prompt)]

    @staticmethod
    def _has_valid_output(x: ArchitextGenotype) -> bool:
        return x.valid

    def _update_seed(self):
        """
        Update the random seed in `self.config.seed` using `self.rng`.
        """
        self.config.seed = int(self.rng.integers(0, 1e8))

    @property
    def max_fitness(self):
        return 0

    @property
    # [start, end) of search intervals
    def behavior_space(self):
        return self.genotype_space

    @property
    def behavior_ndim(self):
        return self.behavior_space.shape[1]


ENVS_DICT['architext'] = Architext
