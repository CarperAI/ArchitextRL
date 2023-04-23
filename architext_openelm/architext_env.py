from typing import Optional, Union
import numpy as np
from omegaconf import DictConfig, OmegaConf
from typing import List
from architext_genotype import ArchitextGenotype
from model import build_default_mutation_model
from model import ArchitextPromptMutation
from openelm.mutation_model import PromptModel
from openelm.environments import ENVS_DICT

architext_init_args = {"config": "architext_cfg.yaml"}


class BaseEnvironment:
    pass


class Architext(BaseEnvironment):
    """
    This will try to mutate layouts using either prompt mutation of diff model mutation.

    The Heat Loss Form Factor will be used as the quality metric, defined as:
    heat loss form factor = heat loss area / treated floor area, assuming that all area is treated.
    Numerically, this is calculated as: hlff = sum(surface_area) / floor_area

    The behavioral descriptors will be layout typology (measured by number of bedrooms and bathrooms) and the entropy
    of the floor area distribution across different spaces in the layout.
    """
    # Record different definitions of behaviour spaces in a dict. Feel free to add.
    behavior_mode_spec = {'hlff_and_fae': {'genotype_ndim': 2,
                                           'genotype_space': np.array([[0.5, 5.5], [0, 2000]]).T
                                           },
                          'entropy_and_typology': {'genotype_ndim': 2,
                                                   'genotype_space': np.array([[0, 5], [0, 10]]).T
                                                   }
                          }

    def __init__(self,
                 config: Union[str, dict, DictConfig],
                 mutation_model: Optional[PromptModel] = None,
                 behavior_mode='entropy_and_typology'
                 ):
        """
        Args:
            config: the config file or dict.
            mutation_model: (Optional) the model used to perform generation and mutation.
            behavior_mode: (Optional) the choice of behavior spaces (defined by diversity metrics)
        """
        self.np_rng = np.random.RandomState(seed=np.random.randint(1, 1e8))

        if isinstance(config, str):
            self.config = OmegaConf.load(config)
        elif isinstance(config, (dict, DictConfig)):
            self.config = DictConfig(config)
        else:
            raise ValueError

        # Use RNG to rotate random seeds during inference.
        self.rng = np.random.default_rng(seed=self.config.seed)

        self.behaviour_mode = behavior_mode
        self.genotype_ndim = self.behavior_mode_spec[self.behaviour_mode]['genotype_ndim']
        self.genotype_space = self.behavior_mode_spec[self.behaviour_mode]['genotype_space']

        self.model = build_default_mutation_model(self.config.mutation_model, self.config) \
            if mutation_model is None else mutation_model
        self.batch_size = self.config.batch_size

    def random(self) -> List[ArchitextGenotype]:
        """
        Sample layouts from the model by randomly selecting prompts.
        Returns:
            the generated layouts as a list of ArchitextGenotype.
        """
        return self.model.mutate_genotypes([None] * self.batch_size)

    def mutate(self, x: list[ArchitextGenotype]) -> List[ArchitextGenotype]:
        """
        Mutate layouts from a given Architext design.
        Args:
            x: the given Architext design.

        Returns:
            the generated layout as a list of ArchitextGenotype.
        """
        return self.model.mutate_genotypes(x)

    @staticmethod
    def fitness(x: ArchitextGenotype | None) -> float:
        if x is not None and x.valid:
            return x.hlff()
        else:
            return -np.inf

    @staticmethod
    def to_string(x: ArchitextGenotype) -> str:
        return str(x)

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
