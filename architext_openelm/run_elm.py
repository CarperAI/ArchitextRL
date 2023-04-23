from collections import defaultdict

import hydra
import pickle
from omegaconf import OmegaConf
from openelm.map_elites import MAPElites
from architext_env import Architext, architext_init_args
from openelm.environments import ENVS_DICT

ARG_DICT = {"architext": architext_init_args}


class ArchitextELM:
    def __init__(self, cfg, model_cls=None, env_args: dict = None) -> None:
        """
        Args:
            cfg: the config (e.g. OmegaConf who uses dot to access members).
            model_cls: (Optional) The class of diff model. One can apply alternative models here for comparison.
            env_args: (Optional) The argument dict for Environment.
        """
        self.cfg = cfg

        # Get the defaults if `env_args` is not specified.
        if env_args is None:
            env_args = ARG_DICT[self.cfg.env_name]
        env_args["config"] = self.cfg  # Override default environment config

        # Override diff model if `model_cls` is specified.
        if model_cls is not None:
            self.mutate_model = model_cls(self.cfg)
            env_args = {**env_args, "mutation_model": self.mutate_model}
        else:
            self.mutate_model = None

        self.environment = ENVS_DICT[self.cfg.env_name](**env_args)
        self.map_elites = MAPElites(
            self.environment,
            map_grid_size=(self.cfg.behavior_n_bins,),
            save_history=True,
            history_length=self.cfg.evo_history_length,
        )

    def run(self, evo_init_step_scheduler=None):
        """
        Run MAPElites for self.cfg.epoch number of times. Can optionally add in an initial step scheduler
        to determine how many random steps are needed for each epoch.

        Args:
            evo_init_step_scheduler: (Optional) the scheduler function that takes in integers 1 ... epoch
                and outputs the corresponding number of initial random steps in each epoch. Note that the
                first epoch (indexed 0) will always perform `self.cfg.evo_init_step` random steps.
                By default, this function is the zero function (no random steps for second epochs and further).

        """
        histories = defaultdict(list)
        if evo_init_step_scheduler is None:
            def evo_init_step_scheduler(step: int):
                return 0

        for i in range(self.cfg.epoch):
            self.map_elites.search(
                init_steps=self.cfg.evo_init_steps, total_steps=self.cfg.evo_n_steps
            )
            # Histories are reset every time when `.search` is called. We have to dump and merge it.
            for key, val in self.map_elites.history.items():
                histories[key].extend(val.copy())

            with open(f'recycled.pkl', 'wb') as f:
                pickle.dump(self.map_elites.recycled, f)
            with open(f'map.pkl', 'wb') as f:
                pickle.dump(self.map_elites.genomes, f)

            self.cfg.evo_init_steps = evo_init_step_scheduler(i+1)


# Load hydra config from yaml files and command line arguments.
@hydra.main(
    config_path="config", config_name="architext_gpt3.5_cfg", version_base="1.2"
)
def main(cfg):
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------  End -----------------")
    elm = ArchitextELM(cfg)
    elm.run()


if __name__ == "__main__":
    main()
