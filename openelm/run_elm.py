from collections import defaultdict

import hydra
import pickle
from omegaconf import OmegaConf
from openelm.map_elites import MAPElites
from architext_env import Architext, architext_init_args

ENVS_DICT = {"architext": Architext}
ARG_DICT = {"architext": architext_init_args}


class ArchitextELM:
    def __init__(self, cfg, diff_model_cls=None, env_args: dict = None) -> None:
        """
        Args:
            cfg: the config (e.g. OmegaConf who uses dot to access members).
            diff_model_cls: (Optional) The class of diff model. One can apply alternative models here for comparison.
            env_args: (Optional) The argument dict for Environment.
        """
        self.cfg = cfg

        # Get the defaults if `env_args` is not specified.
        if env_args is None:
            env_args = ARG_DICT[self.cfg.env_name]
        env_args["config"] = self.cfg  # Override default environment config

        # Override diff model if `diff_model_cls` is specified.
        if diff_model_cls is not None:
            self.diff_model = diff_model_cls(self.cfg)
            env_args = {**env_args, "diff_model": self.diff_model}
        else:
            self.diff_model = None

        self.environment = ENVS_DICT[self.cfg.env_name](**env_args)
        self.map_elites = MAPElites(
            self.environment,
            n_bins=self.cfg.behavior_n_bins,
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
                initsteps=self.cfg.evo_init_steps, totalsteps=self.cfg.evo_n_steps
            )
            # Histories are reset every time when `.search` is called. We have to dump and merge it.
            for key, val in self.map_elites.history.items():
                histories[key].extend(val.copy())

            with open(f'history.pkl', 'wb') as f:
                pickle.dump(histories, f)
            with open(f'map.pkl', 'wb') as f:
                pickle.dump(self.map_elites.genomes, f)

            self.cfg.evo_init_steps = evo_init_step_scheduler(i+1)


# Load hydra config from yaml files and command line arguments.
@hydra.main(
    config_path="config", config_name="architext_cfg", version_base="1.2"
)
def main(cfg):
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------  End -----------------")
    elm = ArchitextELM(cfg)
    elm.run()


if __name__ == "__main__":
    main()
