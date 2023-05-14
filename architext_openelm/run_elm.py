import itertools
from collections import defaultdict

import hydra
import pickle
import numpy as np
from omegaconf import OmegaConf
from openelm.map_elites import MAPElites, Phenotype, MapIndex, Map
from tqdm import trange

from architext_env import Architext, architext_init_args
from openelm.environments import ENVS_DICT, Genotype

from architext_genotype import ArchitextGenotype
from util import save_folder

ARG_DICT = {"architext": architext_init_args}


# TODO: Add in a few features for MapElites class. Eventually they need to be moved to OpenELM.
class MyMAPElites(MAPElites):
    def to_mapindex(self, b: Phenotype) -> MapIndex:
        """Converts a phenotype (position in behaviour space) to a map index."""
        if b is None:
            return None
        # Check out-of-bounds
        if any(x < n for x, n in zip(b, self.env.behavior_space[0])):
            return None
        if any(x >= n for x, n in zip(b, self.env.behavior_space[1])):
            return None

        return tuple(np.digitize(x, bins) for x, bins in zip(b, self.bins))

    def export_genomes(self):
        """
        Exporting genomes without regard of orders.

        Returns:
            A list of Genotypes from `self.genomes` and `self.histories`
        """
        results = []
        for obj in self.genomes.array.flatten():
            if obj != 0.0:  # todo: Worry that this might not work if fill_value is not 0.0. We might need to redesign some stuff.
                results.append(obj)
        results.extend([obj for obj in self.recycled if obj is not None])

        return results

    def import_genomes(self, genotypes: list):
        """
        Importing a list of genomes and populate on the existing map.
        """
        # todo: this is a bit hacky... current_max_genome should be initialized in __init__
        if hasattr(self, "current_max_genome"):
            max_genome = self.current_max_genome
            max_fitness = self.env.fitness(max_genome)
        else:
            max_genome = None
            max_fitness = -np.inf

        for individual in genotypes:
            individual = ArchitextGenotype.from_dict(individual.design_json)
            fitness = self.env.fitness(individual)
            if np.isinf(fitness):
                continue
            map_ix = self.to_mapindex(individual.to_phenotype())
            # if the return is None, the individual is invalid and is thrown
            # into the recycle bin.
            if map_ix is None:
                self.recycled[self.recycled_count % len(self.recycled)] = individual
                self.recycled_count += 1
                continue

            if self.save_history:
                # TODO: thresholding
                self.history[map_ix].append(individual)
            self.nonzero[map_ix] = True

            # If new fitness greater than old fitness in niche, replace.
            if fitness > self.fitnesses[map_ix]:
                self.fitnesses[map_ix] = fitness
                self.genomes[map_ix] = individual
            # If new fitness is the highest so far, update the tracker.
            if fitness > max_fitness:
                max_fitness = fitness
                max_genome = individual

        self.current_max_genome = max_genome

    def load_maps(self, genomes: Map, recycled: list | None = None):
        # todo: the behavior of `init_map` in __init__ is not desirable. Need to rewrite a loading function.
        self.genomes = genomes
        self.nonzero: Map = Map(dims=self.genomes.dims, fill_value=False, dtype=bool)
        for idx in itertools.product(*[range(i) for i in self.genomes.dims]):
            if self.genomes[idx] != 0.0 and self.genomes[idx] is not None:
                self.nonzero[idx] = True

        if recycled is not None:
            self.recycled = recycled

    def search(self, init_steps: int, total_steps: int, atol: float = 1.0, after_step=lambda **kwargs: None) -> str:
        """
        Run the MAP-Elites search algorithm.

        Args:
            init_steps (int): Number of initial random solutions to generate.
            total_steps (int): Total number of steps to run the algorithm for,
                including initial steps.
            atol (float, optional): Tolerance for how close the best performing
                solution has to be to the maximum possible fitness before the
                search stops early. Defaults to 1.
            after_step (function, optional): A callback function to run after each step.

        Returns:
            str: A string representation of the best perfoming solution. The
                best performing solution object can be accessed via the
                `current_max_genome` class attribute.
        """
        tbar = trange(int(total_steps))
        max_fitness = -np.inf
        max_genome = None
        if self.save_history:
            self.history = defaultdict(list)

        for n_steps in tbar:
            if n_steps < init_steps or self.genomes.empty:
                # Initialise by generating initsteps random solutions.
                # If map is still empty: force to do generation instead of mutation.
                # TODO: use a separate sampler, move batch size to qd config.
                new_individuals: list[Genotype] = self.env.random()
            else:
                # Randomly select a batch of elites from the map.
                batch: list[Genotype] = []
                for _ in range(self.env.batch_size):
                    map_ix = self.random_selection()
                    batch.append(self.genomes[map_ix])
                # Mutate the elite.
                new_individuals = self.env.mutate(batch)

            # `new_individuals` is a list of generation/mutation. We put them
            # into the behavior space one-by-one.
            # TODO: account for the case where multiple new individuals are
            # placed in the same niche, for saving histories.
            for individual in new_individuals:
                fitness = self.env.fitness(individual)
                if np.isinf(fitness):
                    continue
                map_ix = self.to_mapindex(individual.to_phenotype())
                # if the return is None, the individual is invalid and is thrown
                # into the recycle bin.
                if map_ix is None:
                    self.recycled[self.recycled_count % len(self.recycled)] = individual
                    self.recycled_count += 1
                    continue

                if self.save_history:
                    # TODO: thresholding
                    self.history[map_ix].append(individual)
                self.nonzero[map_ix] = True

                # If new fitness greater than old fitness in niche, replace.
                if fitness > self.fitnesses[map_ix]:
                    self.fitnesses[map_ix] = fitness
                    self.genomes[map_ix] = individual
                # If new fitness is the highest so far, update the tracker.
                if fitness > max_fitness:
                    max_fitness = fitness
                    max_genome = individual

                    tbar.set_description(f"{max_fitness=:.4f}")
                # Stop if best fitness is within atol of maximum possible fitness.
                if np.isclose(max_fitness, self.env.max_fitness, atol=atol):
                    break

            after_step(locals=locals())

        self.current_max_genome = max_genome
        return str(max_genome)



class ArchitextELM:
    def __init__(self, cfg, model_cls=None, env_args: dict = None, behavior_mode=None) -> None:
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
        if behavior_mode is not None:
            env_args["behavior_mode"] = behavior_mode

        # Override diff model if `model_cls` is specified.
        if model_cls is not None:
            self.mutate_model = model_cls(self.cfg)
            env_args = {**env_args, "mutation_model": self.mutate_model}
        else:
            self.mutate_model = None

        self.environment = ENVS_DICT[self.cfg.env_name](**env_args)
        self.map_elites = MyMAPElites(
            self.environment,
            map_grid_size=(self.cfg.behavior_n_bins,),
            save_history=True,
            history_length=self.cfg.evo_history_length,
        )

    def run(self, evo_init_step_scheduler=None, suffix="", save_each_epochs=False, progress_bar=None):
        """
        Run MAPElites for self.cfg.epoch number of times. Can optionally add in an initial step scheduler
        to determine how many random steps are needed for each epoch.

        Args:
            evo_init_step_scheduler: (Optional) the scheduler function that takes in integers 1 ... epoch
                and outputs the corresponding number of initial random steps in each epoch. Note that the
                first epoch (indexed 0) will always perform `self.cfg.evo_init_step` random steps.
                By default, this function is the zero function (no random steps for second epochs and further).
            suffix: (Optional) filename suffix.
            save_each_epochs: (Optional) save after each epoch.
            progress_bar: (Optional) a streamlit progress bar.

        """
        if evo_init_step_scheduler is None:
            def evo_init_step_scheduler(step: int):
                return 0

        def after_step(locals):
            if progress_bar is not None:
                progress_bar.progress((locals["n_steps"] + 1) / locals["total_steps"],
                                      text="Generation in progress. Please wait.")

        for i in range(self.cfg.epoch):
            self.map_elites.search(
                init_steps=self.cfg.evo_init_steps, total_steps=self.cfg.evo_n_steps,
                after_step=after_step,
            )
            if save_each_epochs:
                # Histories are reset every time when `.search` is called. We have to dump and merge it.
                with open(str(save_folder / f'recycled.pkl'), 'wb') as f:
                    pickle.dump(self.map_elites.recycled, f)
                if suffix:
                    suffix = "_" + suffix
                with open(str(save_folder / f'map{suffix}.pkl'), 'wb') as f:
                    pickle.dump(self.map_elites.genomes, f)
                with open(str(save_folder / f'history.pkl'), 'wb') as f:
                    pickle.dump(self.map_elites.history, f)

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
    elm.run(save_each_epochs=True)


if __name__ == "__main__":
    main()
