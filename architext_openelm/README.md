# OpenELM integration
This module integrates with [OpenELM](https://github.com/CarperAI/OpenELM) by providing the Architext 
Genotype/Environment, a local inference module and an example script to run everything.

## Quick start
To run the MAP-Elites generations using OpenELM, besides installing OpenELM, we only need to run the following.
```bash
pip install -r requirements.txt
python3 run_elm.py --config-name=architext_cfg  run_name=test
```
All the configs are in the file `architext_cfg.yaml`.

To dig into further details of the implementations, feel free to check out the codes of the following.
- `architext_genotype.py` defines each individual Architext design. They are what we want to evolve in the map.

- `architext_env.py` defines the environment including the mutation operator.

- `model.py` The local inference module that completes the prompt with floor design using coordinates
(See [architext/gptj-162M](https://huggingface.co/architext/gptj-162M)). 