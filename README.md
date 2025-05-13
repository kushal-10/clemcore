### Updates
(March 2025): Version 2.0 of the benchmark has been [released](https://clembench.github.io/). And the framework is now pip installable. The games 
that make the benchmark got their own [repository](https://github.com/clp-research/clembench).

(February 2024): We have updated the framework code. If you have written games using the initial release version, see 
[this guide](docs/howto_update_to_v1.md) on how to update your game.

# clembench: A Framework for the Systematic Evaluation of Chat-Optimized Language Models as Conversational Agents

The cLLM (chat-optimized Large Language Model, "clem") framework tests such models' ability to engage in games – 
rule-constituted activities played using language.
The framework is a systematic way of probing for the situated language understanding of language using agents.

This repository contains Clemcore, the core framework code used to run the games discussed in
> Chalamalasetti, K., Götze, J., Hakimov, S., Madureira, B., Sadler, P., & Schlangen, D. (2023). clembench: Using Game 
> Play to Evaluate Chat-Optimized Language Models as Conversational Agents (arXiv:2305.13455). arXiv. 
> https://doi.org/10.48550/arXiv.2305.13455

### Clembench benchmark game set
The main set of games on which the [leaderboard](https://clembench.github.io/leaderboard.html) is based is now found in a separate repository:  
[Clembench repository](https://github.com/clp-research/clembench) You can find details of the contained games there.

### Evaluation Results
Results of Clembench benchmark runs can be found on the [main project website](https://clembench.github.io), under [leaderboard](https://clembench.github.io/leaderboard.html).

# Using the clemcore CLI
**Clemcore is now available as a library on PyPI, making it installable using pip.**  
We highly recommend installing Clemcore in its own separate Python 3.10 virtual environment, to assure that dependencies 
of the framework and the games are managed well. For the following examples, a default Python venv named `myclem` is 
assumed to be created and active.  
You can simply install the packaged library using a terminal:
```
(myclem) pip install clemcore
```
This means that there is no need to checkout this repository to run the framework.

> **Note to framework developers:** 
> 
> Framework developers that want to contribute to the clemcore framework, should still fork and checkout the repository and install the framework locally using `pip install -e .` for testing and then create a pull request with the changes.

Additional installation options are:
```
(myclem) pip install clemcore[huggingface] # dependencies for the local huggingface transformers backend
(myclem) pip install clemcore[vllm]        # dependencies for the local vllm backend
(myclem) pip install clemcore[slurk]       # dependencies for the slurk backend 
```
After the installation you will have access to the `clem` CLI tool. The main functions are:
```
(myclem) clem list games               # list the games available for a run
(myclem) clem list backends            # list the backends available for a run
(myclem) clem list models              # list the models available for a run
(myclem) clem run -g <game> -m <model> # runs specified game using specified model
(myclem) clem transcribe               # translates interactions into html files
(myclem) clem score                    # computes individual performance measures
(myclem) clem eval                     # computes overall performances measures; requires scores
```

The games to `run` can be checkout from the [clembench repository](https://github.com/clp-research/clembench).

This repository is tested on `Python 3.10`.