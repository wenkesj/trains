_Turn your Evals into Trains._

> This project is in it's alpha stages. We're trying something new here! Laying down the tracks for something potentially really cool and easy to use. For that reason, we're open to changing directions, the code, the protocol, you name it. The examples in this repo are suggestive, they are not meant to prove any specific hyperparameters work and each of them should be tuned on their own. Inference and training can get expensive, especially to experiment, so please be mindful.
>
> **Please train on data responsibly and respectfully.**

Trains is an **experimental** Python library for converting model traces, native agent sessions, benchmark trajectories, and datums into [Hugging Face Datasets](https://huggingface.co/docs/datasets). Inject Trains in your evals loop, and you're inches away from discovering RL.

Trains _tries it's best_ to recover messages, tools, model identity, rewards, and task groups from traces and popular eval framework formats out there.

There is no one standard that exists out there right now, this isn't try to be one.

![Trains workflow: tracing, benchmark artifacts, and native agent sessions become HF chat datasets, provider-native datum datasets, and updated policies.](images/overview.png)

## Start with your data

| Source | Next step |
| --- | --- |
| Phoenix, AX, Langfuse, or LangSmith experiments | [Tracing](#/docs/guides/tracing) |
| Harbor, Harvey LAB, or Inspect artifacts | [Benchmarks](#/docs/guides/benchmarks) |
| Native coding-agent sessions or Hub traces | [Sessions](#/docs/guides/sessions) |
| Training preparation | [Datum](#/docs/guides/datum) |

## Install

Install from GitHub source; PyPI publishing is disabled. Git and Python 3.12+ are required.

```installation
pip install "trains-sdk @ git+https://github.com/wenkesj/trains.git#subdirectory=trains"
uv add "trains-sdk @ git+https://github.com/wenkesj/trains.git#subdirectory=trains"
poetry add "git+https://github.com/wenkesj/trains.git#subdirectory=trains"
```

## Run an example

The [examples](#/docs/guides/examples) cover dataset generation to training.

## Python reference

Browse the [Python](#/docs/reference/python) SDK reference.
