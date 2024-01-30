# Large Language Models as Generalizable Policies for Embodied Tasks

This software project accompanies the research paper, [Large Language Models as Generalizable Policies for Embodied Tasks](https://arxiv.org/abs/2310.17722). See [llm-rl.github.io](https://llm-rl.github.io) for more information

**Abstract**: We show that large language models (LLMs) can be adapted to be generalizable policies for embodied visual tasks. Our approach, called Large LAnguage model Reinforcement Learning Policy (LLaRP), adapts a pre-trained frozen LLM to take as input text instructions and visual egocentric observations and output actions directly in the environment. Using reinforcement learning, we train LLaRP to see and act solely through environmental interactions. We show that LLaRP is robust to complex paraphrasings of task instructions and can generalize to new tasks that require novel optimal behavior. In particular, on 1,000 unseen tasks it achieves 42% success rate, 1.7x the success rate of other common learned baselines or zero-shot applications of LLMs. Finally, to aid the community in studying language conditioned, massively multi-task, embodied AI problems we release a novel benchmark, Language Rearrangement, consisting of 150,000 training and 1,000 testing tasks for language-conditioned rearrangement.

## Getting Started

### Installation
- Setup Python environment:
    - `conda create -n llarp -y python=3.9`
    - `conda activate llarp`
- Install [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) `conda install -y habitat-sim==0.3.0 withbullet  headless -c conda-forge -c aihabitat`
- Install [Habitat-Lab](https://github.com/facebookresearch/habitat-lab):
    - `git clone -b 'v0.3.0' --depth 1 https://github.com/facebookresearch/habitat-lab.git ~/habitat-lab`
    - `cd ~/habitat-lab`
    - `pip install -e habitat-lab`
    - `pip install -e habitat-baselines`
- Install [VC-1](https://eai-vc.github.io): 
    - `git clone -b 76fe35e87b1937168f1ec4b236e863451883eaf3 https://github.com/facebookresearch/eai-vc.git ~/eai-vc`
    - `git submodule update --init --recursive`
    - `pip install -e ./vc_models`
- Install this repository, first clone this repo and then run `pip install -e .` in the cloned directory.
- Download YCB and ReplicaCAD dataset for the Language Rearrangement task. Run the following in this code base's directory.
    - `conda install -y -c conda-forge git-lfs`
    - `python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets`
- Download LLaMA-1 weights. Instructions on how to do this are [here](https://huggingface.co/docs/transformers/main/model_doc/llama). Place the model weights in `data/hf_llama_7B/` in the `llarp` directory. To verify the download is correct, `ls data/hf_llama_7B` should return `pytorch_model-00001-of-00001.bin`, `config.json`, etc.

### Commands
**Training Commands**:
- Train LLaRP on Language Rearrangement task: `python llarp/run.py --config-name=baseline/llarp.yaml`
- The trainer is built on Habitat-Baselines v0.3.0. Most Habitat-Baselines options also apply to LLaRP training. See [this README](https://github.com/facebookresearch/habitat-lab/tree/afe4058a7f8aa5ab71a133575cdaa79f0308af6a/habitat-baselines) for more information about how to use Habitat Baselines.
- To run training on multiple GPUs, use [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html).
- More configurable options are under `llarp/config/baseline/policy/llarp_policy.yaml`.

**Evaluation Commands**: First, get the checkpoint generated by the training command. Then run: `python llarp/run.py --config-name=baseline/llarp.yaml habitat_baselines.evaluate=True habitat_baselines.rl.policy.main_agent.hierarchical_policy.high_level_policy.is_eval_mode=True habitat_baselines.eval_ckpt_path_dir=path/to/checkpoint.pth habitat.dataset.data_path=datasets/DATASET.pickle` where `DATASET` refers to one of the Language Rearrangement evaluation splits under `data/datasets`: `train`, `new_scenes`, `rephrasing`, `referring_expressions`, `spatial_relationships`, `context`, `irrelevant_text`, `multiple_rearrangements`, `novel_objects`, `multiple_objects`, or `conditional_instructions`.

**Dataset Generation Commands**: Since the episodes are procedurally generated, it is possible to generate more episodes beyond the 150,000 in the training dataset.
- Generate dataset: `python llarp/dataset/create_episodes.py --run --config rl_llm/dataset/configs/dataset.yaml --num-episodes 100 --out data/datasets/dataset_path.pickle --limit-scene-set scene_train_split --instruct-path llarp/dataset/configs/instructions.yaml --tag v2_train --verbose --n-procs 112 --seed 7 --procs-per-gpu 14`
- Validate generated episodes are solvable and remove unsolvable episodes: `python rl_llm/dataset/dataset_validator.py --cfg rl_llm/config/task/lang_cond.yaml --n-procs 56 habitat.dataset.data_path=/mnt/task_runtime/projects/rl_llm/data/datasets/dataset_path_validated.pickle task_obs=all`

**Other**:
- Run tests: `pytest test`. This checks that LLaRP weights are updated exactly as expected after several training iterations under a variety of training settings

## Documentation

Code Structure (under `llarp/` directory):
- `config/`: Hydra configuration YAML files.
    - `baseline/`: Config files for policies and trainers.
    - `task/`: Config files for the Language Rearrangement task.
- `dataset/`: Utilities for generating the Language Rearrangement dataset files.
    - `configs/`: Config files for Language Rearrangement dataset generation. In this directory, `instructions.yaml` contains the language instruction templates and `dataset.yaml` defines the possible objects and receptacles. Refer to Appendix Section B of the paper for details on the instruction templates.
    - `create_episodes.py` is the entry point for the episode generation.
    - `dataset_validator.py` processes an already created dataset and ensures all the episodes are solvable. It deletes unsolvable episodes.
- `policies/`: Defines the LLaRP policy module.
    - `cores/decoder.py`: This contains the bulk of the policy code for sampling from and updating the policy.
    - `action_decoders.py`: Contains the action decoder head.
    - `llm_policy.py`: Entry point for the LLaRP policy. This integrates the LLaRP policy with the Habitat Baselines trainer.
    - `transformer_storage.py`: Modified rollout buffer for transformers in PPO.
    - `vis_bridge.py`: The observation encoder module.
    - `visual_encoders.py`: The visual encoder (VC-1). 
- `task/`: Code to setup the Language Rearrangement task in Habitat-Lab.
- `trainer/`: Contains the core RL loop, PPO loss calculation, evaluation, environment creation, and distributed training utilities.
Language Rearrangement episode datasets are included in this repository under the `datasets/` directory.

## License
The code is provided under [Apple Sample Code license](https://github.pie.apple.com/aiml-oss/ml-aim/blob/main/LICENSE).

## Citation
```
@article{szot2023large,
  title={Large Language Models as Generalizable Policies for Embodied Tasks},
  author={Szot, Andrew and Schwarzer, Max and Agrawal, Harsh and Mazoure, Bogdan and Talbott, Walter and Metcalf, Katherine and Mackraz, Natalie and Hjelm, Devon and Toshev, Alexander},
  journal={arXiv preprint arXiv:2310.17722},
  year={2023}
}
```