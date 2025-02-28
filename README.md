# Can LVLMs and Automatic Metrics Capture Underlying Preferences of Blind and Low-Vision Individuals for Navigational Aid?

## 📜 News

🚀 [2025/2/15] The [arxiv paper](https://arxiv.org/abs/2502.14883) is released!


## 😀 Summary
- **BLV User Preferences on LVLMs** – This study explores Blind-and-Low-Vision (BLV) user preferences on different response styles from Large Vision-Language Models (LVLMs) for navigational aid.
- **Eye4B Dataset & Benchmark** – The Eye4B dataset includes 1.1k human-validated indoor/outdoor scenes with BLV-relevant requests, and an Eye4B benchmark evaluates how well existing metrics align with BLV preferences.
- **User Study & Key Evaluation Criteria** – An in-depth user study with eight BLV participants assesses six LVLMs based on Afraidness, Nonactionability, Sufficiency, and Conciseness, providing insights for developing BLV-aware AI systems.


## ``Requirements`` 

All the requirements are in [`environs/`](environs/).

| Environment name | Description |
| --- | --- |
| `brl` | training |
| `lric` | evaluation |
| `llava` | for [LLaVA](https://github.com/haotian-liu/LLaVA) model |
| `intern_clean` | for [InternLM](https://github.com/InternLM/InternLM) model |
| `polo` | for [Polaris](https://github.com/keio-smilab24/Polos) dataset |


## ``Data Structure`` 

```none
/projects/brl
├── mobility
│   ├── chosen_final
│   ├── ├── sideguide
│   ├── ├── sidewalk
│   ├── ├── outdoor
│   ├── ├── indoor
│   ├── results
│   ├── score_results
│   ├── irb
│   ├── ├── nov
│   ├── ├── dec
├── education
```


## ``Scenario Generation`` 

```
export OPENAI_API_KEY=[YOUR API KEY]
```
```
bash scripts/generate_scenario_[one_sample/pilot_samples/final_samples].sh
```
```
bash scripts/translate_korean_final_samples.sh
```


## ``Deep Context Generation`` 

7B models 

```
cd VL-ICL
python I2T_inference.py \
--query_dataset [query.json/mobility_pilot_study.json/mobility_pilot_study_extra.json] \
--engine [qwen-vl/openflamingo/llava16-7b/internlm-x2/otter-llama/qwen-vl-chat]
```

GPT-4o 

```
export OPENAI_API_KEY=[YOUR API KEY]
```
```
bash scripts/generate_deepcontexts_[one_sample/pilot_samples/final_samples].sh
```


## ``Deep Context Evaluation`` 

```
bash scripts/evaluate_[final_samples].sh
```

| Dataset | Context Dataset |
| --- | --- |
| `brl` | `*3/4-shot_mobility_pilot_study.json` |
| `polaris` | `polaris_test.csv` |
| `pascal50s` | `VOCdevkit/VOC2010` |
| `foil` | `foilv1.0_test_2017.json` |
| `flickr8k_expert` | `flickr8k.json` |
| `flickr8k_cf` | `crowdflower_flickr8k.json` |
| `filtered_oid` | `OID-rated-image-captions.v2.dev.alignment.tsv` |
| `filtered_polaris` | `yuwd` |
| `imgreward_test` | `ImageReward/data` |
| `brl_new` | `export*` |
| `brl_final` | `gp_overall/gp_avg` |


## ``BLIP-based Metric Training`` 

```
cd Github/ImageReward/train
bash scripts/train_one_node.sh
```


## ``Reward Model-based Metric Training`` 

Change configurations in [`recipes/samples/rm_bt.yaml`](recipes/samples/rm_bt.yaml).  
The accelerate configurations are in [`accelerate_config/ds3.yaml`](accelerate_config/ds3.yaml).  

```
python train_bt_pilot.py 
```
```
sh scripts/train_bt_pilot.sh
```

## ``References`` 

- [VL-ICL](https://github.com/ys-zong/VL-ICL)
- [ImageReward](https://github.com/THUDM/ImageReward.git)
- [Polos](https://github.com/keio-smilab24/Polos.git)
- [LongCLIP](https://github.com/beichenzbc/Long-CLIP.git)
