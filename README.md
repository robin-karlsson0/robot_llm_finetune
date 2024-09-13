# robot_llm_finetune
Robot LLM Finetuning Package


# Installation

Create and source virtualenv
```
pyenv virtualenv 3.10.12 robot_llm_finetune
source ~/.pyenv/versions/robot_llm_finetune/bin/activate
```

Install dependencies
```
pip install -r requirements.txt
```

Install `torchtune` via github
Ref: https://pytorch.org/torchtune/stable/install.html

```
git clone https://github.com/pytorch/torchtune.git
cd torchtune
pip install -e .
```

# Set up finetuning

### Download base model with `*.safetensors`

```
mkdir models
sh download_llama3_1.sh
```

Create symbolic link to model files
```
ln -s /PTH/TO/robot_llm_finetune/models/llama_3_1_8B_instruct /tmp/Meta-Llama-3.1-8B-Instruct
```

### Set up dataset

Locate JSON dataset file formatted as `chat_dataset` in `datasets/` dir
```
mkdir datasets
cp YOUR-DATASET.json datasets/
```

### Set up training config

Find config templates
```
tune ls
```

Copy config to file
```
tune cp llama3_1/8B_lora_single_device custom_config.yaml
```

Modfiy config file
```
tokenizer:
  max_seq_len: 8000

save_adapter_weights_only: True

dataset:
   _component_: torchtune.datasets.chat_dataset
   source: json
   data_files: /home/robin/projects/robot_llm_finetune/datasets/robot_instr_dataset_v0_3.json
   conversation_column: messages
   conversation_style: json
   train_on_input: False  # Default
   packed: False
   split: train
```

### Run training process

```
tune run lora_finetune_single_device --config custom_config.yaml --epochs=20
```

Provide _1) training recipie_, _2) training config_, and _3) overwrite parameters_

### Evaluate model

For Action Decision model
```
python eval/eval_action_decision.py \
    --peft_model_id models/llama_3_1_8B_instruct_ad_peft_v0_3_40ep \
    --dataset_pth datasets/robot_instr_dataset_v0_3_eval.json
```

Using LLM action server
```
python eval/eval_action_decision_ros.py \
    http://192.168.1.63:5001/llm \
    --dataset_pth datasets/robot_instr_dataset_v0_3_eval.json
```