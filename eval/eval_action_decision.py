import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLM():

    def __init__(self, model_id: str, peft_model_id: str = None):

        print(f'Loading base model: {model_id}')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        # Load PEFT model adapter if specified
        if peft_model_id:
            print(f'Loading PEFT model: {peft_model_id}')
            self.model = PeftModel.from_pretrained(
                self.model,
                peft_model_id,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def __call__(
        self,
        system_msg: str,
        user_msg: str,
        max_tokens: int = 8000,
        temp: float = 0.6,
    ):

        messages = [
            {
                'role': 'system',
                'content': system_msg
            },
            {
                "role": "user",
                "content": user_msg
            },
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            return_tensors="pt").to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temp,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        output = outputs[0]
        res = output[input_ids.shape[-1]:]
        res = self.tokenizer.decode(res, skip_special_tokens=True)

        return res


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id',
                        type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument('--peft_model_id', type=str, default=None)
    parser.add_argument('--dataset_pth',
                        type=str,
                        default="datasets/robot_instr_dataset_v0_3.json")
    parser.add_argument('--temp', type=float, default=0.6)
    return parser.parse_args()


if __name__ == '__main__':
    '''
    Args:
        model_id: Base model ID to use
            Ex: "meta-llama/Meta-Llama-3.1-8B-Instruct"
        peft_model_id: Directory containing finetuned adapters (e.g. directory
            with 'adapter_39.pt' files) created by torchtune
        dataset_pth: Path to dataset JSON file to evaluate on
    '''
    args = parse_args()

    llm = LLM(args.model_id, args.peft_model_id)

    with open(args.dataset_pth, "r") as f:
        samples = json.load(f)

    correct = 0
    num_samples = len(samples)

    for idx, sample in enumerate(samples):
        messages = sample["messages"]

        system_msg = messages[0]['content']
        user_msg = messages[1]['content']
        action = messages[2]['content']

        res = llm(system_msg, user_msg, temp=args.temp)

        if res == action:
            correct += 1

        print(f'{str(idx+1).zfill(2)} | Pred: {res} | Target: {action}')

    print(f"Accuracy: {correct / num_samples * 100 :.2f}%")
