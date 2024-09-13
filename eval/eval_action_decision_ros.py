import argparse
import json

import requests


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('flask_server_url', type=str)
    parser.add_argument('--dataset_pth',
                        type=str,
                        default="datasets/robot_instr_dataset_v0_3.json")
    return parser.parse_args()


if __name__ == '__main__':
    '''
    Args:
        flask_server_url: http://localhost:5000/llm
    '''
    args = parse_args()

    with open(args.dataset_pth, "r") as f:
        samples = json.load(f)

    correct = 0
    num_samples = len(samples)

    for idx, sample in enumerate(samples):
        messages = sample["messages"]

        system_msg = messages[0]['content']
        user_msg = messages[1]['content']
        action = messages[2]['content']

        data = {
            'system_prompt': system_msg,
            'prompt': user_msg,
            'max_tokens': 8000,
        }

        str_buffer = ''

        response = requests.post(args.flask_server_url, data=data, stream=True)

        for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
            if chunk:
                str_buffer += chunk

        if str_buffer == action:
            correct += 1

        print(f'{str(idx+1).zfill(2)} | Pred: {str_buffer} | Target: {action}')

    print(f"Accuracy: {correct / num_samples * 100 :.2f}%")
