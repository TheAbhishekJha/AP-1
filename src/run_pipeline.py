import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pickle
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on parts of data.')
    parser.add_argument('--start', type=int, help='Starting index of data')
    parser.add_argument('--end', type=int, help='Ending index of data')
    args = parser.parse_args()

    start_idx = args.start - 1  # Convert to zero-based index
    end_idx = args.end

    df = pd.read_csv('data/phrases_v1.csv')
    actions = df['Phrase'].tolist()

    actions = actions[start_idx:end_idx]

    def return_prompt(i):
        prompt = (
            f"what are the possible motivations behind {actions[i]}. List all possible motivations in a json format, along with the probabilities of them being the actual cause. "
            "Make sure the formatting of response json is correct and free of any errors."
        )
        return prompt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device='cpu'
    # Load the tokenizer and model
    path = "/chronos_data/pretrained_models/llama2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)
    model.to(device)

    response_dict = {}
    for j in range(len(actions)):
        print('curr j:', start_idx + j)
        prompt = return_prompt(j)
        with torch.no_grad():
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
            response = model.generate(inputs, max_new_tokens=256, num_beams=5, top_k=10, temperature=0.001, eos_token_id=tokenizer.eos_token_id)
        decoded_response = tokenizer.decode(response[0], skip_special_tokens=True)
        del inputs, response
        torch.cuda.empty_cache()
        response_dict[j] = decoded_response
        if j%49==0:
            with open(f'data/response_dict_final_{start_idx}-{end_idx}.pickle', 'wb') as handle:
                pickle.dump(response_dict, handle)
        print('=====================')