import pickle, json, re, pandas as pd, torch
from glob import glob

def extract_information(text):
    results = re.findall(r'behind (.*?)\. List all possible motivations', text, re.DOTALL)
    
    # Iterate over each extracted sentence
    for result in results:
        # print("Text after 'behind':\n", result.strip())
        text_after_behind = result.strip()
        motives = ''

        # Find the list structure in JSON format
        list_start = text.find('[', text.find(result))
        list_end = text.find(']', list_start) + 1

        # Extract the list data from the located indices
        while list_start != -1 and list_end != -1:
            list_text = text[list_start:list_end]
            try:
                motivations_list = json.loads(list_text.lower().replace('"name":','"motivation":'))
                motives_dict = {"Motives": motivations_list}
                motives = json.dumps(motives_dict, indent=4)
                break
            except json.JSONDecodeError:
                motives = ''
                break
    return text_after_behind, motives

def get_embeddings(text, tokenizer, model):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Move model to the appropriate device
    model = model.to(device)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    last_hidden_states = outputs.hidden_states[-1]
    embeddings = last_hidden_states.mean(dim=1).squeeze()

    embeddings = embeddings.cpu().numpy()
    return embeddings

