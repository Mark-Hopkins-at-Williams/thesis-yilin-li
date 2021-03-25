import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_dir = './BBPE_spaced'

tokenizer = GPT2Tokenizer.from_pretrained(model_dir, max_len=512)
model = GPT2LMHeadModel.from_pretrained(model_dir)
model.eval()

def show_prob(text):
    input_ids = torch.tensor([tokenizer.encode(text)])\

    with torch.no_grad():
        index = 0
        outputs = model(input_ids=input_ids)
        logits = outputs[0][0]
        probs = torch.softmax(logits, 1)
        for index in range(0, len(input_ids[0])):
            token_id = input_ids[0][index]
            probability = probs[index][token_id].item()
            print(f"Probability for the token \"{tokenizer.decode(token_id.item())}\" is {probability}")

show_prob("the professor says that doing research is fun")
show_prob("the prefassor say that doing research is fun")