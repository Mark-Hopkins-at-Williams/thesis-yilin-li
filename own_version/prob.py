import torch
from tokenizers import Tokenizer

model_dir = './UniLM_spaced'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = Tokenizer.from_file(model_dir + "/vocab.json")
model = torch.load(model_dir + "/model.pt").to(device)
model.eval()

def show_prob(text):
    input_ids = torch.tensor([tokenizer.encode(text)])

    with torch.no_grad():
        index = 0
        outputs = model(input_ids=input_ids)
        print(outputs)
        logits = outputs[0][0]
        probs = torch.softmax(logits, 1)
        for index in range(0, len(input_ids[0])):
            token_id = input_ids[0][index]
            probability = probs[index][token_id].item()
            print(f"Probability for the token \"{tokenizer.decode(token_id.item())}\" is {probability}")

def show_next(text):
    input_ids = torch.tensor(tokenizer.encode(text).ids).to(device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs[0]
        probs = torch.softmax(logits, 1)[0]
        tops = torch.topk(probs, 5)
        result = {}
        for i in range(5):
            token = tokenizer.id_to_token(tops.indices[i].item())
            prob = tops.values[i].item()
            result[i+1] = (token, prob)
        print(result)

show_next("Paris is the capital of ")



#show_prob("the professor says that doing research is fun")
#show_prob("the prefassor say that doing research is fun")