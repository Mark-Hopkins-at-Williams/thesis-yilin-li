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
        probs = torch.softmax(logits, 1)[-1]
        tops = torch.topk(probs, 5)
        print(tops)
        result = {}
        for i in range(5):
            token = tokenizer.id_to_token(tops.indices[i].item())
            prob = tops.values[i].item()
            result[i+1] = (token, prob)
        print(result)

def perplexity():
    file_path = '../Data/test2014.txt'
    from tqdm import tqdm
    import torch
    with open(file_path, encoding="utf-8") as f:
        test = f.read()
    encodings = tokenizer.encode(test)
    max_length = 128
    stride = 128

    lls = []
    for i in tqdm(range(0, len(encodings.ids), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, len(encodings.ids))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.ids[begin_loc:end_loc]
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
        target_ids = input_ids.clone().to(device)
        target_ids[:-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl

print(perplexity())
#show_next("Paris is the capital of")
#show_prob("the professor says that doing research is fun")
#show_prob("the prefassor say that doing research is fun")