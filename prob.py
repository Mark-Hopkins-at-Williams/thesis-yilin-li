import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_dir = './BBPE_spaced'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
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

def show_output(text):
    input_ids = torch.tensor([tokenizer.encode(text)])
    print(tokenizer.tokenize(text))
    with torch.no_grad():
        output = model(input_ids)
        print(output[0][0].shape)

def perplexity():
    file_path = './Data/test2014.txt'
    from tqdm import tqdm
    with open(file_path, encoding="utf-8") as f:
        test = f.read()
    encodings = tokenizer(test, return_tensors='pt')
    max_length = model.config.n_positions
    stride = 128

    lls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone().to(device)
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl

print(perplexity())
#show_output("hello world")
#show_prob("the professor says that doing research is fun")
#show_prob("the prefassor say that doing research is fun")