import torch
from torch.utils.data.dataset import Dataset
from pathlib import Path
from tokenizers import Tokenizer, ByteLevelBPETokenizer
from tokenizers.models import BPE, Unigram
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer, UnigramTrainer
from model import GPT2

paths = [str(x) for x in Path("../Data/").glob("*.txt")]
VOCAB_SIZE = 52_000
model_dir = './UniLM_spaced'

class OwnDataset(Dataset):

    def __init__(self, tokenizer, file_path):

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = [tokenizer.encode(line) for line in lines]
        for e in self.examples:
            e.pad(length=128)
        self.examples = [e.ids for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


def create_tokenizer():
    tokenizer = Tokenizer(Unigram())
    trainer = UnigramTrainer(special_tokens=["<|endoftext|>"])
    tokenizer.enable_padding(pad_token="<|endoftext|>")
    tokenizer.train(paths, trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="<|endoftext|> $A <|endoftext|>",
        special_tokens=[
            ("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>")),
        ],
    )
    tokenizer.save(model_dir+"/vocab.json")
    en = tokenizer.encode("hwllewpf oianfokgew df")
    en.pad(length=128)


def training():
    from torch.utils.data import DataLoader
    from transformers import AdamW

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = GPT2()
    model.to(device)
    model.train()

    tokenizer = Tokenizer.from_file(model_dir + "/vocab.json")
    train_dataset = OwnDataset(tokenizer, "../Data/train.en.txt")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    n_batches = len(train_loader)
    optim = AdamW(model.parameters(), lr=5e-5)
    n_epochs = 1
    print("=== STARTING TRAINING ===")
    for epoch in range(n_epochs):
        for i, data in enumerate(train_loader, 0):
            optim.zero_grad()
            input_ids = data.to(device)
            outputs = model(input_ids, input_ids)
            loss = outputs[0]
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f}".format(
                    epoch + 1, int(100 * (i + 1) / n_batches),
                    loss.data.item()))

    torch.save(model, model_dir+"/model.pt")
    model.eval()
    print("=== FINISH TRAINING ===")

if __name__ == "__main__":
    create_tokenizer()
    training()