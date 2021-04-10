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
            text = f.read()
        block_size = 128
        tokenized_text = tokenizer.encode(text).ids  # gives a list of ids
        self.examples = []
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
            self.examples.append(tokenized_text[i: i + block_size])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


def create_tokenizer():
    tokenizer = Tokenizer(Unigram())
    trainer = UnigramTrainer(special_tokens=["<|endoftext|>"])
    tokenizer.train(paths, trainer)
    tokenizer.save(model_dir+"/vocab.json")


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