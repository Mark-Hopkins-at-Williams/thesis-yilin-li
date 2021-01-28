import os
from pathlib import Path
from tokenizers import BertWordPieceTokenizer

paths = [str(x) for x in Path(".").glob("Data/*.txt")]

# Initialize a tokenizer
tokenizer = BertWordPieceTokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

if not os.path.isdir("./BertMLM"):
    os.mkdir("BertMLM")
tokenizer.save_model("BertMLM")