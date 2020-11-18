import os
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path(".").glob("Data/*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

if not os.path.isdir("./RoBertMLM"):
    os.mkdir("RoBertMLM")
tokenizer.save_model("RoBertMLM")