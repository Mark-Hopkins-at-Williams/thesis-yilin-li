from pathlib import Path
from transformers import PreTrainedTokenizer
from tokenizers import Tokenizer, ByteLevelBPETokenizer
from tokenizers.models import BPE, Unigram
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer, UnigramTrainer

paths = [str(x) for x in Path(".").glob("Data/*.txt")]

def create_tokenizer():
    tokenizer = Tokenizer(Unigram())
    trainer = UnigramTrainer(special_tokens=["<|endoftext|>"])
    tokenizer.enable_padding()
    tokenizer.train(paths, trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="<|endoftext|> $A <|endoftext|>",
        special_tokens=[
            ("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>")),
        ],
    )
    tokenizer.save("BPE_spaced/vocab.json")

def byte_level_tokenizer():
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=paths,
                    vocab_size=1000,
                    min_frequency=2,
                    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>", "</w>"])
    tokenizer.save_model("BPE_spaced")
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("BPE_spaced", max_len=512)
    print(tokenizer("hello world"))

def load_tokenizer():
    tokenizer = PreTrainedTokenizer.from_pretrained("BPE_spaced/vocab.json")

    print(tokenizer("hello world"))

if __name__ == "__main__":
    create_tokenizer()