from pathlib import Path
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer
from os.path import join

paths = [str(x) for x in Path("../Data/").glob("*.txt")]
VOCAB_SIZE = 52_000
model_dir = './train_en'


def create_tokenizer():
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=paths,
                    vocab_size=VOCAB_SIZE,
                    min_frequency=2,
                    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    tokenizer.save_model("train_en")


def validate_tokenizer():
    # Encoding(num_tokens=7, ...)
    # tokens: ['<s>', 'Mi', 'Ġestas', 'ĠJuli', 'en', '.', '</s>']
    from tokenizers.implementations import ByteLevelBPETokenizer
    from tokenizers.processors import BertProcessing
    tokenizer = ByteLevelBPETokenizer(
        join(model_dir, "vocab.json"),
        join(model_dir, "merges.txt"),
    )
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=512)


class EsperantoDataset(Dataset):
    def __init__(self, evaluate: bool = False):
        tokenizer = ByteLevelBPETokenizer(
            join(model_dir, "vocab.json"),
            join(model_dir, "merges.txt")
        )
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        tokenizer.enable_truncation(max_length=512)
        # or use the RobertaTokenizer from `transformers` directly.
        self.examples = []
        src_files = Path("./data/").glob("*-eval.txt") if evaluate else Path("./data/").glob("*-train.txt")
        for src_file in src_files:
            print("🔥", src_file)
            lines = src_file.read_text(encoding="utf-8").splitlines()
            self.examples += [x.ids for x in tokenizer.encode_batch(lines)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])


def init_trainer():
    from transformers import RobertaConfig
    from transformers import RobertaTokenizerFast
    config = RobertaConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )
    tokenizer = RobertaTokenizerFast.from_pretrained(model_dir, max_len=512)
    from transformers import RobertaForMaskedLM
    model = RobertaForMaskedLM(config=config)
    print('Num parameters: {}'.format(model.num_parameters()))
    from transformers import LineByLineTextDataset
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="../Data/train.en.txt",
        block_size=128,
    )
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    from transformers import Trainer, TrainingArguments
    training_args = TrainingArguments(
        output_dir=model_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_gpu_train_batch_size=64,
        save_steps=10_000,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        prediction_loss_only=True,
    )
    return trainer


def start_training(trainer):
    trainer.train()
    trainer.save_model(model_dir)


def validate_training():
    from transformers import pipeline
    fill_mask = pipeline(
        "fill-mask",
        model=model_dir,
        tokenizer=model_dir
    )
    print(fill_mask("Paris is the <mask> of France"))


def pipeline():
    create_tokenizer()
    trainer = init_trainer()
    start_training(trainer)
    validate_training()

pipeline()