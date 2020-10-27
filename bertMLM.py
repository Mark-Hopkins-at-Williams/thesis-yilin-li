import os 
from tokenizers import SentencePieceBPETokenizer, ByteLevelBPETokenizer
from transformers import (BertConfig, 
                         BertTokenizerFast, 
                         BertForMaskedLM, 
                         LineByLineTextDataset, 
                         DataCollatorForLanguageModeling, 
                         Trainer, 
                         TrainingArguments, 
                         pipeline)

paths = ["Data/Pride_and_Prejudice.txt", 
         "Data/Mansfield_Park.txt", 
         "Data/Sense_and_Sensibility.txt"] 

def build_tokenizer():
    tokenizer = ByteLevelBPETokenizer()

    # preparing datasets for train and evaluate

    tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    os.system("mkdir bpe_tokenizer")
    tokenizer.save_model("bpe_tokenizer")

def train_model():

    config = BertConfig(
        vocab_size=52_000,
        max_position_embeddings=512,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )

    model = BertForMaskedLM(config = config)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    #tokenizer = BertTokenizerFast.from_pretrained("./bpe_tokenizer", max_len=512)

    training_set = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=paths[0],
        block_size=128,
    )

    evaluate_set = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=paths[1],
        block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
    )

    training_args = TrainingArguments(
        output_dir="./bpe_MLM",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=64,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=training_set,
        eval_dataset=evaluate_set, 
        prediction_loss_only=True,
    )

    trainer.train()

    trainer.save_model("./bpe_MLM")

def evaluate():

    fill_mask = pipeline(
        "fill-mask",
        model="./bpe_MLM",
        tokenizer=BertTokenizerFast.from_pretrained('bert-base-uncased')
    )

    print(fill_mask("The rest of the [MASK] brought her little amusement."))
    '''
    [{'sequence': '[CLS] the rest of the, brought her little amusement. [SEP]', 'score': 0.0724494531750679, 'token': 1010, 'token_str': ','}, 
    {'sequence': '[CLS] the rest of the. brought her little amusement. [SEP]', 'score': 0.039655812084674835, 'token': 1012, 'token_str': '.'}, 
    {'sequence': '[CLS] the rest of the the brought her little amusement. [SEP]', 'score': 0.03040272928774357, 'token': 1996, 'token_str': 'the'}, 
    {'sequence': '[CLS] the rest of the of brought her little amusement. [SEP]', 'score': 0.024838656187057495, 'token': 1997, 'token_str': 'of'}, 
    {'sequence': '[CLS] the rest of the to brought her little amusement. [SEP]', 'score': 0.02378414012491703, 'token': 2000, 'token_str': 'to'}]
    '''

def main():
    '''
    if not os.path.isdir("bpe_tokenizer"):
        build_tokenizer()
    '''
    if not os.path.isdir("bpe_MLM"):
        train_model()
    evaluate()



if __name__ == "__main__":
    main()