from transformers import (BertConfig,
                          BertTokenizer,
                          BertForMaskedLM,
                          LineByLineTextDataset,
                          DataCollatorForLanguageModeling,
                          Trainer, 
                          TrainingArguments )

config = BertConfig(
    vocab_size=52_000,
    max_position_embeddings=512,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

tokenizer = BertTokenizer.from_pretrained("./BertMLM", max_len=512)
model = BertForMaskedLM(config=config)

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./Data/Pride_and_Prejudice.txt",
    block_size=128,
)

val_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./Data/Mansfield_Park.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./BertMLM",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset = val_dataset,
    prediction_loss_only=True,
)

trainer.train()
trainer.save_model("./BertMLM")