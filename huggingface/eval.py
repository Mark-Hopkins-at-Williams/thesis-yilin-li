from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="esperberto",
    tokenizer="esperberto"
)

print(fill_mask("You drink the <mask>"))
print(fill_mask("You <mask> off"))