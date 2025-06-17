# Text-Generation-with-GPT-2
pip install transformers datasets accelerate
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Important: GPT-2 does not have a pad token by default
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))
from datasets import load_dataset

# Load your dataset from text
dataset = load_dataset('text', data_files={'train': 'your_custom_data.txt'})
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    per_device_train_batch_size=2,
    evaluation_strategy="no",
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    warmup_steps=50,
    weight_decay=0.01,
    save_total_limit=1,
    fp16=True  # if you're using a GPU that supports it
)
from transformers import Trainer, DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=Falsefrom transformers import pipeline

text_generator = pipeline("text-generation", model="fine-tuned-gpt2", tokenizer=tokenizer)

prompt = "Once upon a time"
generated = text_generator(prompt, max_length=100, num_return_sequences=1)

print(generated[0]["generated_text"])

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)
trainer.train()
trainer.save_model("fine-tuned-gpt2")
tokenizer.save_pretrained("fine-tuned-gpt2")
