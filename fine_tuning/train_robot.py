import json
from gliner import GLiNER
from gliner.training import TrainingArguments, Trainer
from gliner.data_processing.collator import SpanDataCollator  # ✅ Fixed import

print("--- Step 1: Loading Dataset ---")
with open("train_data.json", "r") as f:
    train_dataset = json.load(f)
print(f"Successfully loaded {len(train_dataset)} training sentences!")

print("--- Step 2: Loading Base Model ---")
model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

print("--- Step 3: Setting up the Matrix ---")
training_args = TrainingArguments(
    output_dir="custom_models/robot_brain_checkpoints",
    learning_rate=1e-5,
    weight_decay=0.01,
    others_lr=1e-5,
    lr_scheduler_type="linear",
    warmup_steps=20,
    max_steps=200,
    per_device_train_batch_size=4,
    save_steps=50,
    logging_steps=10,
)

data_collator = SpanDataCollator(  
    model.config,
    data_processor=model.data_processor,
    prepare_labels=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,  
)

print("--- Step 4: BEGIN TRAINING ---")
trainer.train()

print("--- Step 5: Saving Final Masterpiece ---")
model.save_pretrained("custom_models/robot_brain_small_FINAL")
print("SUCCESS! Your new AI is saved in the 'custom_models' folder.")