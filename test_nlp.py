from gliner import GLiNER

print("Waking up the custom robot brain...")

model_path = "fine_tuning/custom_models/small_model/robot_brain_small_FINAL"

model = GLiNER.from_pretrained(model_path, local_files_only=True)

text = "turn left on place, find Prof. Olov Andersson, and grab the green box on the right"
labels = ["action command", "person name", "object color", "physical object", "spatial direction"]

entities = model.predict_entities(text, labels, threshold=0.9)

print(f"\nOriginal Command: {text}")
print("--- Extracted Data ---")
for entity in entities:
    print(f"{entity['label'].upper()}: {entity['text']}")