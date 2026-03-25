import time
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    BitsAndBytesConfig, 
    DataCollatorWithPadding
)
from torch.utils.data import DataLoader
from codecarbon import EmissionsTracker
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# =========================================================
# 1. VERİ HAZIRLIĞI
# =========================================================
dataset = load_dataset("tweet_eval", "sentiment")
test_ds = dataset["test"].filter(lambda x: x["label"] != 1)

def remap_labels(example):
    example["label"] = 1 if example["label"] == 2 else 0
    return example

test_ds = test_ds.map(remap_labels)
test_ds = test_ds.shuffle(seed=42).select(range(1000))

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_ds = test_ds.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# DataCollator 'label'ı 'labels' yapar, o yüzden hazırlıklı olmalıyız
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
dataloader = DataLoader(tokenized_ds, batch_size=32, collate_fn=data_collator)

# =========================================================
# 2. EVALUATION FONKSİYONU (HATA DÜZELTİLDİ)
# =========================================================
def run_experiment(model, name):
    print(f"\n>>> {name} Ölçülüyor...")
    model.eval()
    tracker = EmissionsTracker(log_level="error", measure_power_secs=1)
    tracker.start()
    
    preds, labels = [], []
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{name} Progress", leave=False):
            # Batch içindeki her şeyi GPU'ya taşı (labels dahil)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Model çıktılarını al
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            
            batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            preds.extend(batch_preds)
            
            # Hata Buradaydı: DataCollator sonrası anahtar 'labels' olabilir
            actual_labels = batch.get("labels", batch.get("label")).cpu().numpy()
            labels.extend(actual_labels)
            
    duration = time.time() - start_time
    emissions = tracker.stop()
    acc = accuracy_score(labels, preds)
    
    return {
        "acc": acc, "time": duration, "emissions": emissions,
        "green_score": (acc / emissions) if emissions > 0 else 0
    }

# =========================================================
# 3. MODELLERİN ÇALIŞTIRILMASI
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- A. NORMAL MODEL ---
model_normal = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
res_norm = run_experiment(model_normal, "Normal Model")

del model_normal
torch.cuda.empty_cache()

# --- B. 8-BIT MODEL ---
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model_quant = AutoModelForSequenceClassification.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto"
)
res_quant = run_experiment(model_quant, "8-bit Model")

# =========================================================
# 4. SONUÇ TABLOSU
# =========================================================
print("\n" + "="*70)
print(f"{'METRİK':<15} | {'NORMAL MODEL':<15} | {'8-BIT MODEL':<15} | {'DEĞİŞİM (%)'}")
print("-" * 70)
metrics = [
    ("Accuracy", res_norm['acc'], res_quant['acc']),
    ("Süre (sn)", res_norm['time'], res_quant['time']),
    ("CO2 (kg)", res_norm['emissions'], res_quant['emissions']),
    ("Green Score", res_norm['green_score'], res_quant['green_score'])
]
for label, n, q in metrics:
    diff = ((q - n) / n * 100) if n > 0 else 0
    print(f"{label:<15} | {n:<15.4f} | {q:<15.4f} | {diff:>+10.2f}%")
print("="*70)