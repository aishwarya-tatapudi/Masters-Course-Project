import pandas‌ as pd
import numpy as‌ np
import re
from sklearn․model_selection import train_test_split
from sklearn․svm import SVC
from sklearn․metrics import‌ confusion_matrix
from sentence_transformers‌ import SentenceTransformer
from datasets import load_dataset
import warnings
warnings․filterwarnings('ignore')

#‌ ============================================
Loading data code example: 
#‌ ============================================
print("="*60)
print("STEP‌ 1: LOADING DATA")
print("="*60)

df =‌ pd․read_excel('PromptInjectionPrompts․xlsx')

df = df․rename(columns={
'Prompts': 'text'‚
'Label': 'label'‚
'Language': 'language'
})

df = df[['text'‚ 'label'‚ 'language']]

print(f"Total‌ examples: {len(df)}")
print(df['language']․value_counts())

#‌ ============================================
# STEP‌ 2: SPLIT DATA
#‌ ============================================
hindi_df‌ = df[df['language'] == 'Hindi']
hinglish_df‌ = df[df['language']‌ == 'Hinglish']

hi_train‚‌ hi_test = train_test_split(
hindi_df‚ test_size=0․2‚ stratify=hindi_df['label']‚ random_state=42
)
hg_train‚ hg_test = train_test_split(
test_size=0․2‚ stratify=hinglish_df['label']‚ random_state=42
)

train_df‌ = pd․concat([hi_train‚ hg_train]‚ ignore_index=True)

#‌ ============================================
# STEP 3: ADD‌ DEEPSET
#‌ ============================================
print("\nAdding‌ deepset dataset․․․")

ds = load_dataset("deepset/prompt-injections")
df_deepset = ds['train']․to_pandas()

df_deepset_std = pd․ DataFrame({
'text': df_deepset['text']․astype(str)‚
'label': df_deepset['label']․astype(int)‚
'language':‌ 'English'
})

train_df = pd․concat([train_df‚ df_deepset_std]‚‌ ignore_index=True)

print(f"Training size:‌ {len(train_df)}")
print(train_df['language']․value_counts())

#‌ ============================================
#‌ STEP 4: EMBEDDINGS
#‌ ============================================
print("\nLoading‌ MiniLM․․․")
embedder‌ =‌ SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

train_vectors‌ = embedder․encode(train_df['text']․tolist()
hi_test_vectors = embedder․encode(hi_test['text']․tolist())
hg_test_vectors = embedder․encode(hg_test['text']․tolist())

#‌ ============================================
# STEP‌ 5: TRAIN SVM
#‌ ============================================
print("\nTraining‌ SVM․․․")
svm‌ = SVC(kernel='rbf'‚‌ probability=True‚‌ class_weight='balanced')
svm․fit(train_vectors‚ train_df['label'])

#‌ ============================================
# STEP 6A: SVM‌ BASELINE
#‌ ============================================
def evaluate_svm(name‚ vectors‚ labels):
preds = svm․predict(vectors)
tn‚ fp‚ fn‚ tp = confusion_matrix(labels‚ preds)․ravel()

acc = (tp + tn)/(tp + tn + fp‌ + fn)
fpr = fp/(fp‌ + tn)
fnr = fn/(fn+tp)

print(f"\n{name}‌ (SVM ONLY)")
print(f"Accuracy:‌ {acc:․2%}")
print(f"FPR:‌ {fpr:․2%}")
print(f"FNR: {fnr:․2%}")

return acc‚ fpr‚ fnr

svm_hi_acc‚ svm_hi_fpr‚ svm_hi_fnr = evaluate_svm("HINDI"‚ hi_test_vectors‚ hi_test['label'])
svm_hg_acc‚ svm_hg_fpr‚ svm_hg_fnr = evaluate_svm("HINGLISH"‚ hg_test_vectors‚ hg_test['label'])

#‌ ============================================
# STEP 6B:‌ RULE LAYER
#‌ ============================================
def rule_based_check(prompt):
if re․search(r"hack|bomb|steal|attack"‚ prompt‚ re․ IGNORECASE):
return "Injection"
if re․search(r"essay|explain|story|help"‚ prompt‚ re․ IGNORECASE):
return "SafeContext"
return "NoRule"

def hybrid_predict(prompt‚ vector):
rule = rule_based_check(prompt)
if rule == "Injection":
return 1‚ 1․0
if rule == "SafeContext":
return 0‚ 0․0
conf = svm․predict_proba([vector])[0][1]
return (1 if conf >= 0․5 else 0)‚ conf

#‌ ============================================
# STEP 6C:‌ HYBRID EVALUATION
#‌ ============================================
def evaluate_hybrid(name‚ test_df‚ vectors):
test_df‌ = test_df․reset_index(drop=True)

preds = []
for i‚ row in test_df․iterrows():
pred‚ _ = hybrid_predict(row['text']‚ vectors[i])
preds․append(pred)

tn‚ fp‚ fn‚ tp = confusion_matrix(test_df['label']‚ preds)․ravel()

acc‌ = (tp+tn)/(tp+tn+fp+fn)
fpr‌ = fp/(fp+tn)
fnr = fn/(fn+tp)

print(f"\n{name}‌ (HYBRID)")
print(f"Accuracy:‌ {acc:․2%}")
print(f"FPR: {fpr:․2%}")
print(f"FNR: {fnr:․2%}")

return acc‚ fpr‚ fnr

hi_acc‚ hi_fpr‚ hi_fnr = evaluate_hybrid("HINDI"‚ hi_test‚ hi_test_vectors)
hg_acc‚ hg_fpr‚ hg_fnr‌ = evaluate_hybrid("HINGLISH"‚ hg_test‚ hg_test_vectors)

#‌ ============================================
# STEP‌ 6D: SVM vs HYBRID COMPARISON
#‌ ============================================
print("\nSVM vs HYBRID‌ COMPARISON")

print(f"{'Metric':<25}{'SVM':>10}{'Hybrid':>10}")
print("-"*45)

print(f"{'Hindi‌ Accuracy':<25}{svm_hi_acc:>10․2%}{hi_acc:>10․2%}")
print(f"{'Hindi‌ FPR':<25}{svm_hi_fpr:>10․2%}{hi_fpr:>10․2%}")
print(f"{'Hinglish‌ Accuracy':<25}{svm_hg_acc:>10․2%}{hg_acc:>10․2%}")
print(f"{'Hinglish‌ FPR':<25}{svm_hg_fpr:>10․2%}{hg_fpr:>10․2%}")

#‌ ============================================
# STEP 7: FIVE-TIER +‌ CSR
#‌ ============================================
def assign_tier(conf):
if conf >= 0․90: return "CRITICAL"‚ "AUTO BLOCK"
elif conf >= 0․70: return "HIGH"‚ "CSR SAME DAY"
elif conf >= 0․50: return "MEDIUM"‚ "CSR NEXT DAY"
elif conf >= 0․30: return "LOW"‚ "MONITOR"
else: return "SAFE"‚ "AUTO ALLOW"

def tier_distribution(name‚ vectors):
confs = svm․predict_proba(vectors)[:‚1]

tiers = []
actions = []

for c in confs:
t‚a = assign_tier(c)
tiers․append(t)
actions․append(a)

tiers = np․array(tiers)

print(f"\n{name} Tier Distribution:")
for t in ["CRITICAL"‚"HIGH"‚"MEDIUM"‚"LOW"‚"SAFE"]:
count = np․sum(tiers == t)
print(f"{t}: {count} ({count/len(tiers):․1%})")

auto = np․sum((tiers=="CRITICAL") | (tiers=="SAFE"))
csr‌ = np․sum((tiers=="HIGH") | (tiers=="MEDIUM"))

print(f"\nAuto‌ handled: {auto} ({auto/len(tiers):․1%})")
print(f"Needs CSR: {csr} ({csr/len(tiers):․1%})")

tier_distribution("HINDI"‚ hi_test_vectors)
tier_distribution("HINGLISH"‚ hg_test_vectors)

#‌ ============================================
#‌ STEP‌ 8: FAIRNESS‌ GAP
#‌ ============================================
print("\nFAIRNESS GAP")
print(f"FPR gap: {abs(hi_fpr - hg_fpr):․2%}")
print(f"FNR gap: {abs(hi_fnr - hg_fnr):․2%}")