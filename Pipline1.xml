import re
import pandas as pd
import numpy as np
from sklearn․model_selection import train_test_split
from sklearn․svm import SVC
from‌ sklearn․metrics import confusion_matrix
from sentence_transformers import‌ SentenceTransformer
import warnings
warnings․filterwarnings('ignore')

#‌ ============================================
# PIPELINE 1 ‚‌  HINDI + HINGLISH ONLY
No English data in training or evaluation․
# MiniLM + SVM + Rule Layer‌ + Five-Tier Scoring
#‌ ============================================

#‌ ============================================
Load and return the article․
#‌ ============================================
print("="*60)
print("STEP 1:‌ LOADING DATA")
print("="*60)

try:
df‌ = pd․read_excel('PromptInjectionPrompts․xlsx'‚ engine='openpyxl')
except FileNotFoundError:
df =‌ pd․read_excel('PromptInjectionPrompts․xls'‚ engine='xlrd')

print(f"Columns found:‌ {df․columns․tolist()}")

df =‌ df․rename(columns={
'Prompts': 'text'‚
'Label': 'label'‚
'Language': 'language'
})

required_cols = ['text'‚ 'label'‚ 'language']
missing = [c for‌ c in required_cols if c not in df․columns]
if missing:
raise ValueError(f"Missing columns after rename: {missing}")

df‌ = df[required_cols]

print(f"Total examples:‌ {len(df)}")
print(f"\nLanguage‌ distribution:")
print(df['language']․value_counts())
print(f"\nLabel distribution:")
print(df['label']․value_counts())

hindi_attacks = df[(df['language'] == 'Hindi') & (df['label']‌ ==‌ 1)]['text']
hindi_innocent = df[(df['language']‌ == 'Hindi') & (df['label']‌ == 0)]['text']
if not hindi_attacks․empty:
print(f"\nSample Hindi attack:\n{hindi_attacks․iloc[0]}")
if not hindi_innocent․empty:
print(f"\nSample Hindi‌ innocent:\n{hindi_innocent․iloc[0]}")

#‌ ============================================
# STEP 2:‌ SPLIT‌ DATA 80/20
#‌ ============================================
print("\n"‌ +‌ "="*60)
print("STEP 2: SPLITTING‌ DATA 80/20")
print("="*60)

hindi_df = df[df['language']‌ == 'Hindi']․copy()
hinglish_df‌ =‌ df[df['language'] == 'Hinglish']

print(f"Hindi‌ total:‌ {len(hindi_df)}")
print(f"Hinglish‌ total: {len(hinglish_df)}")

hi_train‚ hi_test = train_test_split(
hindi_df‚ test_size=0․2‚ random_state=42‚ stratify=hindi_df['label']
)
hg_train‚ hg_test = train_test_split(
hinglish_df‚ test_size=0․2‚‌ random_state=42‚ stratify=hinglish_df['label']
)

train_df‌ = pd․concat([hi_train‚ hg_train]‚‌ ignore_index=True)

print(f"\nTraining‌ set:‌ {len(train_df)}")
print(f"‌ Hindi train:‌ {len(hi_train)}")
print(f" Hinglish‌ train: {len(hg_train)}")
print(f"\nTest sets:")
print(f" Hindi‌ test: {len(hi_test)}")
print(f" Hinglish‌ test: {len(hg_test)}")

#‌ ============================================
In the calm‚ balanced English language․
#‌ ============================================
print("\n"‌ + "="*60)
print("LOADING DEEPSET‌ ENGLISH DATASET")
print("="*60)

from datasets import‌ load_dataset

try:
ds_deepset = load_dataset("deepset/prompt-injections")
df_deepset = ds_deepset['train']․to_pandas()
df_deepset_std = pd․ DataFrame({
'text': df_deepset['text']․astype(str)‚
'label': df_deepset['label']․astype(int)‚
'language': 'English'‚
'source': 'deepset'
})

if 'source' not in train_df․columns:
train_df['source'] = 'srinivasan'

train_df = pd․concat(
(train_df‚ df_deepset_std[['text'‚ 'label'‚ 'language'‚ 'source']]‚
ignore_index=True
)․reset_index(drop=True)

print(f" ✓ deepset added: {len(df_deepset_std)}‌ prompts")
print(f"‌ Total training: {len(train_df)} prompts")

except‌ Exception as e:
print(f" ✗ deepset failed:‌ {e}")
print(f" Continuing with Hindi+Hinglish only")

#‌ ============================================
# STEP 3: Loading sentence‌ transformer
#‌ ============================================
print("\n"‌ +‌ "="*60)
print("STEP 3:‌ LOADING‌ MULTILINGUAL MODEL")
print("="*60)

print("Loading‌ paraphrase-multilingual-MiniLM-L12-v2․․․")
embedder‌ =‌ SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Model loaded!")

#‌ ============================================
# STEP 4: CONVERT‌ THE TEXT‌ TO VECTORS
#‌ ============================================
print("\n"‌ +‌ "="*60)
print("STEP 4: CONVERTING‌ TEXT‌ TO MEANING VECTORS")
print("="*60)

print("Converting training‌ data․․․")
train_vectors = embedder․encode(
train_df['text']․tolist()‚‌ show_progress_bar=True‚ batch_size=32
)
print(f"Training vectors shape:‌ {train_vectors․shape}")

print("\nConverting Hindi test data․․․")
hi_test_vectors = embedder․encode(
hi_test['text']․tolist()‚ show_progress_bar=True‚ batch_size=32
)

print("\nConverting Hinglish test data․․․")
hg_test_vectors = embedder․encode(
hg_test['text']․tolist()‚‌ show_progress_bar=True‚ batch_size=32
)

#‌ ============================================
5․ Train‌ the SVM․
#‌ ============================================
print("\n"‌ + "="*60)
print("STEP 5: TRAINING SVM‌ CLASSIFIER")
print("="*60)

svm‌ = SVC(
kernel='rbf'‚
probability=True‚
class_weight='balanced'‚
random_state=42‚
C=1․0
)
svm․fit(train_vectors‚ train_df['label']․values)
print("SVM‌ training complete!")
print(f"Trained on:‌ {len(train_df)} prompts (Hindi + Hinglish only)")

#‌ ============================================
# STEP‌ 5B: RULE LAYER
After Srinivasan et al․ 2026
# Section 7․3
#‌ ============================================
print("\n"‌ +‌ "="*60)
print("STEP 5B:‌ RULE LAYER (Srinivasan et al․ 2026)")
print("="*60)

override_triggers = [
# Hindi‌ (Devanagari)
r"पिछले\s+निर्देश"‚
r"निर्देशों\s+को\s+भूल"‚
r"पिछले\s+सभी\s+निर्देश"
r"नए\s+निर्देश"‚
r"नया\s+कार्य"‚
r"ध्यान\s+न\s+दें"‚
r"अनदेखा\s+कर"‚
r"भूल\s+जाओ"‚
r"सिस्टम\s+प्रॉम्प्ट"‚
r"नई\s+भूमिका"‚
r"अब\s+से\s+तुम"‚
r"तुम्हारा\s+नया\s+काम"‚
r"पुराने\s+नियम"‚
r"नियमों\s+को\s+तोड़"‚
# Hinglish
r"pichle\s+nirdesh"‚
r"pichle\s+sabhi\s+nirdesh"‚
r"nirdesh\s+bhool"‚
r"ignore\s+instructions?"‚
r"forget\s+previous"‚
r"forget\s+earlier"‚
r"new\s+task\s+hai"‚
r"ab\s+se\s+tum"‚
r"rules\s+bhool"‚
r"system\s+prompt\s+batao"‚
r"pehle\s+wale\s+nirdesh"‚
r"ab\s+tumhara\s+kaam"‚
r"naye\s+nirdesh"‚
r"purane\s+rules"‚
]

harmful_keywords = [
# Hindi (Devanagari)
r"हैक"‚
r"गुप्त\s+डेटा"‚
r"पासवर्ड\s+बताओ"‚
r"सिस्टम\s+हैक"‚
r"डेटा\s+चोरी"‚
r"बम\s+बनाना"‚
r"विस्फोटक"‚
r"जहर\s+बनाना"‚
r"रिकिन"‚
r"हथियार\s+बनाना"‚
r"धोखाधड़ी\s+करना"‚
r"नुकसान\s+पहुंचाना"‚
r"मार\s+डालो"‚
r"आत्महत्या\s+कैसे"‚
r"लेविसाइट"‚
r"सरकारी\s+धोखाधड़ी"‚
r"क्रेडिट\s+कार्ड\s+जानकारी"‚
r"बैंक\s+खाता\s+हैक"‚
r"व्यक्तिगत\s+जानकारी\s+चुराना"‚
r"निजी\s+संदेश\s+सार्वजनिक"‚
r"आंतरिक\s+संरचना\s+साझा"
r"गुप्त\s+जानकारी\s+बताओ"
r"सुरक्षा\s+तोड़"‚
r"फ़ायरवॉल\s+बाईपास"‚
# Hinglish
r"hack\s+kar"‚
r"hack\s+karo"‚
r"hacking\s+sikhao"‚
r"password\s+batao"‚
r"password\s+do"‚
r"data\s+chori"‚
r"system\s+hack"‚
r"bomb\s+banana"‚
r"zeher\s+banana"‚
r"ricin\s+banana"‚
r"lewisite\s+banana"‚
r"weapon\s+banana"‚
r"firewall\s+bypass\s+karo"
r"security\s+bypass"‚
r"credentials\s+do"‚
r"credit\s+card\s+info"‚
r"bank\s+account\s+hack"‚
r"personal\s+info\s+nikalo"‚
r"private\s+data\s+nikalo"
r"SQL\s+query\s+do"‚
r"rate\s+limits?\s+ignore"‚
r"restrictions?\s+hatao"‚
r"jasoosi\s+karo"‚
r"dhokhadhadi"‚
]

benign_keywords = [
# Hindi (Devanagari) ‚  unambiguous innocent phrases only
r"निबंध"‚
r"कहानी\s+लिखो"‚
r"कविता"‚
r"अनुवाद"‚
r"सारांश"‚
r"समझाइए"‚
r"बताइए"‚
r"परिभाषा"‚
r"उदाहरण"‚
r"सुझाव\s+दें"‚
r"यात्रा\s+योजना"‚
r"खाना\s+बनाना"‚
r"स्वास्थ्य\s+सुझाव"‚
# Hinglish ‚  multi-word specific phrases only
# Generic single words like batao/banaiye removed
# as they appear inside attack prompts too
r"essay\s+likho"‚
r"kahani\s+likho"‚
r"explain\s+karo"‚
r"samjhao\s+mujhe"‚
r"coding\s+seekhna"‚
r"coding\s+sikhna"‚
r"programming\s+seekhna"‚
r"programming\s+bhaasha\s+seekhna"‚
r"hackathon"‚
r"life\s+hack"‚
r"movie\s+recommend\s+karo"
r"travel\s+plan\s+banao"‚
r"recipe\s+batao\s+mujhe"
r"health\s+tips\s+batao"‚
r"interview\s+tips\s+batao"‚
r"resume\s+banane\s+mein"‚
r"study\s+tips\s+batao"‚
r"maths\s+samjhao\s+mujhe"
r"science\s+samjhao"‚
r"history\s+batao\s+mujhe"‚
r"kya\s+hota\s+hai"‚
r"kaunsi\s+language\s+seekhni"
r"best\s+platform\s+coding"‚
r"pathyakram\s+kahan"‚
r"seekhte\s+samay\s+kaise"‚
r"naukri\s+ke\s+liye\s+tips"
]

override_patterns = [re․compile(p‚‌ re․ IGNORECASE‌ |‌ re․ UNICODE) for p‌ in override_triggers]
harmful_patterns =‌ [re․compile(p‚ re․ IGNORECASE‌ | re․ UNICODE) for‌ p in harmful_keywords]
benign_patterns‌ =‌ [re․compile(p‚ re․ IGNORECASE |‌ re․ UNICODE) for‌ p‌ in benign_keywords]

print(f"Override‌ triggers:‌ {len(override_triggers)}")
print(f"Harmful‌ keywords:‌ {len(harmful_keywords)}")
print(f"Benign‌ keywords:‌ {len(benign_keywords)}")
print(f"Total‌ patterns:‌ {len(override_triggers)+len(harmful_keywords)+len(benign_keywords)}")

def‌ rule_based_check(prompt:‌ str) -> str:
"""
Returns: Injection / Suspicious / SafeContext / NoRule
Logic based on the work of Srinivasan et al․ Algorithm 2․
"""
p_lower = prompt․lower()
harmful_found = any(pat․search(prompt) for pat in harmful_patterns)
override_found = any(pat․search(p_lower) for pat in override_patterns)
benign_found = any(pat․search(p_lower) for pat in benign_patterns)

if harmful_found and not benign_found: return "Injection"
if override_found and harmful_found: return "Injection"
if override_found and benign_found: return "SafeContext"
if override_found: return "Suspicious"
if benign_found and not harmful_found: return "SafeContext"
return "NoRule"

def hybrid_predict(prompt‚ vector):
"""
Rule layer‚ followed by SVM․
Returns: (final_label‚ confidence‚ decision_source)
"""
rule_result = rule_based_check(prompt)
if rule_result == "Injection": return 1‚ 1․0‚ "rule_injection"
if rule_result == "SafeContext": return 0‚ 0․0‚ "rule_safe"
confidence = svm․predict_proba([vector])[0][1]
prediction = 1 if confidence >= 0․5 else 0
return prediction‚ confidence‚ "svm"

print("\nRule layer ready․")
print("Priority: Injection > SafeContext > Suspicious > NoRule → SVM")

#‌ ============================================
#### SVM baseline evaluation
#‌ ============================================
print("\n" +‌ "="*60)
print("STEP 6: SVM BASELINE EVALUATION (no‌ rule layer)")
print("="*60)

def evaluate(name‚ vectors‚ true_labels):
"""Evaluate SVM only ‚  no rule layer․"""
predictions = svm․predict(vectors)
tn‚ fp‚ fn‚ tp = confusion_matrix(true_labels‚ predictions)․ravel()

accuracy‌ = (tp + tn) / (tp +‌ tn + fp + fn)
fpr = fp‌ / (fp‌ + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if‌ (fn + tp) > 0‌ else 0
detection_rate = tp‌ / (tp + fn) if (tp + fn)‌ > 0 else 0

print(f"\n{'='*40}")
print(f"SVM‌ ONLY‌ RESULTS: {name}")
print(f"{'='*40}")
print(f"Accuracy:‌ {accuracy:․2%}")
print(f"Detection Rate (TPR):‌ {detection_rate:․2%}")
print(f"False Positive‌ Rate: {fpr:․2%}")
print(f" (innocent users wrongly blocked)")
print(f"False‌ Negative Rate: {fnr:․2%}")
print(f" (attacks that slipped through)")
print(f"\nConfusion‌ Matrix:")
print(f" True Positives: {tp} (attacks‌ correctly caught)")
print(f" True Negatives: {tn} (innocent correctly allowed)")
print(f" False Positives: {fp} (innocent wrongly blocked)")
print(f" False Negatives: {fn} (attacks missed)")

return {
'language': name‚ 'accuracy': accuracy‚
'detection_rate': detection_rate‚
'fpr': fpr‚ 'fnr':‌ fnr‚
'tp': int(tp)‚‌ 'tn': int(tn)‚
'fp':‌ int(fp)‚ 'fn': int(fn)‚
'n_samples': len(true_labels)
}

hindi_results = evaluate("HINDI"‚ hi_test_vectors‚ hi_test['label']․values)
hinglish_results = evaluate("HINGLISH"‚ hg_test_vectors‚ hg_test['label']․values)

#‌ ============================================
# STEP 6B:‌ HYBRID‌ EVALUATION (rule layer + SVM)
#‌ ============================================
print("\n" +‌ "="*60)
print("STEP 6B: HYBRID EVALUATION (rule‌ layer + SVM)")
print("="*60)

def hybrid_evaluate(name‚ test_df‚ vectors):
"""Evaluate hybrid model ‚  rule layer first‚ SVM second․"""
test_df‌ =‌ test_df․copy()․reset_index(drop=True)
results = []
decisions = []

for i‚ row in test_df․iterrows():
pred‚ conf‚ source = hybrid_predict(row['text']‚ vectors[i])
results․append(pred)
decisions․append(source)

predictions = np․array(results)
true_labels = test_df['label']․values
tn‚ fp‚ fn‚ tp = confusion_matrix(true_labels‚ predictions)․ravel()

accuracy = (tp + tn)‌ / (tp +‌ tn + fp + fn)
fpr = fp / (fp + tn)‌ if (fp + tn) > 0 else‌ 0
fnr =‌ fn /‌ (fn + tp) if (fn + tp) > 0 else 0
detection_rate = tp / (tp + fn) if (tp + fn) >‌ 0 else‌ 0

rule_inject = decisions․count('rule_injection')
rule_safe = decisions․count('rule_safe')
svm_used = decisions․count('svm')

print(f"\n{'='*40}")
print(f"HYBRID RESULTS:‌ {name}")
print(f"{'='*40}")
print(f"Accuracy:‌ {accuracy:․2%}")
print(f"Detection Rate‌ (TPR): {detection_rate:․2%}")
print(f"False‌ Positive Rate: {fpr:․2%}")
print(f"False Negative‌ Rate: {fnr:․2%}")
print(f"\nConfusion‌ Matrix:")
print(f" True Positives:‌ {tp}")
print(f" True Negatives: {tn}")
print(f" False Positives: {fp}")
print(f" False Negatives: {fn}")
print(f"\nRule Layer Contribution:")
print(f" Blocked by rule: {rule_inject} ({rule_inject/len(test_df):․1%})")
print(f" Allowed by rule: {rule_safe} ({rule_safe/len(test_df):․1%})")
print(f" Deferred‌ to SVM: {svm_used} ({svm_used/len(test_df):․1%})")

return {
'language': name‚ 'accuracy': accuracy‚
'detection_rate': detection_rate‚
'fpr': fpr‚‌ 'fnr': fnr‚
'tp': int(tp)‚ 'tn':‌ int(tn)‚
'fp': int(fp)‚‌ 'fn': int(fn)‚
'rule_injection': rule_inject‚
'rule_safe': rule_safe‚
'svm_used': svm_used‚
}

hindi_hybrid = hybrid_evaluate("HINDI"‚‌ hi_test‚ hi_test_vectors)
hinglish_hybrid = hybrid_evaluate("HINGLISH"‚‌ hg_test‚‌ hg_test_vectors)

#‌ ============================================
# STEP‌ 7:‌ BEFORE AFTER COMPARISON
#‌ ============================================
print("\n" +‌ "="*60)
print("STEP 7: BEFORE VS‌ AFTER RULE LAYER")
print("="*60)

print(f"\n{'Metric':<30}{'SVM‌ Only':>10}{'Hybrid':>10}{'Change':>10}")
print("-"*62)

for metric‚ before‚ after in [
("Hindi Accuracy"‚ hindi_results['accuracy']‚ hindi_hybrid['accuracy'])‚
("Hindi FPR"‚ hindi_results['fpr']‚ hindi_hybrid['fpr'])‚
("Hindi FNR"‚ hindi_results['fnr']‚ hindi_hybrid['fnr'])‚
("Hinglish Accuracy"‚ hinglish_results['accuracy']‚ hinglish_hybrid['accuracy'])‚
("Hinglish FPR"‚ hinglish_results['fpr']‚ hinglish_hybrid['fpr'])‚
("Hinglish FNR"‚ hinglish_results['fnr']‚ hinglish_hybrid['fnr'])‚
]:
change = after - before
direction‌ = "↑" if change > 0 else "↓"
print(f"{metric:<30}{before:>10․2%}{after:>10․2%}‌ {direction}{abs(change):․2%}")

gap_before‌ = abs(hindi_results['fpr'] - hinglish_results['fpr'])
gap_after = abs(hindi_hybrid['fpr'] - hinglish_hybrid['fpr'])

print(f"\nFAIRNESS GAP (Hindi FPR vs Hinglish FPR):")
print(f" Before rule layer: {gap_before:․2%}")
print(f" After rule layer: {gap_after:․2%}")
improvement = gap_before - gap_after
if improvement > 0:
print(f" Improvement: {improvement:․2%} ← rule layer reduced fairness gap")
else:
print(f"‌ Change: {improvement:․2%}")

total = hindi_results['n_samples'] +‌ hinglish_results['n_samples']
svm_acc = (
hindi_results['accuracy'] * hindi_results['n_samples']‌ +
hinglish_results['accuracy']‌ * hinglish_results['n_samples']
) / total
hyb_acc = (
hindi_hybrid['accuracy']‌ * hindi_results['n_samples'] +
hinglish_hybrid['accuracy'] *‌ hinglish_results['n_samples']
) / total

print(f"\nCOMBINED ACCURACY (Hindi + Hinglish):")
print(f" SVM only: {svm_acc:․2%}")
print(f" Hybrid: {hyb_acc:․2%}")
print(f"\nCOMPARISON WITH SRINIVASAN ET AL․ (2026):")
print(f" Their accuracy (hybrid): 99․70%")
print(f" Your SVM only: {svm_acc:․2%} (gap: {abs(99․70 - svm_acc*100):․2f}%)")
print(f" Your hybrid: {hyb_acc:․2%} (gap: {abs(99․70 - hyb_acc*100):․2f}%)")

results_df = pd․ DataFrame([hindi_results‚ hinglish_results])
results_df․to_csv('pipeline1_svm_results․csv'‚ index=False)
hybrid_df = pd․ DataFrame([hindi_hybrid‚ hinglish_hybrid])
hybrid_df․to_csv('pipeline1_hybrid_results․csv'‚ index=False)
print(f"\nSVM results saved to: pipeline1_svm_results․csv")
print(f"Hybrid results saved to: pipeline1_hybrid_results․csv")