Multilingual Prompt Injection Detection: A Fairness Audit in Hindi and Hinglish

Author: Aishwarya Anand Tatapudi
Course: M․S․ Computer Science Capstone (Plan III)
Affiliated with: University of Colorado Denver
Date: May 2026
Contact / full report: aishanand1999@gmail․com · [LinkedIn](https://www․linkedin․com/in/aishwarya-anand-tatapudi-96b530267/)
License: CC BY-NC-ND 4․0 (see LICENSE)
This project does two things:

1․ Re-implements a multilingual injection detector with a rule layer plus embedding model on a public Hindi/Hinglish dataset‚ and
2․ Audits that detector for fairness: it measures if the classifier has a higher false positive rate for Hindi than for code-mixed Hinglish inputs․

The fairness audit is the contribution․ The previous published work involved optimizing for aggregate detection accuracy and reporting a single number․ It did not turn the lens back on its own classifier to ask whether the two user groups are treated equally․ That is the question this project asks․



The finding

The false-positive rate (FPR) on the same MiniLM+SVM model is 2․50% for Hindi‚ and 15․00% for Hinglish‚ representing a gap of six-fold․ The FPR gap is about 12 percentage points: a Hinglish user writing an innocuous prompt is wrongfully blocked far more often than a Hindi user‚ for no reason other than the script they wrote in․

There are two additional results in the report that suggest this is a property of the task rather than a quirk of one model:

- All 22 Hinglish false positives were in NoRule‚ indicating the wider diversity is due to the differing embedding space positions of code-mixed texts rather than the rule layer's capacity to learn rules․
- Adding the rule layer improves accuracy by 0․25% but also marginally widens the equity gap (12․00% → 12․50%)․ More machinery leads to marginally better accuracy‚ but at the expense of equity․

This is not a problem of accuracy‚ with the aggregate Hindi/Hinglish detection accuracy being 93․12% and accuracy on the English (SPML) set being 98․92%․ The point of this project is what that high aggregate number conceals․



Why a deliberately minimal model

This pipeline has a frozen paraphrase-multilingual-MiniLM-L12-v2 encoder (it was not fine-tuned on the dataset)‚ a small rule layer‚ and a CPU-trained SVM․ Compared to the fine-tuned transformer plus large rule dictionary in the previous work‚ this is a weaker system that gets several points lower headline accuracy․

That's the tradeoff of the approach; a well-fit model can hide which groups it's not serving well․ In the bare-bones approach‚ we can see per-language classifier decisions‚ and the gap is there․ The perception is that seeing the gap in a bare-bones model is a stronger indicator that the gap is there than in a well-sharpened model․



Relationship to prior work

The Hindi/Hinglish dataset and the rule-layer-plus-transformer detection approach discussed in Pipeline 1 are as follows․

Srinivasan‚ J․‚ Regi‚ S․ A․‚ Anbarasan‚ A․ K․‚ Suresh‚ A․‚ Vetriselvi‚ T․‚ & Venu‚ S․ (2026)․ Detection and analysis of prompt injection in Indian multilingual large language models․ Scientific Reports‚ 16‚ 16208․ https://doi․org/10․1038/s41598-026-43883-0

The paper built a 4‚000-prompt Hindi/Hinglish dataset and a hybrid rule-based + transformer (XLM-RoBERTa) classifier achieving ~99․7% aggregate accuracy․ The detection architecture used here is in the same family as the one presented in that paper‚ and should be read as a re-implementation․ What is not in that paper - to the best of the author's reading of its results tables - is a per-language fairness breakdown of the detector's own false positives․ This project aspires to fill that gap․



Contribution vs․ prior work

| Component | Origin | Notes |
|---|---|---|
| Hindi + Hinglish dataset | Prior work (Srinivasan et al․ 2026‚ public on Kaggle) | Used as-is; not created here․ |
| Rule layer + model hybrid | Prior work | Also uses the same architectural principle‚ but uses MiniLM + SVM rather than XLM-RoBERTa and a much smaller rule set․
| English detection (Pipeline 2‚ SPML) | Standard baseline | MiniLM + SVM versus zero-shot NLI ‚  this is not a new method․
| Zero-shot NLI as a classifier | Existing technique | Used as a no-training comparison point․ |
| Per-language fairness audit (Hindi vs․ Hinglish FPR disparity) | This work | Original analysis: ~12-point FPR gap on the MiniLM+SVM pipeline․ |
| TF-IDF generalization-failure experiment | This project | Shows why a lexical baseline fails to generalize․



Project structure

Pipeline 1 ‚  Hindi + Hinglish (fairness audit)
- Loads the Hindi/Hinglish dataset
- Trains MiniLM + SVM‚ applies rule layer‚ outputs five-tier confidence score․
- Computes the per-language false-positive rates‚ and reports the difference in Hindi vs․ Hinglish‚ the core analysis․

Pipeline 2 ‚  English (baseline comparison)
- Loads the SPML dataset
- Experiment A: MiniLM + SVM (trained): 98․92%‚ 0․14% FPR
- Experiment B: zero-shot NLI (no training) showed an accuracy of 56․84%‚ an FNR of 86․31%․
- The 42 point accuracy gap suggests that reliable detection requires labeled training data to train on



Files

| File | Description |
|---|---|
pipeline1․py Hindi + Hinglish detection and fairness audit․
| pipeline2․py | English detection (MiniLM+SVM vs․ zero-shot NLI) |
| requirements․txt | Dependencies |
| LICENSE | CC BY-NC-ND 4․0 |

The capstone report (report․pdf) is available upon request (see contact above) and is not included within this repository․



Datasets required

1․ Hindi + Hinglish dataset‚ Srinivasan et al․ (2026)․ The dataset is also available at: sillaannregi/hindi-and-hinglish-prompts (Kaggle)․
2․ SPML Chatbot Prompt Injection dataset: pairs of prompts from the system and user․
3․ deepset/prompt-injections ‚  loaded automatically by Hugging Face (used as English training augmentation in Pipeline 1)․



How to run

``bash
1․ Install dependencies
pip install -r requirements․txt

2․ Run Pipeline 1 (Hindi + Hinglish + fairness audit)
python pipeline1․py

3․ Run Pipeline 2 (compare against English baseline)
python pipeline2․py
`



Scope and limitations

- The Hindi-vs-Hinglish gap was measured on the MiniLM + SVM pipeline‚ and the same should be validated using a stronger model (e․g․ a fine-tuned multi-lingual transformer)‚ an obvious next step but left for later work․ However‚ this could also be interpreted as "this detector treats the two groups unequally" rather than "this disparity is intrinsic to the task"․
- Training and evaluation are done on a single public dataset‚ no cross-dataset generalization is claimed․
- High aggregate accuracy on a clean‚ balanced dataset is expected and is not the point of this work; the fairness behavior is․



Future directions

- Check the model's "map․" This audit measures (1) how wide the gap is in who gets blocked and (2) whether it's earlier on in the process․ Is it in the encoder for code-mixed text before any decisions are made? If there is measurable per-language separability in the raw embeddings‚ then that means that this is something that is part of the encoder itself‚ and every system built on that encoder inherits it․
This relies on the same minimal pipeline as the previous model‚ which achieves this accuracy (~99․7%) with a fine-tuned XLM-RoBERTa and a large rule layer (Srinivasan et al․ (2026))․ Comparing a version without the rule layer‚ at a per-language FPR that remains constant‚ would likely help discern whether this discrepancy is only due to the pipeline or would also arise if aggregate accuracy did not conflate the two․
- Other Indian languages and their code-mixed varieties like Tamil‚ Telugu‚ Bengali‚ Malayalam etc․ should also be considered‚ any discrepancy in one of the code-mixed varieties is worth checking for․



Reproducibility notes

- All experiments were run on CPU․
- Embedding model: paraphrase-multilingual-MiniLM-L12-v2 (frozen‚ 384-dimensional)
- SVM: RBF kernel‚ C = 1․0‚ class_weight=balanced
- Train/test split: 80/20‚ random seed fixed at 42



Code and materials © 2026 Aishwarya Anand Tatapudi․ CC BY-NC-ND 4․0: Attribution required for non-commercial use․ No derivatives․ Commercial use and adaptations require written permission․
