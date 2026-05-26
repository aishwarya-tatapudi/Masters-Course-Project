Prompt Injection Detection Project
=================================

Author: Aishwarya
Course: MSCS Course Project
University of Colorado Denver
Date: May 2026

------------------------------------------------------------
PROJECT OVERVIEW
------------------------------------------------------------

The project contains a dual-pipeline system for prompt
cross-language injection detection:

Pipeline 1:
- Hindi + Hinglish detection
- MiniLM + SVM + Rule Layer + Five-Tier Scoring
- Fairness across different languages

Pipeline 2:
- English detection using SPML dataset
- Experiment A: MiniLM + SVM (trained model)
- Experiment B: Zero-shot NLI (no training)

------------------------------------------------------------
FILES INCLUDED
------------------------------------------------------------

Pipeline1.py (Hindi + Hinglish)
- Loads dataset
- Trains MiniLM + SVM
- Applies rule layer
- Performs fairness analysis
- Generates five-tier scoring

Pipline2․py (English)
- Loads SPML dataset
- Trains MiniLM + SVM
- Runs a zero-shot NLI comparison․
- Outputs performance comparison

report․pdf
Final project report

------------------------------------------------------------
DATASETS REQUIRED
------------------------------------------------------------

1․ PromptInjectionPrompts․xlsx
- Hindi + Hinglish dataset (Srinivasan et al․ 2026)
A DataFrame with columns: 'Prompt'‚ 'Label'‚ 'Language'

2․ spml_prompt_injection․csv
- SPML Chatbot Prompt Injection dataset
- Columns:
- System Prompt
- User Prompt
- Prompt injection (label)

3․ deepset/prompt-injections
- As a result‚ it can be automatically loaded with HuggingFace

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------

Step 1: Install dependencies

pip install -r requirements․txt

Step 2: Run Pipeline 1 (Hindi + Hinglish)

python Pipline2․py

Step 3: Run Pipeline 2 (English)

python Pipline2․py

------------------------------------------------------------
OUTPUT FILES
------------------------------------------------------------

Pipeline 1:
- pipeline1_svm_results․csv
- pipeline1_hybrid_results․csv
- pipeline1_hindi_inspection․csv
- pipeline1_hinglish_inspection․csv

Pipeline 2:
- pipeline2_comparison․csv

------------------------------------------------------------
NOTES
------------------------------------------------------------

- All experiments were run on a CPU
- MiniLM model: paraphrase-multilingual-MiniLM-L12-v2
- SVM with RBF kernel‚ C=1․0
- Random seed fixed (42) for reproducibility

------------------------------------------------------------
Copyright (c) 2026 Aishwarya Anand Tatapudi. All rights reserved. No commercial or non-commercial use, reproduction, or distribution of this code and material is permitted without explicit written permission.
