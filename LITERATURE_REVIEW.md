# Literature Review Summary Tables
### Supporting reference for Base Paper 045 — "Improving Phishing Email Detection Performance through Deep Learning with Adaptive Optimization"
Prepared by Nagarampalli Sarvan Kumar — CH.SC.U4CSE24130

---

## Table 1: Application Papers (Best Model → Worst Model)

| S.No | Paper Title | Year | Novelty / What is Proposed | Summary | Key Finding |
|---|---|---|---|---|---|
| 1 | Phishing Email Detection Using Improved RCNN Model with Multilevel Vectors and Attention Mechanism (THEMIS) | 2019 | Models email at 4 levels at once — header, body, character, word | RCNN + attention, trained on a realistic unbalanced dataset | **99.85% accuracy, 0.043% FPR** — best raw numbers in the whole set |
| 2 | HELPHED: Hybrid Ensemble Learning Phishing Email Detection | 2023 | Fuses content + textual traits into one hybrid feature set | Ensemble stacking + soft voting over multiple base classifiers | **F1 = 0.9942** on an imbalanced dataset |
| 3 | Novel Interpretable and Robust Web-Based AI Platform for Phishing Email Detection | 2024 | Adds Explainable AI (XAI) on top of a high-accuracy classifier | Real-time deployable web app, largest public dataset used | **F1 ≈ 0.99**, plus explainability for user trust |
| 4 | **⭐ OUR BASE PAPER — Improving Phishing Email Detection Performance through Deep Learning with Adaptive Optimization (BCG-MHeadAttention-MGO)** | **2025** | **BERT → Multi-Head Attention → CNN-1D → Bi-GRU, hyperparameters tuned by the Mountain Gazelle Optimizer (MGO)** | **Hybrid deep learning stack benchmarked against 7 other metaheuristics and 8 other deep learning architectures** | **Accuracy 97.22%, F1 96.43% (Table 9) — but the abstract separately states 96.8% accuracy / 96.3% F1, a numeric mismatch flagged in our Phase 2 review** |
| 5 | URLTran: Improving Phishing URL Detection Using Transformers | 2021 | First to fine-tune BERT/RoBERTa against adversarial URL attacks | Transformer model tested against homoglyph & compound-split attacks | **86.8% TPR at 0.01% FPR** (URL-level, not email body — not directly comparable) |
| 6 | Phishing Email Detection Using Robust NLP Techniques | 2018 | Uses simple features (word count, stopwords, punctuation) instead of complex ones | Weighted-SVM ensemble on noisy real-world email | **83% TP / 96% TN** |
| 7 | Phishing Email Detection Using Persuasion Cues | 2022 | Adds gain/loss psychological framing as explicit features | 3 ML models built around persuasion-cue features | **+5–20% F-score** over baseline (relative gain, no absolute number stated) |
| 8 | Attention-Based 1D CNN-BiLSTM Hybrid Model for Korean Voice Phishing Detection | 2023 | Applies the CNN+attention+BiLSTM pattern to voice/call transcripts | FastText embeddings + attention CNN-BiLSTM | Surpasses authors' own baselines (no single headline % reported) |
| 9 | Enhancing Phishing Email Detection through Ensemble Learning and Undersampling (FMPED/FMMPED) | 2023 | Two new undersampling algorithms for realistic class imbalance | Removes overlapping benign emails, then undersamples + ensembles | Outperforms ML/DL baselines on imbalanced data (no single headline % reported) |
| 10 | A Novel Hybrid Whale Optimization Algorithm with Flower Pollination Algorithm for Feature Selection | 2021 | Hybridizes WOA + FPA with opposition-based learning for feature selection | Applied to email spam as the case study | Outperforms comparable metaheuristics (no single headline % reported) |
| 11 | A Content-Based Phishing Email Detection Method | 2017 | Content-based detection method | Thinly indexed; only title/venue independently verifiable | No verifiable performance numbers found |

---

## Table 2: Survey / Review Papers

| S.No | Paper Title | Year | Novelty / What is Proposed | Summary | Key Finding |
|---|---|---|---|---|---|
| 1 | Defending Against Phishing Attacks: Taxonomy of Methods, Current Issues and Future Directions | 2018 | Builds a taxonomy classifying phishing defense methods | General framing of the attacker-vs-detector arms race | Establishes the categorization other papers build on |
| 2 | Phishing Email Detection Using NLP Techniques: A Literature Survey | 2021 | First survey focused specifically on NLP+ML for phishing | Reviews feature-extraction approaches (bag-of-words, topic models) | Flags surface-text dependence as the field's core weakness |
| 3 | A Systematic Literature Review on Phishing Email Detection Using NLP Techniques | 2022 | Expands to a full 100-paper systematic review (2006–2022) | Catalogs most-used features, algorithms, datasets | SVM + TF-IDF dominate; flags dataset scarcity/quality |
| 4 | Advances in Artificial Rabbits Optimization: A Comprehensive Review | 2024/25 | Reviews the ARO metaheuristic across variants/applications | Surveys 100+ ARO-based studies | General metaheuristic-landscape context, not phishing-specific |
| 5 | Recent Applications and Advances of African Vultures Optimization Algorithm | 2024 | Reviews the AVOA metaheuristic (one of the base paper's actual benchmarked competitors) | Analyzes 112 AVOA studies | Deeper background on a real competitor algorithm in the base paper's own results tables |

---

**Where our base paper stands:** It ranks **4th of the 11** application papers by headline F1/accuracy — behind THEMIS, HELPHED, and Al-Subaiey et al., ahead of everything else. It does **not** claim state-of-the-art on raw numbers alone; its own comparison tables only benchmark against 7 metaheuristics and 8 deep learning architectures it selected itself, not against THEMIS, HELPHED, or Al-Subaiey directly. Also worth flagging: the paper's own abstract (96.8% accuracy / 96.3% F1) and its own Table 9 (97.22% accuracy / 96.43% F1) don't fully agree with each other — a real, citable inconsistency, not a criticism you're inventing.

**Note:** Papers ranked 7–11 in Table 1 don't report a single comparable headline number found via independent search — ranking there is by strength of qualitative claim, not a shared metric. State this plainly if asked to justify the ranking in a review.