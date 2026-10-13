import os
os.environ["USE_TF"] = "0"; os.environ["TRANSFORMERS_NO_TF"] = "1"

import joblib
import json
import pandas as pd
import streamlit as st

ART = "phishing_artifacts"
CFG = {"max_len": 128, "n_heads": 8, "cnn_filters": 128, "cnn_kernel": 5, "gru_units": 64, "dropout": 0.3}
st.set_page_config(page_title="Phishing Email Detector", page_icon="🎣")


@st.cache_resource
def load_a():
    m = joblib.load(f"{ART}/A_tfidf_logreg.joblib")
    return m["tfidf"], m["clf"]


@st.cache_resource
def load_c():
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, BertConfig, BertModel

    class BCGMHeadAttention(nn.Module):  # same as training notebook
        def __init__(self, cfg):
            super().__init__()
            self.bert = BertModel(BertConfig())
            H = self.bert.config.hidden_size
            self.bn = nn.BatchNorm1d(H)
            self.mha = nn.MultiheadAttention(H, cfg["n_heads"], batch_first=True, dropout=cfg["dropout"])
            self.lnorm = nn.LayerNorm(H)
            self.conv = nn.Conv1d(H, cfg["cnn_filters"], cfg["cnn_kernel"], padding="same")
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool1d(2)
            self.gru = nn.GRU(cfg["cnn_filters"], cfg["gru_units"], batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(cfg["dropout"])
            self.fc1 = nn.Linear(2 * cfg["gru_units"], 64)
            self.fc2 = nn.Linear(64, 1)

        def forward(self, input_ids, attention_mask):
            x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            x = self.bn(x.transpose(1, 2)).transpose(1, 2)
            a, _ = self.mha(x, x, x, key_padding_mask=(attention_mask == 0))
            x = self.lnorm(x + a)
            x = self.pool(self.relu(self.conv(x.transpose(1, 2)))).transpose(1, 2)
            _, h = self.gru(x)
            h = self.drop(torch.cat([h[0], h[1]], dim=1))
            return self.fc2(self.relu(self.fc1(h))).squeeze(1)

    model = BCGMHeadAttention(CFG)
    state = torch.load(f"{ART}/C_corrected_finetuned_full.pt", map_location="cpu")
    model.load_state_dict(state, strict=False)  # strict=False: only position_ids differ
    model.eval()
    return model, AutoTokenizer.from_pretrained(f"{ART}/tokenizer")


XAI = "xai_outputs"
st.markdown(
    """<style>
    .hero{padding:1.4rem 1.6rem;border-radius:16px;margin-bottom:1rem;color:#fff;
      background:linear-gradient(120deg,#0f2027,#203a43 45%,#2c5364)}
    .hero h1{margin:0;font-size:2rem}.hero p{margin:.3rem 0 0;opacity:.8}
    [data-testid=stMetric]{background:rgba(120,140,160,.12);padding:.8rem 1rem;border-radius:12px}
    </style>
    <div class="hero"><h1>🎣 Phishing Email Detector</h1>
    <p>BERT + CNN + GRU with multi-head attention vs. a TF-IDF baseline, with explainability</p></div>""",
    unsafe_allow_html=True,
)


@st.cache_data
def results():
    r = pd.read_csv(f"{ART}/results_all_models.csv")
    r["short"] = r["model"].str.split(".").str[0].str.replace(" (reported)", "", regex=False)
    return r


R = results()
tab_det, tab_met, tab_gr, tab_xai = st.tabs(["🔍 Detector", "📊 Metrics", "📈 Graphs", "🧠 Explainability"])

with tab_det:
    choice = st.radio(
        "Model",
        ["A. TF-IDF + LogReg (fast, explains words)", "C. Fine-tuned BERT (98.6% acc)"],
        horizontal=True,
    )
    text = st.text_area("Paste an email body", height=200)
    if st.button("Analyze", type="primary") and text.strip():
        if choice.startswith("A"):
            tfidf, clf = load_a()
            X = tfidf.transform([text])
            p = clf.predict_proba(X)[0, 1]
        else:
            import torch
            with st.spinner("Loading BERT model (first run only)..."):
                model, tok = load_c()
            enc = tok(text, truncation=True, padding="max_length", max_length=CFG["max_len"], return_tensors="pt")
            with torch.no_grad():
                p = torch.sigmoid(model(enc["input_ids"], enc["attention_mask"]))[0].item()
        (st.error if p >= 0.5 else st.success)(f"{'PHISHING' if p >= 0.5 else 'SAFE'}")
        st.progress(float(p), text=f"phishing probability {p:.1%}")
        if choice.startswith("A"):
            contrib = X.multiply(clf.coef_[0]).tocoo()
            words = tfidf.get_feature_names_out()
            df = pd.DataFrame({"word": [words[j] for j in contrib.col], "contribution": contrib.data})
            st.subheader("Why? (positive = phishing, negative = safe)")
            top = pd.concat([df.nlargest(8, "contribution"), df.nsmallest(8, "contribution")])
            st.bar_chart(top.set_index("word"), horizontal=True)
        else:
            st.caption("Only the first 128 tokens are scored. Word-level explanation is available with Model A.")

with tab_met:
    best = R.iloc[1:].sort_values("f1", ascending=False).iloc[0]
    paper = R.iloc[0]
    st.subheader(f"Best model: {best['model']}")
    c = st.columns(4)
    for col, m in zip(c, ["accuracy", "precision", "recall", "f1"]):
        col.metric(m.upper(), f"{best[m]:.2%}", f"{(best[m] - paper[m]) * 100:+.1f} pts vs paper")
    st.metric("ROC-AUC", f"{best['roc_auc']:.4f}")
    st.dataframe(
        R.drop(columns="short").set_index("model").style.format(
            {k: "{:.2%}" for k in ["accuracy", "precision", "recall", "f1"]} | {"roc_auc": "{:.4f}"}
        ).background_gradient(subset=["accuracy", "f1"], cmap="Greens"),
        width="stretch",
    )
    st.subheader("Significance tests (McNemar)")
    st.dataframe(pd.read_csv(f"{ART}/significance_tests.csv"), hide_index=True, width="stretch")

with tab_gr:
    st.subheader("Metric comparison")
    st.bar_chart(R.set_index("short")[["accuracy", "precision", "recall", "f1"]])
    st.subheader("Confusion matrices")
    cm = pd.read_csv(f"{ART}/confusion_matrices.csv", index_col=0)
    cm.loc["error rate"] = (cm.loc["FP"] + cm.loc["FN"]) / cm.sum()
    st.dataframe(cm.style.format("{:.0f}", subset=pd.IndexSlice[["TN", "FP", "FN", "TP"], :])
                 .format("{:.2%}", subset=pd.IndexSlice[["error rate"], :]).background_gradient(cmap="Blues", axis=1),
                 width="stretch")
    st.subheader("Training curves")
    H = json.load(open(f"{ART}/training_histories.json"))
    names = {"hist_b": "B paper-literal", "hist_abl": "B-ablation MLP", "hist_c": "C corrected", "hist_mgo": "D MGO-tuned"}
    pick = st.segmented_control("Metric", ["val_loss", "val_acc", "train_loss"], default="val_loss")
    curves = pd.DataFrame({names[k]: pd.Series(v[pick or "val_loss"]) for k, v in H.items()})
    curves.index = curves.index + 1
    st.line_chart(curves, x_label="epoch")
    st.subheader("MGO vs random search (hyperparameter tuning)")
    st.dataframe(pd.read_csv(f"{ART}/mgo_vs_random_summary.csv"), hide_index=True, width="stretch")
    for f in ["curves_overlay", "mgo_vs_random", "confusion_all"]:
        st.image(f"{ART}/plots/{f}.png", width="stretch")

with tab_xai:
    st.subheader("Strongest words overall (Model A)")
    g = pd.read_csv(f"{XAI}/global_A_top_words.csv")
    a, b = st.columns(2)
    a.caption("Pushes toward SAFE"); a.bar_chart(g.set_index("safe_word")["safe_weight"], horizontal=True, color="#4C72B0")
    b.caption("Pushes toward PHISHING"); b.bar_chart(g.set_index("phish_word")["phish_weight"], horizontal=True, color="#C44E52")
    st.info("Top safe words like 'enron' are dataset-source artefacts, not real safety signals. See the shortcut test below.")
    st.subheader("Deletion test: do highlighted words matter?")
    d = pd.read_csv(f"{XAI}/deletion_test.csv").groupby("model")[["drop_top5", "drop_random5"]].mean()
    st.bar_chart(d)
    st.subheader("Shortcut-word removal")
    st.dataframe(pd.read_csv(f"{XAI}/shortcut_removal_test.csv").style.format(
        {"accuracy": "{:.2%}", "f1": "{:.3f}", "acc_on_emails_with_shortcuts": "{:.2%}"}),
        hide_index=True, width="stretch")
    st.image(f"{XAI}/shortcut_curve.png", width="stretch")
