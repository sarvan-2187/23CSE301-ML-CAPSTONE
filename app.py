import os
os.environ["USE_TF"] = "0"; os.environ["TRANSFORMERS_NO_TF"] = "1"

import joblib
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


st.title("🎣 Phishing Email Detector")
choice = st.radio(
    "Model",
    ["A. TF-IDF + LogReg (fast, explains words)", "C. Fine-tuned BERT (98.6% acc)"],
)
text = st.text_area("Paste an email body", height=220)

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
    (st.error if p >= 0.5 else st.success)(
        f"{'PHISHING' if p >= 0.5 else 'SAFE'}: phishing probability {p:.1%}"
    )
    if choice.startswith("A"):
        contrib = X.multiply(clf.coef_[0]).tocoo()
        words = tfidf.get_feature_names_out()
        df = pd.DataFrame({"word": [words[j] for j in contrib.col], "contribution": contrib.data})
        st.subheader("Why? (right = phishing, left = safe)")
        top = pd.concat([df.nlargest(8, "contribution"), df.nsmallest(8, "contribution")])
        st.bar_chart(top.set_index("word"))
    else:
        st.caption("Only the first 128 tokens are scored. Word-level explanation is available with Model A.")

with st.expander("Model comparison"):
    st.dataframe(pd.read_csv(f"{ART}/results_all_models.csv").iloc[:, :6])
