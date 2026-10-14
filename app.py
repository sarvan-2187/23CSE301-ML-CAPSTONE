import os
os.environ["USE_TF"] = "0"; os.environ["TRANSFORMERS_NO_TF"] = "1"

import html
import joblib
import json
import pandas as pd
import streamlit as st

ART = "phishing_artifacts"
CFG = {"max_len": 128, "n_heads": 8, "cnn_filters": 128, "cnn_kernel": 5, "gru_units": 64, "dropout": 0.3}
st.set_page_config(page_title="Phishing Email Detector", page_icon=None, layout="wide")


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

        def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
            x = self.bert(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
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


@st.cache_resource
def load_heads():
    """B, B-ablation and D: small heads on top of the frozen pretrained BERT pooled output."""
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    class PaperLiteralModel(nn.Module):  # same as training notebook
        def __init__(self, gru_units):
            super().__init__()
            self.bn = nn.BatchNorm1d(768)
            self.mha = nn.MultiheadAttention(768, CFG["n_heads"], batch_first=True)
            self.conv = nn.Conv1d(768, CFG["cnn_filters"], CFG["cnn_kernel"], padding="same")
            self.relu = nn.ReLU()
            self.gru = nn.GRU(CFG["cnn_filters"], gru_units, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(CFG["dropout"])
            self.fc1 = nn.Linear(2 * gru_units, 64)
            self.fc2 = nn.Linear(64, 1)

        def forward(self, pooled):
            x = self.bn(pooled).unsqueeze(1)
            x, _ = self.mha(x, x, x)
            x = self.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
            _, h = self.gru(x)
            h = self.drop(torch.cat([h[0], h[1]], dim=1))
            return self.fc2(self.relu(self.fc1(h))).squeeze(1)

    class PlainMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.BatchNorm1d(768), nn.Linear(768, 128), nn.ReLU(), nn.Dropout(CFG["dropout"]),
                                     nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, x):
            return self.net(x).squeeze(1)

    heads = {}
    for name, f in [("B. Paper-literal (frozen BERT)", "B_paper_literal_head"),
                    ("B-abl. Plain MLP", "B_ablation_mlp_head"),
                    ("D. Paper-literal + MGO head", "D_mgo_tuned_head")]:
        state = torch.load(f"{ART}/{f}.pt", map_location="cpu")
        m = PlainMLP() if f.startswith("B_ablation") else PaperLiteralModel(state["gru.weight_hh_l0"].shape[1])
        m.load_state_dict(state); m.eval()
        heads[name] = m
    return heads, AutoModel.from_pretrained("bert-base-uncased").eval()


def ig_explain(fwd, emb, tok, text, steps=32, batch=8):
    """Integrated Gradients over word embeddings (same method as the explainability notebook, Step 6).
    fwd(inputs_embeds, attention_mask) -> logits."""
    import torch
    enc = tok(text, truncation=True, padding="max_length", max_length=CFG["max_len"], return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]
    special = set(tok.all_special_ids)
    keep = torch.tensor([t in special for t in ids[0].tolist()])
    base = torch.where(keep, ids[0], torch.full_like(ids[0], tok.pad_token_id)).unsqueeze(0)
    with torch.no_grad():
        x, x0 = emb(ids), emb(base)
    grads = torch.zeros_like(x)
    for chunk in ((torch.arange(steps, dtype=torch.float32) + 0.5) / steps).split(batch):
        xs = (x0 + chunk.view(-1, 1, 1) * (x - x0)).requires_grad_(True)
        out = fwd(xs, mask.expand(len(chunk), -1))
        g, = torch.autograd.grad(out.sum(), xs)
        grads += g.sum(0, keepdim=True)
    attr = ((x - x0) * grads / steps).sum(-1)[0].detach().numpy()
    words, scores = [], []
    for t, a, i, m in zip(tok.convert_ids_to_tokens(ids[0].tolist()), attr, ids[0].tolist(), mask[0].tolist()):
        if m == 0 or i in special:
            continue
        if t.startswith("##") and words:  # merge word pieces
            words[-1] += t[2:]; scores[-1] += float(a)
        else:
            words.append(t); scores.append(float(a))
    return words, scores  # in email order


XAI = "xai_outputs"
RB = "robustness_outputs"
SHIELD = '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>'
ICON_BAD = SHIELD + '<path d="M12 8v4"/><path d="M12 16h.01"/>'
ICON_OK = SHIELD + '<path d="m9 12 2 2 4-4"/>'
BAD, OK = "#dc2626", "#059669"


def svg(inner, size=22, color="currentColor"):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round">{inner}</svg>')


def verdict_html(p, title=None, sub=None, big=None, frac=None, tick=0.5, scale=("0%", "50% threshold", "100%")):
    bad = p >= 0.5
    c = BAD if bad else OK
    frac = p if frac is None else frac
    return (f'<div class="verdict" style="--c:{c}">'
            f'<div class="vrow"><div class="vlabel">{svg(ICON_BAD if bad else ICON_OK, 22, c)}{title or ("Phishing" if bad else "Safe")}</div>'
            f'<div class="vright"><div class="vnum">{big or f"{p:.1%}"}</div>'
            f'<div class="vsub">{sub or "probability that this email is phishing"}</div></div></div>'
            f'<div class="meter"><div style="width:{frac * 100:.0f}%"></div><i style="left:{tick * 100:.0f}%"></i></div>'
            f'<div class="mscale"><span>{scale[0]}</span><span>{scale[1]}</span><span>{scale[2]}</span></div></div>')


def section(title, sub=""):
    st.markdown(f'<div class="sec"><h3>{title}</h3>{f"<p>{sub}</p>" if sub else ""}</div>', unsafe_allow_html=True)


CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500&display=swap');
:root{--ink:#0a0a0a;--mut:#6b6b76;--line:#e4e4e7;--brand:#0a0a0a;--r:8px}
html,body,.stApp,button,input,textarea,select,p,h1,h2,h3,h4,h5,h6,li,label,a,td,th,small,[data-testid=stMarkdownContainer],[data-testid=stMetricValue],[data-testid=stMetricLabel]{font-family:'Geist',system-ui,sans-serif!important}
code,pre,code *,pre *{font-family:'Geist Mono',ui-monospace,monospace!important}
.stApp{background:#fff}
.block-container{max-width:1440px;padding:3.5rem 2rem 4rem!important}
header[data-testid=stHeader],[data-testid=stToolbar],[data-testid=stDecoration]{display:none!important}
h1,h2,h3,h4{letter-spacing:-.02em;color:var(--ink)}
.hero{display:flex;align-items:center;gap:.9rem;padding:1.3rem 1.6rem;border-radius:var(--r);margin-bottom:1.4rem;color:#fff;background:#0a0a0a}
.hero svg{flex:none}
.hero h1{margin:0!important;padding:0!important;font-size:1.5rem;font-weight:600;color:#fff;letter-spacing:-.03em;line-height:1.2}
.hero p{margin:.15rem 0 0!important;opacity:.65;font-size:.92rem;line-height:1.4}
.stTabs [data-baseweb=tab-list]{gap:1.6rem;border-bottom:1px solid var(--line);padding:0;overflow-x:auto}
.stTabs [data-baseweb=tab]{height:2.6rem;padding:0;font-weight:500;color:var(--mut);background:none}
.stTabs [aria-selected=true]{color:var(--ink)!important}
.stTabs [data-baseweb=tab-highlight]{background:var(--ink)!important;height:2px}
.stTabs [data-baseweb=tab-border]{display:none}
.stTabs [data-baseweb=tab-panel]{padding-top:1.6rem}
.sec{margin:.2rem 0 .7rem}.sec h3{margin:0!important;padding:0!important;font-size:1.1rem;font-weight:600;line-height:1.3}
.sec p{margin:.2rem 0 0!important;color:var(--mut);font-size:.9rem}
.stTextArea textarea{border-radius:var(--r);border:1px solid var(--line);background:#fff;font-size:.95rem;padding:.9rem 1rem}
.stTextArea textarea:focus{border-color:var(--ink);box-shadow:none}
.stButton>button[kind=primary]{background:var(--ink);border:0;border-radius:var(--r);height:2.6rem;padding:0 1.3rem;font-weight:600}
.stButton>button[kind=primary]:hover{background:#262626}
div[role=radiogroup]{gap:.5rem;flex-wrap:wrap}
div[role=radiogroup]>label{background:#fff;border:1px solid var(--line);border-radius:var(--r);padding:.45rem .85rem}
div[role=radiogroup]>label:has(input:checked){border-color:var(--ink);box-shadow:inset 0 0 0 1px var(--ink)}
div[role=radiogroup]>label>div:first-child{display:none}
[data-testid=stMetric]{background:#fff;border:1px solid var(--line);padding:.9rem 1.1rem;border-radius:var(--r)}
[data-testid=stMetricLabel]{color:var(--mut)}
[data-testid=stDataFrame]{border:1px solid var(--line);border-radius:var(--r);overflow:hidden}
[data-testid=stImage] img{border-radius:var(--r);border:1px solid var(--line);background:#fff}
.stAlert{border-radius:var(--r)}
.verdict{margin:1.2rem 0 1.6rem;padding:1.2rem 0 .2rem;border-top:1px solid var(--line)}
.vrow{display:flex;align-items:flex-start;justify-content:space-between;gap:1rem}
.vlabel{display:flex;align-items:center;gap:.55rem;font-size:2rem;font-weight:700;letter-spacing:-.04em;color:var(--c);line-height:1.1}
.vlabel svg{align-self:center}
.vnum{font-size:2rem;font-weight:600;letter-spacing:-.04em;color:var(--ink);font-variant-numeric:tabular-nums}
.vright{text-align:right}
.vsub{color:var(--mut);font-size:.9rem;margin:.3rem 0 0}
.verdict>.meter{margin-top:1.1rem}
.meter{position:relative;height:6px;background:#eeeef0;border-radius:2px}
.meter>div{height:100%;background:var(--c);border-radius:2px}
.meter>i{position:absolute;top:-4px;width:2px;height:14px;background:var(--ink);margin-left:-1px}
.mscale{display:flex;justify-content:space-between;color:var(--mut);font-size:.75rem;margin-top:.5rem;font-variant-numeric:tabular-nums}
.card{background:#fff;border:1px solid var(--line);border-radius:var(--r);padding:1rem 1.1rem;height:100%}
.card.win{border-color:var(--ink);box-shadow:inset 0 0 0 1px var(--ink)}
.card .kind{color:var(--mut);font-size:.78rem;margin-bottom:.15rem;min-height:1.1rem}
.card h4{margin:0;font-size:.95rem;font-weight:600;min-height:2.5rem;line-height:1.3}
.card .pct{font-size:2.1rem;font-weight:700;letter-spacing:-.04em;margin:.5rem 0 .1rem;line-height:1;font-variant-numeric:tabular-nums}
.card small{color:var(--mut)}
.card .meter{margin:.9rem 0 .7rem;height:5px}
.card .status{display:flex;align-items:center;gap:.4rem;font-weight:600;font-size:.88rem}
.exp{line-height:2.1;background:#fff;border:1px solid var(--line);border-radius:var(--r);padding:.9rem 1.1rem}
.legend{display:flex;gap:1.2rem;color:var(--mut);font-size:.82rem;margin:.2rem 0 .6rem}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:.35rem}
</style>
<div class="hero">""" + svg(ICON_OK, 28, "#fff") + """<div><h1>Phishing Email Detector</h1><p>BERT + CNN + GRU with multi-head attention vs. four baselines, with word-level explanations</p></div></div>"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def results():
    r = pd.read_csv(f"{ART}/results_all_models.csv")
    r["short"] = r["model"].str.split(".").str[0].str.replace(" (reported)", "", regex=False)
    return r


R = results()
tab_det, tab_cou, tab_met, tab_gr, tab_xai, tab_rob = st.tabs(
    [":material/search: Detector", ":material/groups: Model Council", ":material/analytics: Metrics",
     ":material/bar_chart: Graphs", ":material/lightbulb: Explainability", ":material/shield: Robustness"])

with tab_det:
    MODELS = ["A. TF-IDF + LogReg", "B. Paper-literal (frozen BERT)", "B-abl. Plain MLP",
              "D. Paper-literal + MGO head", "C. Fine-tuned BERT (proposed)"]
    section("Analyze an email", "Pick a model, paste the email body, and see the verdict with word-level reasons.")
    choice = st.radio("Model", MODELS, horizontal=True, label_visibility="collapsed")
    text = st.text_area("Paste an email body", height=200, placeholder="Paste an email body here...", label_visibility="collapsed")
    if st.button("Analyze email", type="primary", icon=":material/search:") and text.strip():
        import torch
        if choice.startswith("A"):
            tfidf, clf = load_a()
            X = tfidf.transform([text])
            p = clf.predict_proba(X)[0, 1]
        elif choice.startswith("C"):
            with st.spinner("Loading BERT model (first run only)..."):
                model, tok = load_c()
            enc = tok(text, truncation=True, padding="max_length", max_length=CFG["max_len"], return_tensors="pt")
            with torch.no_grad():
                p = torch.sigmoid(model(enc["input_ids"], enc["attention_mask"]))[0].item()
        else:
            with st.spinner("Loading models (first run only)..."):
                heads, bert = load_heads()
                _, tok = load_c()
            enc = tok(text, truncation=True, padding="max_length", max_length=CFG["max_len"], return_tensors="pt")
            with torch.no_grad():
                p = torch.sigmoid(heads[choice](bert(**enc).pooler_output))[0].item()
        st.markdown(verdict_html(p), unsafe_allow_html=True)
        if choice.startswith("A"):
            contrib = X.multiply(clf.coef_[0]).tocoo()
            words = tfidf.get_feature_names_out()
            df = pd.DataFrame({"word": [words[j] for j in contrib.col], "contribution": contrib.data})
            section("Top words", "Positive = phishing, negative = safe.")
            top = pd.concat([df.nlargest(8, "contribution"), df.nsmallest(8, "contribution")])
            st.bar_chart(top.set_index("word"), horizontal=True)
        else:
            with st.spinner("Explaining with Integrated Gradients..."):
                if choice.startswith("C"):
                    words, scores = ig_explain(lambda x, m: model(inputs_embeds=x, attention_mask=m),
                                               model.bert.embeddings.word_embeddings, tok, text)
                else:
                    words, scores = ig_explain(lambda x, m: heads[choice](bert(inputs_embeds=x, attention_mask=m).pooler_output),
                                               bert.embeddings.word_embeddings, tok, text)
            s = pd.Series(scores, index=words).groupby(level=0).sum()
            s = s[s.index.str.contains(r"[A-Za-z0-9]")]  # drop pure-punctuation tokens
            top = pd.concat([s.nlargest(8), s.nsmallest(8)]).drop_duplicates()
            m = max(map(abs, scores)) or 1
            if max(map(abs, scores)) < 1e-6:
                st.warning("All attributions are zero: this head's output does not change with the input words here "
                           "(its ReLU layer is inactive for this email), so Integrated Gradients has nothing to attribute.")
            spans = "".join(
                f'<span title="{sc:+.3f}" style="background:rgba({"220,60,60" if sc > 0 else "60,110,220"},'
                f'{min(abs(sc) / m, 1):.2f});padding:2px 5px;margin:1px;border-radius:6px;display:inline-block">'
                f"{html.escape(w)}</span>"
                for w, sc in zip(words, scores)
            )
            section("Word by word", "Integrated Gradients per word. Hover a word for its exact score.")
            st.markdown('<div class="legend"><span><i style="background:#dc2626"></i>pushes toward phishing</span>'
                        '<span><i style="background:#3c6edc"></i>pushes toward safe</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="exp">{spans}</div>', unsafe_allow_html=True)
            section("Top words", "Positive = phishing, negative = safe.")
            st.bar_chart(top.rename("contribution"), horizontal=True)
            st.caption("Integrated Gradients on the first 128 tokens only (the rest is never seen by the model).")

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

with tab_rob:
    st.subheader("Do the models still work on a different dataset?")
    S = json.load(open(f"{RB}/robustness_summary.json"))
    st.caption(f"Trained on Phishing_Email.csv, tested without retraining on {S['new_dataset_file']} "
               f"({S['overlap']['kept']:,} emails, {S['overlap']['exact_overlap'] + S['overlap']['near_overlap']} overlapping with training removed).")
    RES = pd.read_csv(f"{RB}/robustness_results.csv")
    DROP = pd.read_csv(f"{RB}/robustness_drop.csv").set_index("model")
    best = DROP["acc_new"].idxmax()
    c = st.columns(3)
    c[0].metric("Most robust", best.split(".")[0], f"{DROP.loc[best, 'acc_new']:.1%} on new data")
    c[1].metric("Its accuracy drop", f"{DROP.loc[best, 'acc_drop_points']:.1f} pts")
    worst = DROP["acc_drop_points"].idxmax()
    c[2].metric("Biggest drop", worst.split(".")[0], f"-{DROP.loc[worst, 'acc_drop_points']:.1f} pts", delta_color="inverse")
    st.subheader("Accuracy: original test vs new dataset")
    st.image(f"{RB}/robustness_accuracy.png", width="stretch")
    st.subheader("All metrics")
    st.dataframe(RES.drop(columns=["acc_ci_low", "acc_ci_high"]).style.format(
        {k: "{:.2%}" for k in ["accuracy", "precision", "recall", "f1", "false_alarm_rate", "missed_phish_rate"]} | {"roc_auc": "{:.4f}"}),
        hide_index=True, width="stretch")
    st.subheader("Confusion matrices on the new dataset")
    st.image(f"{RB}/robustness_confusion.png", width="stretch")
    st.subheader("Which kinds of phishing are caught? (recall %)")
    bt = pd.read_csv(f"{RB}/robustness_by_type.csv")
    bt = bt[bt["column"] == "phishing_type"].set_index("value")
    st.dataframe(bt[[c for c in bt.columns if c.endswith("_caught_%")]].dropna(how="all")
                 .style.format("{:.1f}").background_gradient(cmap="RdYlGn", vmin=0, vmax=100), width="stretch")
    st.subheader("Is it ranking or just the threshold?")
    st.caption("A new cut-off is picked on 20% of the new data and scored on the other 80%. "
               "If accuracy recovers, the model still ranks well and only needs recalibration.")
    st.dataframe(pd.read_csv(f"{RB}/robustness_threshold.csv"), hide_index=True, width="stretch")
    st.subheader("Is Model C really better? (McNemar)")
    st.dataframe(pd.read_csv(f"{RB}/robustness_mcnemar.csv"), hide_index=True, width="stretch")
    st.subheader("How different is the new data?")
    st.dataframe(pd.read_csv(f"{RB}/length_shift.csv"), hide_index=True, width="stretch")
    st.info("New emails are short (median 35 words) and template-like, so almost nothing is cut at 128 tokens. "
            "Model C stays the best (about 90% accuracy, AUC 0.999), while the frozen-BERT heads B, B-abl. and D fall hardest.")

with tab_cou:
    section("Model council", "All 5 models read the same email and vote. Majority decides.")
    ctext = st.text_area("Paste an email body", height=200, key="council_text", placeholder="Paste an email body here...", label_visibility="collapsed")
    if st.button("Ask the council", type="primary", icon=":material/groups:") and ctext.strip():
        import torch
        tfidf, clf = load_a()
        with st.spinner("Loading models (first run only)..."):
            heads, bert = load_heads()
            model, tok = load_c()
        enc = tok(ctext, truncation=True, padding="max_length", max_length=CFG["max_len"], return_tensors="pt")
        votes = {"A. TF-IDF + LogReg": ("words (TF-IDF)", clf.predict_proba(tfidf.transform([ctext]))[0, 1])}
        with torch.no_grad():
            pooled = bert(**enc).pooler_output
            for n, h in heads.items():
                votes[n] = ("frozen BERT vector", torch.sigmoid(h(pooled))[0].item())
            votes["C. Fine-tuned BERT (proposed)"] = ("full fine-tuned BERT", torch.sigmoid(model(enc["input_ids"], enc["attention_mask"]))[0].item())
        n_phish = sum(p >= 0.5 for _, p in votes.values())
        verdict = "PHISHING" if n_phish >= 3 else "SAFE"
        st.markdown(verdict_html(1.0 if n_phish >= 3 else 0.0, f"Council verdict: {verdict}",
                                 f"{n_phish} of 5 models say phishing; 3 or more decide it",
                                 big=f"{n_phish} / 5", frac=n_phish / 5, tick=0.6, scale=("0", "majority at 3", "5")),
                    unsafe_allow_html=True)
        for col, (name, (inp, p)) in zip(st.columns(5, gap="medium"), votes.items()):
            bad = p >= 0.5
            color = BAD if bad else OK
            win = name.startswith("C.")
            col.markdown(
                f'<div class="card{" win" if win else ""}"><div class="kind">{"Proposed model" if win else "Baseline"}</div>'
                f'<h4>{html.escape(name)}</h4><div class="pct">{p:.1%}</div><small>phishing probability</small>'
                f'<div class="meter" style="--c:{color}"><div style="width:{p * 100:.0f}%"></div><i style="left:50%"></i></div>'
                f'<div class="status" style="color:{color}">{svg(ICON_BAD if bad else ICON_OK, 16, color)}{"Phishing" if bad else "Safe"}</div>'
                f'<p style="margin:.7rem 0 0"><small>Reads: {html.escape(inp)}</small></p></div>',
                unsafe_allow_html=True,
            )
        probs = [p for _, p in votes.values()]
        if max(probs) - min(probs) > 0.5:
            st.warning("The models disagree strongly. On new data the frozen-BERT heads (B, B-abl., D) often miss phishing "
                       "that A and C catch; see the Robustness tab. Use the Detector tab to see why a model decided.")
