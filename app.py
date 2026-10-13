import joblib
import pandas as pd
import streamlit as st

ART = "phishing_artifacts"
st.set_page_config(page_title="Phishing Email Detector", page_icon="🎣")


@st.cache_resource
def load():
    m = joblib.load(f"{ART}/A_tfidf_logreg.joblib")
    return m["tfidf"], m["clf"]


tfidf, clf = load()
st.title("🎣 Phishing Email Detector")
st.caption("TF-IDF + Logistic Regression (test acc 98.3%, ROC-AUC 0.998)")

text = st.text_area("Paste an email body", height=220)
if st.button("Analyze", type="primary") and text.strip():
    X = tfidf.transform([text])
    p = clf.predict_proba(X)[0, 1]
    (st.error if p >= 0.5 else st.success)(
        f"{'PHISHING' if p >= 0.5 else 'SAFE'} — phishing probability {p:.1%}"
    )
    # per-word contribution = tfidf value * coefficient
    contrib = X.multiply(clf.coef_[0]).tocoo()
    words = tfidf.get_feature_names_out()
    df = pd.DataFrame({"word": [words[j] for j in contrib.col], "contribution": contrib.data})
    st.subheader("Why? (right = phishing, left = safe)")
    top = pd.concat([df.nlargest(8, "contribution"), df.nsmallest(8, "contribution")])
    st.bar_chart(top.set_index("word"))

with st.expander("Model comparison"):
    st.dataframe(pd.read_csv(f"{ART}/results_all_models.csv").iloc[:, :6])
