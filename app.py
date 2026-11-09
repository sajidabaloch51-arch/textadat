import os
import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

st.set_page_config(page_title="BERT + Balochi GPT", page_icon="🧠", layout="centered")


def get_setting(name, default):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)


# ---- Model IDs (change in Streamlit "Secrets" or edit here) ----
# Encoder-only multilingual BERT, already fine-tuned for NER on high-resource languages
BERT_MODEL_ID = get_setting("BERT_MODEL_ID", "Davlan/bert-base-multilingual-cased-ner-hrl")
# Decoder-only GPT-2 fine-tuned on Balochi (your own model on Hugging Face)
GPT_MODEL_ID = get_setting("GPT_MODEL_ID", "sajida12/balochi-gpt2-generation")
GPT_BASE_ID = "goldfish-models/bgp_latn_full"

st.title("NLP Demo: BERT and GPT")
tab1, tab2 = st.tabs(["Section 1: BERT (Encoder-only)", "Section 2: GPT (Decoder-only)"])


# ======================= SECTION 1: BERT =======================
@st.cache_resource
def load_bert():
    return pipeline(
        "token-classification",
        model=BERT_MODEL_ID,
        aggregation_strategy="simple",
    )


with tab1:
    st.header("Multilingual Named Entity Recognition")
    st.write(
        "Model: multilingual **BERT** (encoder-only). Task: find **people (PER), "
        "organizations (ORG) and locations (LOC)** in a short sentence. "
        "Works for high-resource languages such as English, German, French, "
        "Spanish, Italian, Dutch, Portuguese, Arabic and Chinese."
    )
    examples = {
        "English": "Sajida studies at the University of Gwadar in Pakistan.",
        "German": "Angela Merkel besuchte Berlin im Auftrag von Siemens.",
        "French": "Emmanuel Macron a rencontré des responsables de Google à Paris.",
        "Spanish": "Lionel Messi juega para el Inter Miami en Estados Unidos.",
    }
    choice = st.selectbox("Pick an example (or type your own below)", list(examples))
    text = st.text_area("Short sentence", value=examples[choice], key="bert_input")

    if st.button("Find entities", type="primary"):
        if not text.strip():
            st.warning("Please type a sentence.")
        else:
            try:
                with st.spinner("Loading BERT model (first time can take a minute)..."):
                    ner = load_bert()
                results = ner(text)
                if results:
                    st.subheader("Detected entities")
                    st.table(
                        [
                            {
                                "Text": r["word"],
                                "Type": r["entity_group"],
                                "Confidence": round(float(r["score"]), 3),
                            }
                            for r in results
                        ]
                    )
                else:
                    st.info("No named entities found.")
            except Exception as exc:
                st.error(f"BERT model failed: {exc}")


# ======================= SECTION 2: GPT =======================
@st.cache_resource
def load_gpt():
    try:
        tok = AutoTokenizer.from_pretrained(GPT_MODEL_ID, use_fast=False)
    except Exception:
        tok = AutoTokenizer.from_pretrained(GPT_BASE_ID, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(GPT_MODEL_ID)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model


with tab2:
    st.header("Balochi Text Generation")
    st.write(
        "Model: **GPT-2** (decoder-only) fine-tuned on Balochi (low-resource, "
        "monolingual, Latin script). Type the start of a Balochi sentence and the "
        "model continues it."
    )
    prompt = st.text_area("Balochi prompt", placeholder="Man Balochi zaban", key="gpt_input")
    max_new = st.slider("New tokens", 10, 100, 40)
    temperature = st.slider("Temperature", 0.2, 1.2, 0.8, 0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)

    if st.button("Generate", type="primary"):
        if not prompt.strip():
            st.warning("Please type a Balochi prompt.")
        else:
            try:
                with st.spinner("Loading GPT model..."):
                    tok, model = load_gpt()
                text_in = (tok.cls_token or "") + prompt.strip()
                inputs = tok(text_in, return_tensors="pt", truncation=True, max_length=400)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                    )
                st.subheader("Generated Balochi text")
                st.write(tok.decode(out[0], skip_special_tokens=True))
                st.caption("Generated text may contain errors. A fluent Balochi speaker should review it.")
            except Exception as exc:
                st.error(f"GPT model failed: {exc}")
