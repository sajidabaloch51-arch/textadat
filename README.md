# BERT + Balochi GPT (Streamlit app)

One Streamlit app with two sections.

## Section 1 - BERT (encoder-only, multilingual, high-resource languages)
- Model: multilingual BERT fine-tuned for Named Entity Recognition (PER, ORG, LOC).
- Default: `Davlan/bert-base-multilingual-cased-ner-hrl` (English, German, French, Spanish, Italian, Dutch, Portuguese, Arabic, Chinese, Latvian).
- Our own fine-tuning notebook (mBERT on WikiANN): `sajida12/bert-english-ner`.
- Input: a short sentence. Output: the entities found.

## Section 2 - GPT (decoder-only, monolingual, low-resource: Balochi)
- Base model: `goldfish-models/bgp_latn_full` (Eastern Balochi, Latin script).
- Fine-tuned on the `Balochi_sentences` column of `Aynkader/Balochi-dataset` (causal language modelling).
- Input: a short Balochi prompt. Output: generated continuation.
- Generated text needs review by a fluent Balochi speaker.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Configuration (Streamlit Secrets)
```
BERT_MODEL_ID = "Davlan/bert-base-multilingual-cased-ner-hrl"
GPT_MODEL_ID = "YOUR_HF_USERNAME/balochi-gpt2-generation"
```
