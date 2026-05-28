# Natural Language Processing (NLP)

Overview
- NLP enables machines to analyze, understand, and generate human language.

Important subtopics
- Tokenization, embeddings (word2vec, GloVe, contextual BERT embeddings)
- Sequence models: RNNs, LSTMs, Transformers
- Tasks: classification, NER, translation, summarization, question answering

Key notes
- Preprocessing: clean text, handle punctuation and casing, consider subword tokenization.
- Use pretrained transformers (BERT, GPT) for many tasks.

Quick example (sentiment analysis)
- Fine-tune a pretrained transformer on labeled sentiment data.

have a look
Mermaid pipeline
```mermaid
flowchart LR
  A[Raw text] --> B[Tokenize]
  B --> C[Embeddings]
  C --> D[Model (Transformer)]
  D --> E[Prediction]
```

Notes on images
- Add an attention heatmap example: `images/nlp_attention.png`.
