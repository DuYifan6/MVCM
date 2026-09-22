# Model placement

Model weights are intentionally excluded from Git. Download them from their
official sources and place them in the following directories:

```text
models/
├── bert-base-uncased/
├── all-mpnet-base-v2/
└── nli-model/
```

The recommended NLI checkpoint used by the experiments is
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`. The configured contradiction
class index is `2`.

Alternative locations can be supplied with `MVCT_BERT_MODEL`,
`MVCT_SBERT_MODEL`, and `MVCT_NLI_MODEL`. The code loads models in offline mode
where reproducibility requires a local snapshot.
