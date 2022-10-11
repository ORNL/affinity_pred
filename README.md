# affinity_pred: Affinity prediction for protein-ligand complexes 

This repository implements a transformer model for binding affinity prediction of small molecules in complex with proteins, as described in the paper
[Language models for the prediction of SARS-CoV-2 inhibitors](https://doi.org/10.1177/10943420221121804)

Local installation

```
pip install git+https://github.com/jglaser/affinity_pred
```

Evaluation on Google Colab

- [Regex tokenizer model](./eval_regex.ipynb)[![Regex tokenizer model](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jglaser/affinity_pred/blob/master/eval_regex.ipynb)
- [BERT tokenizer model](./eval_bert.ipynb) [![BERT tokenizer model](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jglaser/affinity_pred/blob/master/eval_bert.ipynb)

Validation on SARS-CoV-2 Mpro experimental data
![alt text](https://github.com/jglaser/affinity_pred/blob/master/data/postera_pr_2.50.png?raw=true)
