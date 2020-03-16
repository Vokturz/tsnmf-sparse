# tsnmf-sparse

This repository contains an implementation of Topic-Supervised Non-Negative Matrix Factorization (TS-NMF) [1] with Sparse Matrices in Python, using a Scikit-Learn's compatible API.

## How it Works
From [1]:  Suppose that one supervises *k << n* documents and identifies *l << t* topics that were contained in a subset of  the  documents. One can supervise the `NMF` method using this information, represented by an *n×d topic supervision* matrix *L*.The elements of *L* contrain the importance weights of matrix *W* and are of the following form:

<img align='middle' src='https://latex.codecogs.com/gif.latex?%5Clarge%20L_%7Bij%7D%3D%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%201%20%26%5Ctext%7Bif%20topic%20%7D%20j%20%5Ctext%7B%20is%20permitted%20in%20document%20%7D%20i%5C%5C%200%20%26%5Ctext%7Bif%20topic%20%7D%20j%20%5Ctext%7B%20is%20%5Ctextit%7Bnot%7D%20permitted%20in%20document%20%7D%20i%5C%5C%20%5Cend%7Bmatrix%7D%5Cright.'/>

Then, for a term-document matrix *V* and supervision matrix *L*, TS-NMF seeks matrices *W* and *H* that minimize

<img align='middle' src='https://latex.codecogs.com/gif.latex?%5Clarge%20D_%7BTS%7D%28W%2CH%29%3D%7C%7CV-%28W%20%5Ccirc%20L%29%20H%7C%7C%5E2%2C%5Cquad%20W%20%5Cgeq%200%2C%5Cquad%20H%20%5Cgeq0.'/>

Where ○ represent the Hadamard (element-wise) product operator.

## Installation
You can install TS-NMF via pip:

```python
pip install tsnmf
```

Or clonning this repository and running `setup.py`:

```python
python setup.py install
```
## Usage
TS-NMF is used in a similar way as the module `decomposition.NMF` from Scikit-Learn. The extra thing that you need is a `list of list` that contains the labels to build the matrix *L*.

Suppose you want to get 3 topics from 5 documents. The 5 documents should be represented in a matrix `V`, the most used way is apply a TF-IDF Vectorizer, which reflect how important a word is to a document.

Each element of the `list of list` of labels correspond to a document. These elements contain a list of topics that contrain the document. For example

```python

labels = [[],
          [0,2], # document 1
          [],
          [],
          [1]] # document 4
```

means that the document 1 is contrained to be topic 0 or 2 and document 4 to be topic 1. For the other documents all the topics are permitted.

Finally, to run TS-NMF:

```python
from tsnmf import TSNMF

tsnmf = TSNMF(n_components=3, random_state=1)
W = tsnmf.fit_transform(V, labels=labels)
H = tsnmf.components_
```

## Credits

  - Developed mainly by Victor Navarro (@vokturz), under the guidance of Eduardo Graells-Garrido (@carnby), in the context of CONICYT Fondo de Fomento al Desarrollo Científico y Tecnológico (FONDECYT) Proyecto de Iniciación 11180913.
  - Based on [scikit-learn's NMF code](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/decomposition/_nmf.py) and the original [ws-nmf](https://github.com/kelsey-macmillan/ws-nmf). 

## References

  1. MacMillan, Kelsey, and James D. Wilson. ["Topic supervised non-negative matrix factorization."](https://arxiv.org/abs/1706.05084) _arXiv preprint arXiv:1706.05084_ (2017).
