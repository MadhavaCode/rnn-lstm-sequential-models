# rnn-lstm-sequential-models

Recurrent Neural Networks and LSTMs for processing sequential data.

## 📓 Notebooks

| Notebook | Topics |
|---|---|
| `RNNs.ipynb` | SimpleRNN theory, vanishing gradient problem, Keras implementation |
| `LSTM.ipynb` | LSTM gates (forget/input/output/cell state), multi-layer LSTM with regularization |

## 🏗 LSTM Architecture (IMDB Sentiment)
Dataset: 25,000 train / 25,000 test movie reviews, padded to 50 tokens.

```
LSTM(1024, return_sequences=True) → BatchNorm → Dropout(0.3)
LSTM(512,  return_sequences=True) → BatchNorm → Dropout(0.2)
LSTM(256,  return_sequences=True) → BatchNorm → Dropout(0.2)
LSTM(128)                         → BatchNorm → Dropout(0.1)
Dense(64, relu) → Dense(1, sigmoid)
```
Compiled: `binary_crossentropy` + `Adam(lr=1e-3)`, 10 epochs.
Improved model adds L2 regularization (`kernel_regularizer=l2(0.01)`).

## 🛠 Tech Stack
`Python` · `TensorFlow` · `Keras` · `NumPy` · `scikit-learn`

## 👤 Author
**Madhava Narra** — [@MadhavaCode](https://github.com/MadhavaCode) · Built Apr–May 2025
