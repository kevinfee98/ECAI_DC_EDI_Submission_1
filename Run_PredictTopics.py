import numpy as np
import tensorflow as tf
from sympy import ceiling
from tensorflow.keras import layers, models, losses, optimizers
import json

from tensorflow.python.layers.core import dropout


def readDataset(file_path):
    dict = {}
    with open(file_path) as f:
        dict = json.load(f)
    return dict

def remove_all_zeros_years(dict):
    updated_dict = {}
    isZero = True
    for key, value in dict.items():
        for v in value:
            if v != 0:
                isZero = False
        if isZero == False:
            updated_dict[key] = value
    return updated_dict

def get_years_topics(dict):
    years = []
    topics = []
    for key, value in dict.items():
        years.append(key)
        for v in value:
            uri_split = v.split('//')
            uri_split = uri_split[1].split('/')
            topics.append(uri_split[len(uri_split)-1])
    return np.unique(years), np.unique(topics)

def get_year_topics_freq(dict):
    topic_freq = {}
    for year, value in dict.items():
        freqs = []
        for v in value:
            freqs.append(dict[year][v])
        topic_freq[year] = freqs
    return topic_freq
def get_year_topics(dict):
    year_topics = {}
    for year, value in dict.items():
        topics = []
        for v in value:
            uri_split = v.split('//')
            uri_split = uri_split[1].split('/')
            topic = uri_split[len(uri_split) - 1]
            topics.append(topic)
        year_topics[year] = topics
    return year_topics

topics_dict = readDataset("Data/topic_year_counts.json")
topics_dict = remove_all_zeros_years(topics_dict)
years, topics = get_years_topics(topics_dict)
topics_freq = get_year_topics_freq(topics_dict)
print(f"years of proceedings: {years}\n"
      f"Number of years: {len(years)}\n")
print(f"topics of proceedings: {topics}\n"
      f"Number of topics: {len(topics)}\n")
print(f"topics frequencies: {topics_freq}\n")
# ----------------------------
# Data preparation
# ----------------------------
# Suppose you have:
# years: 1D array of years, length = Y
# freqs: 2D array of shape (Y, N) with nonnegative counts per topic per year
# Example dummy data (replace with real data)
Y = len(years)  # number of years
N = len(topics)  # number of topics
def dict_to_freqs_array(freqs_dict):
    years_sorted = sorted((int(k) for k in freqs_dict.keys()))
    freq_rows = [np.asarray(freqs_dict[str(year)], dtype=np.float32) for year in years_sorted]
    return np.stack(freq_rows, axis=0), years_sorted

freqs, years_order = dict_to_freqs_array(topics_freq)

Y, N = freqs.shape
np.random.seed(0)
#years = np.arange(2000, 2000 + Y)
#freqs = topics_freq #np.random.poisson(lam=5.0, size=(Y, N)).astype(np.float32)

# Hyperparameters
lags = 10  # how many past years to use for prediction
train_ratio = 0.8  # train/val split
batch_size = 16
hidden_dim = 128
num_layers = 5
learning_rate = 1e-3
num_epochs = 50

# Create sequences of shape (num_samples, lags, N) for inputs and (num_samples, N) for targets
def create_sequences(freqs, lags):
    X, y = [], []
    for t in range(lags, len(freqs)):
        X.append(freqs[t - lags:t, :])  # (lags, N)
        y.append(freqs[t, :])          # (N,)
    X = np.stack(X)  # (samples, lags, N)
    y = np.stack(y)  # (samples, N)
    return X, y

X, y = create_sequences(freqs, lags)
num_samples = X.shape[0]
input_dim = N
output_dim = N

# Train/val split
split = int(train_ratio * num_samples)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
print(f"X_train: {X_train}\ny_train: {y_train}")
# Build TensorFlow Dataset
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(buffer_size=1024).batch(batch_size)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(batch_size)

# ----------------------------
# Model: Multivariate LSTM Regressor
# ----------------------------
def build_multivariate_lstm(input_dim, lag, hidden_dim, num_layers, output_dim):
    """
    input_dim: number of features per time step (N topics)
    lag: sequence length (lags)
    hidden_dim: LSTM hidden size
    num_layers: number of LSTM layers
    output_dim: number of outputs (N topics)
    """
    inputs = tf.keras.Input(shape=(lag, input_dim), name="inputs")  # (batch, lag, N)
    dropout = tf.keras.layers.Dropout(0.2)
    x = dropout(inputs)
    # Stack LSTM layers
    for i in range(num_layers):
        # For all but last layer, return_sequences to allow stacking
        return_sequences = (i < num_layers - 1)
        x = layers.LSTM(hidden_dim, return_sequences=return_sequences,
                        name=f"LSTM_{i+1}")(x)

    # x has shape (batch, hidden_dim) after the last LSTM
    outputs = layers.Dense(output_dim, name="Output")(x)  # (batch, N)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MultivariateLSTM")
    return model
from MultivariateLSTM import MultivariateLSTM
# Instantiate the model
model = MultivariateLSTM(input_dim=N, lag=lags,
                         hidden_dim=hidden_dim, num_layers=num_layers,
                         output_dim=N, dropout_rate=0.0)

# Compile
model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
              loss=losses.MeanAbsoluteError(),
              metrics=[tf.keras.metrics.MeanSquaredError()])

# ----------------------------
# Training
# ----------------------------
# Early stopping to prevent overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6,
                                              restore_best_weights=True)

history = model.fit(
    train_ds,
    #validation_data=val_ds,
    epochs=num_epochs,
    #callbacks=[early_stop]
)

# ----------------------------
# Evaluation / Inference
# ----------------------------
# Example: forecast the next year after the last observed years
counts_above_zero = 0
if freqs.shape[0] >= lags:
    last_seq = freqs[-lags:, :]  # (lags, N)
    last_seq = last_seq[np.newaxis, ...]  # (1, lag, N)
    pred_next = model.predict(last_seq)  # (1, N)
    pred_next = pred_next[0]  # (N,)
    preds_dict = {}
    for i in range(len(topics)):
        if pred_next[i] < 0.5:
            pred_next[i] = 0
        else:
            pred_next[i] = ceiling(pred_next[i])
        preds_dict[str(topics[i])] = float(pred_next[i])
        if pred_next[i] > 0.0:
            counts_above_zero += 1
    print(f"Predicted Topic Frequencies for ECAI 2025:\n{preds_dict}")
    print(f"Total Predicted Topics: {counts_above_zero}")
    pretty_json = json.dumps(preds_dict, indent=4)
    print(pretty_json)
    with open("Data/ECAI_2025_Topic_Predictions.json", "w") as f:
        json.dump(preds_dict, f, indent=4)
else:
    print("Not enough data to generate prediction with the given lag.")