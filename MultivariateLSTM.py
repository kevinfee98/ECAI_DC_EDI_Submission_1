import tensorflow as tf
from tensorflow.keras import layers

class MultivariateLSTM(tf.keras.Model):
    def __init__(self, input_dim, lag, hidden_dim, num_layers, output_dim, dropout_rate=0.2, name="MultivariateLSTM"):
        super(MultivariateLSTM, self).__init__(name=name)
        self.input_dim = input_dim
        self.lag = lag
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        # Build submodules
        self.input_dropout = layers.Dropout(self.dropout_rate)

        # Create stacked LSTM layers
        self.lstm_layers = []
        for i in range(self.num_layers):
            # For all but last layer, return_sequences=True
            return_sequences = (i < self.num_layers - 1)
            lstm = layers.LSTM(self.input_dim, return_sequences=return_sequences,
                               name=f"LSTM_{i+1}")
            self.lstm_layers.append(lstm)

        # Final dense projection
        self.output_dense = layers.Dense(self.output_dim, name="Output")
        self.softplus = layers.Activation("softplus", name="NonNegativeOutput")
    def call(self, inputs, training=False):
        """
        inputs: shape (batch, lag, input_dim)
        returns: (batch, output_dim)
        """
        x = self.input_dropout(inputs, training=training)

        # Pass through stacked LSTMs
        for i, lstm in enumerate(self.lstm_layers):
            x = lstm(x, training=training)
            # x has shape (batch, hidden_dim) after the last LSTM
        outputs = self.output_dense(x)  # (batch, N)
        outputs = self.softplus(outputs)
        return outputs