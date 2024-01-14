import tensorflow as tf

class LSTMModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, max_length):
        super(LSTMModel, self).__init__()
        
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.lstm1 = tf.keras.layers.LSTM(32, return_sequences=True)
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.lstm2 = tf.keras.layers.LSTM(32)
        self.dropout3 = tf.keras.layers.Dropout(0.3)
        self.dense_output = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, inputs):
        x = self.embedding_layer(inputs)
        x = self.dropout1(x)
        x = self.lstm1(x)
        x = self.dropout2(x)
        x = self.lstm2(x)
        x = self.dropout3(x)
        output = self.dense_output(x)
        return output