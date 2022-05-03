import tensorflow as tf

class EmbeddingModel(tf.keras.Model):
    """Our primary model"""
    def __init__(self, pool_size=123, embedding_size=32, team_size=5,
                 n_hidden_predictor=128, dropout_rate=0.1, activation='tanh'):
        super(EmbeddingModel, self).__init__()

        self.embedding = tf.keras.layers.Embedding(pool_size, embedding_size,
                                                   input_length=team_size)

        self.predictor = tf.keras.Sequential(
            [
             # Takes in an embedding of each team + each team's historical avg. winrate
             tf.keras.layers.InputLayer(input_shape=(embedding_size*2 + 2,)),
             # Regularize with dropout
             tf.keras.layers.Dropout(dropout_rate),
             # Predict with 2 hidden fully connected layers
             tf.keras.layers.Dense(units=n_hidden_predictor, activation=activation),
             tf.keras.layers.Dense(units=n_hidden_predictor, activation=activation),
             # Output a win probability bounded between 0-1
             tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)
            ]
            )
        return


    def call(self, inputs):
        radiant, dire, radiant_wr, dire_wr = inputs

        radiant_embedding = self.embedding(radiant)
        dire_embedding = self.embedding(dire)

        radiant_embedding_sum = tf.reduce_sum(radiant_embedding, axis=1)
        dire_embedding_sum = tf.reduce_sum(dire_embedding, axis=1)

        pred_inputs = tf.concat((radiant_embedding_sum, dire_embedding_sum, 
                                 radiant_wr, dire_wr), axis=-1)
        prediction = self.predictor(pred_inputs)

        return prediction