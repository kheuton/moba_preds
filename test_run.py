import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorboard.plugins import projector
import json
from collections import defaultdict



def run_model(data_dir=None,log_dir=None embedding_size=None, dropout_rate=None, activation=None, learning_rate=None,
              n_hidden_predictor=None, batch_size=None):
    """Main model fitting function
     All args have defaults set by argparser"""

    train_path = os.path.join(data_dir, 'train.csv')
    hero_path = os.path.join(data_dir, 'hero_names.json')

    with open('hero_names.json', 'r') as file:
        hero_names = json.load(file)
    pd.DataFrame(hero_names.values())

    labeled_data = pd.read_csv(train_path)
    no_winner = labeled_data['radiant_win'].isna()
    labeled_data = labeled_data[~no_winner]

    # Convert hero names json to be keyed on id
    hero_id_info = {}
    for name, hero_dict in hero_names.items():
        this_id = hero_dict['id']
        hero_id_info[this_id] = hero_dict

    # Create train test/splits from match data
    n_data = len(labeled_data)

    random = np.random.RandomState(seed=116)
    shuffled_rows = np.arange(n_data)
    random.shuffle(shuffled_rows, )

    train_frac = 0.8
    validate_frac = 0.1
    n_train = int(train_frac * n_data)
    n_test = int(validate_frac * n_data)
    train_rows = shuffled_rows[:n_train]
    test_rows = shuffled_rows[n_train:]

    train_data = labeled_data.iloc[train_rows, :]
    test_data = labeled_data.iloc[test_rows, :]

    train_radiant, train_dire, train_y = get_heroes_and_winner(train_data)
    test_radiant, test_dire, test_y = get_heroes_and_winner(test_data)

    win_rates = get_win_rates(train_radiant, train_dire, train_y, n_train)

    # Feature of average team winrate

    radiant_avg_wr = np.zeros(n_train)
    dire_avg_wr = np.zeros(n_train)
    test_radiant_wr = np.zeros(n_test)
    test_dire_wr = np.zeros(n_test)

    for row in range(n_train):
        radiant = train_radiant.iloc[row, :]
        dire = train_dire.iloc[row, :]

        radiant_winrates = [win_rates[hero] for hero in radiant]
        dire_winrates = [win_rates[hero] for hero in dire]

        radiant_avg = np.mean(radiant_winrates)
        dire_avg = np.mean(dire_winrates)

        radiant_avg_wr[row] = radiant_avg
        dire_avg_wr[row] = dire_avg

    for row in range(n_test):
        radiant = test_radiant.iloc[row, :]
        dire = test_dire.iloc[row, :]

        radiant_winrates = [win_rates[hero] for hero in radiant]
        dire_winrates = [win_rates[hero] for hero in dire]

        radiant_avg = np.mean(radiant_winrates)
        dire_avg = np.mean(dire_winrates)

        test_radiant_wr[row] = radiant_avg
        test_dire_wr[row] = dire_avg

    model = EmbeddingModel(pool_size=112, embedding_size=embedding_size, dropout_rate=dropout_rate,
                           n_hidden_predictor=n_hidden_predictor, activation=activation)
    loss = tf.keras.losses.BinaryCrossentropy()

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    csv_logs = os.path.join(log_dir, 'metrics.csv')
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir),
                 tf.keras.callbacks.CSVLogger(csv_logs)]

    optimizer = tf.keras.optimizers.Adam(1e-4)
    model.compile(optimizer=optimizer, loss=loss, metrics=[tf.metrics.BinaryAccuracy()])

    model.fit(x=[tf.cast(train_radiant, dtype=tf.int32), tf.cast(train_dire, dtype=tf.int32),
                 tf.cast(radiant_avg_wr, dtype=tf.float32), tf.cast(dire_avg_wr, dtype=tf.float32)],
              y=tf.cast(train_y.astype(int), dtype=tf.float32),
              callbacks=callbacks, shuffle=True,
              batch_size=batch_size, epochs=10000,
              validation_data=
              ([tf.cast(test_radiant, dtype=tf.int32), tf.cast(test_dire, dtype=tf.int32),
                tf.cast(test_radiant_wr, dtype=tf.float32), tf.cast(test_dire_wr, dtype=tf.float32)],
               tf.cast(test_y.astype(int), dtype=tf.float32)))

class EmbeddingModel(tf.keras.Model):
    def __init__(self, pool_size=123, embedding_size=32, team_size=5,
                 n_hidden_predictor=128, dropout_rate=0.1, activation='tanh'):
        super(EmbeddingModel, self).__init__()

        self.embedding = tf.keras.layers.Embedding(pool_size, embedding_size,
                                                   input_length=team_size)

        self.predictor = tf.keras.Sequential(
            [
             tf.keras.layers.InputLayer(input_shape=(embedding_size*2 + 2,)),
             tf.keras.layers.Dropout(dropout_rate),
             tf.keras.layers.Dense(units=n_hidden_predictor, activation=activation),
             tf.keras.layers.Dense(units=n_hidden_predictor, activation=activation),
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

        pred_inputs = tf.concat((radiant_embedding_sum, dire_embedding_sum, radiant_wr, dire_wr), axis=-1)
        prediction = self.predictor(pred_inputs)

        return prediction



def get_win_rates(train_radiant, train_dire, train_y, n_train):
    # Calculate historical winrates:
    win_counts = defaultdict(lambda: 0)
    game_counts = defaultdict(lambda: 0)
    for row in range(n_train):
        radiant = train_radiant.iloc[row, :]
        dire = train_dire.iloc[row, :]
        radiant_win = train_y.iloc[row]

        for hero in radiant:
            game_counts[hero] += 1
        for hero in dire:
            game_counts[hero] += 1

        if radiant_win:
            team = radiant
        else:
            team = dire

        for hero in team:
            win_counts[hero] += 1

    win_rates = {}
    for hero in win_counts.keys():
        win_rates[hero] = win_counts[hero] / game_counts[hero]


def get_heroes_and_winner(df):
    radiant_cols = [f'r{idx}_hero' for idx in range(1,6)]
    dire_cols = [f'd{idx}_hero' for idx in range(1,6)]


    # make id's start at 0
    radiant_heroes = df[radiant_cols] - 1
    dire_heroes = df[dire_cols] - 1
    winners = df['radiant_win']

    return radiant_heroes, dire_heroes, winners


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', help='Path to dataset', default='/cluster/tufts/hugheslab/kheuto01/data/dota-2-prediction/')
    parser.add_argument('--log_dir', help='Path to store logs. Must be unique for this particular run', required=True)
    parser.add_argument('--embedding_size', help='Size of hero embedding', type=int, default=32)
    parser.add_argument('--n_hidden_predictor', help='Units in hidden layer of prediction_head', type=int, default=128)
    parser.add_argument('--dropout_rate', help='Dropout rate, 0-1', type=float, default=0)
    parser.add_argument('--activation', help='Hidden layer activation in prediction head', type=str, default='tanh')
    parser.add_argument('--learning_rate', default=1e-4)
    parser.add_argument('--batch_size', default=8196)

    args = parser.parse_args()
    run_model(**vars(args))