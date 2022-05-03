"""Predict winrate given teams"""
import numpy as np
import os
import tensorflow as tf
import json

from model import EmbeddingModel

if __name__ == '__main__':
    data_dir = './data'
    result_dir = './results'
    log_dir = './logs/embedding/'
    hero_path = os.path.join(data_dir, 'hero_names.json')
    checkpoint_path = os.path.join(result_dir, 'final_model.ckpt')

    # Chosen by grid-search
    embedding_size = 16
    dropout_rate = 0
    activation = 'tanh'
    n_hidden_predictor = 128
    learning_rate = 1e-3
    batch_size = 8196
    epochs = 150

    with open(hero_path, 'r') as file:
        hero_names = json.load(file)

    winrates_path = os.path.join(result_dir, 'winrates.json')
    win_rates = json.load(open(winrates_path, 'r'))

    # Convert hero names json to be keyed on id
    hero_name_info = {}
    for name, hero_dict in hero_names.items():
        this_name = hero_dict['localized_name']
        hero_name_info[this_name] = hero_dict

    # load saved model
    model = EmbeddingModel(pool_size=112, embedding_size=embedding_size, dropout_rate=dropout_rate,
                           n_hidden_predictor=n_hidden_predictor, activation=activation)
    model.load_weights(checkpoint_path)

    radiant = []
    dire = []
    examples = np.random.choice(list(hero_name_info.keys()), 10, replace=False)
    for i in range(5):
        hero = input(f'Input the name of hero {i + 1} on the Radiant team, for example, {examples[i]}: ')
        radiant.append(hero)

    for i in range(5):
        hero = input(f'Input the name of hero {i + 1} on the Dire team, for example, {examples[i + 5]}: ')
        dire.append(hero)

    radiant_ids = [hero_name_info[hero]['id'] - 1 for hero in radiant]
    dire_ids = [hero_name_info[hero]['id'] - 1 for hero in dire]
    radiant_winrates = [win_rates[str(hero)] for hero in radiant_ids]
    dire_winrates = [win_rates[str(hero)] for hero in dire_ids]

    radiant_avg = np.mean(radiant_winrates)
    dire_avg = np.mean(dire_winrates)

    # Do formatting to make it think this is a tensor
    radiant_wr_tensor = model(
        [tf.expand_dims(tf.cast(radiant_ids, dtype=tf.int32), 0), tf.expand_dims(tf.cast(dire_ids, dtype=tf.int32), 0),
         tf.expand_dims(tf.cast(radiant_avg, dtype=tf.float32), 0),
         tf.expand_dims(tf.cast(dire_avg, dtype=tf.float32), 0)])
    radiant_pred_wr = radiant_wr_tensor.numpy()[0][0]
    print(f'Radiant have an expected {100 * radiant_pred_wr:.2f}% win rate')
