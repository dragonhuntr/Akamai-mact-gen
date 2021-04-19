import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf


# The mosueevent are broken up by the ; symbol
# 1 step would be for example: 113,3,6169,864,315,-1;
#
# Explanation of the different values:
# 113: The index of the event (this is just a basic integer which counts up)
# 3: Type of the event. The different events are:
# 	- 1 - mousemove
# 	- 2 - click
# 	- 3- mousedown
# 	- 4- mouseup
# 6169: Time of the event (This is calculated from the start of a given timestamp difference.
# Basically it adds the amount it took between events
# 863: X position of the pointer
# 315: Y position of the pointer
# -1: boolean if element is active (almost always -1). These are only added when the type of event is 2/3/4
#
# The maximum amount of mouse events is always 55.
#
# The event with type 2,3,4 are OPTIONAL. These are not required,
# but the x & y cords might be useful for the movement generation.
#
# What I need the AI to learn is to generate these human-like mouse movements.
# Things like straight lines are existent, but not a lot.
# I hope the data I have provided should be enough for the start of the AI.
def process_data(minimum_length=20):
    data = pd.read_csv('data/user_data.csv')
    mact_data = data['mact_data'].to_list()
    result = []
    max_x = 0
    max_y = 0
    for ii in mact_data:
        movements = []
        prev_event, prev_x, prev_y = None, None, None
        for i in ii.split(';'):
            # empty
            if not i.strip():
                continue

            vs = i.split(',')
            # not movement
            if vs[1] != '1':
                continue
            idx, event_type, event_time, x, y = i.split(',')
            max_x = max(max_x, int(x))
            max_y = max(max_y, int(y))
            if prev_x is None:
                # origin
                movements.append([0, 0, 0])
            else:
                movements.append([int(event_time) - int(prev_event) + 100,
                                  int(x) - int(prev_x) + 100,
                                  int(y) - int(prev_y) + 100])
            prev_event = event_time
            prev_x = x
            prev_y = y

        # to short are meaningless
        if len(movements) >= minimum_length:
            result.append(movements)
    return result, (max_x, max_y)


def get_data(data, sequence_length=20):
    input_data = []
    output_data = []
    # idx = np.random.choice(len(data) - batch_size, 1)[0]
    for row in range(len(data)):
        for i in range(len(data[row]) - sequence_length - 2):
            input_data.append(data[row][i:i + sequence_length])
            output_data.append(data[row][i + 1:i + sequence_length + 1])
    return np.array(input_data), np.array(output_data)


def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(256,
                                   activation='relu',
                                   input_shape=(20, 3),
                                   return_sequences=True))
    model.add(tf.keras.layers.LSTM(128,
                                   activation='relu',
                                   return_sequences=True))
    model.add(tf.keras.layers.LSTM(64,
                                   activation='relu',
                                   return_sequences=True))
    # model.add(tf.keras.layers.TimeDistributed(
    #     tf.keras.layers.Dense(100, activation='relu')
    # ))
    model.add(tf.keras.layers.Dense(3, activation='relu'))
    model.compile(loss='mse', optimizer='adam')
    return model


def generate_sequence(model, seed, prediction_length=20):
    prediction_length += random.randint(0, 37)
    while prediction_length > 0:
        preds = model.predict(np.expand_dims(seed[-20:], axis=0))
        seed = np.concatenate((seed, preds[0][-1:]), axis=0)
        prediction_length -= 1
    return seed


def save_model(m, m_path):
    tf.keras.Model.save(m, m_path, save_format='tf')


def load_model(m_path):
    return tf.keras.models.load_model(m_path)


def format_seq(new_seq, max_x, max_y):
    origin_x = random.randint(100, max_x)
    origin_y = random.randint(100, max_y)
    print(f'x:{origin_x}, y:{origin_y}')
    result = []
    prev_x = -1000
    prev_y = -1000
    prev_ts = -1000
    for idx, row in enumerate(new_seq):
        if idx == 0:
            prev_x = origin_x
            prev_y = origin_y
            prev_ts = random.randint(100, 500)
        ts, x, y = row
        ts = int(ts) - 100 + prev_ts
        x = int(x) - 100 + prev_x
        y = int(y) - 100 + prev_y
        prev_x = x
        prev_y = y
        prev_ts = ts
        result.append([ts, x, y])
    return result


def generate_mact(data):
    result = []
    for idx, row in enumerate(data):
        row.insert(0, 1)
        row.insert(0, idx)
        result.append(",".join([str(i) for i in row]))
    return ";".join(result)


def train_model(m_path, data, epochs=1):
    input_data, output_data, = get_data(data)
    model = build_model()
    model.fit(input_data, output_data, epochs=epochs)
    save_model(model, m_path)
    return model


data = None
model = None
max_x = 0
max_y = 0


def generate_mouse_movements():
    SEQ_LENGTH = 20
    global data
    global model
    global max_x
    global max_y
    if model is None:
        data, (max_x, max_y) = process_data(minimum_length=SEQ_LENGTH + 1)
        max_y = min(2000, max_y)
        print(f'x:{max_x}, y:{max_y}')
        model_path = 'data/model'
        if not os.path.isdir(model_path):
            train_model(model_path, data, epochs=3)
        model = load_model(model_path)

    # generate
    input_data, output_data, = get_data(data)
    idx = np.random.choice(len(input_data) - 1, 1)[0]
    seed = input_data[idx][:20]
    new_seq = generate_sequence(model, seed)
    formatted_seq = format_seq(new_seq, max_x, max_y)
    mact = generate_mact(formatted_seq)
    return mact


if __name__ == '__main__':
    mact = generate_mouse_movements()
    print(mact)
