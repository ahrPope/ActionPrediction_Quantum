import tensorflow as tf
import pandas as pd
import json
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from tensorflow.keras.layers import Input, Lambda, Reshape, Dense, Dropout, LSTM, Embedding, Convolution2D, MaxPooling2D, Flatten, Concatenate, Bidirectional, GRU, Multiply, GlobalAveragePooling1D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import argparse
import sys
import os

tf.keras.backend.set_floatx('float64')

def prepare_x_y_action(df, unique_actions, number_of_actions):
    # recover all the actions in order
    actions = df['action'].values
    timestamps = df.index.tolist()
    print(('total actions', len(actions)))
    # use tokenizer to generate indices for every action
    tokenizer = Tokenizer(lower=False)
    tokenizer.fit_on_texts(actions.tolist())
    action_index = tokenizer.word_index  
    # translate actions to indexes
    actions_by_index = []
    for action in actions:
        actions_by_index.append(action_index[action])
    # create the training sets of sequences with a lenght of number_of_actions
    last_action = len(actions) - 1
    X_actions = []
    y = []
    for i in range(last_action-number_of_actions):
        X_actions.append(actions_by_index[i:i+number_of_actions])
        # represent the target action as a one-hot for the softmax
        target_action = ''.join(i for i in actions[i+number_of_actions] if not i.isdigit()) # remove the period if it exists
        target_action_onehot = np.zeros(len(unique_actions))
        target_action_onehot[unique_actions.index(target_action)] = 1.0
        y.append(target_action_onehot)
    return X_actions, y, action_index, tokenizer

def create_action_embedding_matrix_from_file(tokenizer, vector_file, embedding_size):
    data = pd.read_csv(vector_file, sep=",", header=None)
    data.columns = ["action", "vector"]
    action_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(action_index) + 1, embedding_size))
    unknown_actions = {}    
    for action, i in list(action_index.items()):
        try:
            embedding_vector = np.fromstring(data[data['action'] == action]['vector'].values[0], dtype=float, sep=' ')
            embedding_matrix[i] = embedding_vector        
        except:
            if action in unknown_actions:
                unknown_actions[action] += 1
            else:
                unknown_actions[action] = 1
                
    return embedding_matrix, unknown_actions

def LSTM_EMBEDDING_MODEL_QUANTUM(NUMBER_OF_ACTIONS, NUMBER_OF_DISTINCT_ACTIONS, embedding_matrix, embedding_size, qnode, weight_shapes, n_qubits ,trainable_embeddings=True):
    # input layer with action sequences sliding window
    actions = Input(shape=(NUMBER_OF_ACTIONS,), name='actions')

    # embedding layer
    embedding_actions = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=NUMBER_OF_ACTIONS, trainable=trainable_embeddings, name='embedding_actions')   

    # lstm layer
    lstm = LSTM(512, return_sequences=False, input_shape=(None,NUMBER_OF_ACTIONS, embedding_size), name='lstm')
    
    dense= Dense(8, activation='relu', name='dense_0')
    qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim= n_qubits, name="qlayer")
    
    # classification layer fully connected with dropout
    dense_1 = Dense(1024, activation='relu', name='dense_1')
    drop_1 = Dropout(0.8, name='drop_1')
    dense_2 = Dense(1024, activation='relu', name='dense_2')
    drop_2 = Dropout(0.8, name='drop_2')
    output_action = Dense(NUMBER_OF_DISTINCT_ACTIONS, activation='softmax', name='output_action')

    return tf.keras.models.Sequential([actions, embedding_actions, lstm, dense, qlayer, dense_1, drop_1, dense_2, drop_2, output_action])

def LSTM_EMBEDDING_MODEL(NUMBER_OF_ACTIONS, NUMBER_OF_DISTINCT_ACTIONS, embedding_matrix, embedding_size, trainable_embeddings=True):
    # input layer with action sequences sliding window
    actions = Input(shape=(NUMBER_OF_ACTIONS,), name='actions')

    # embedding layer
    embedding_actions = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=NUMBER_OF_ACTIONS, trainable=trainable_embeddings, name='embedding_actions')(actions)    

    # lstm layer
    lstm = LSTM(512, return_sequences=False, input_shape=(NUMBER_OF_ACTIONS, embedding_size), name='lstm')(embedding_actions)
    
    # classification layer fully connected with dropout
    dense_1 = Dense(1024, activation='relu', name='dense_1')(lstm)
    drop_1 = Dropout(0.8, name='drop_1')(dense_1)
    dense_2 = Dense(1024, activation='relu', name='dense_2')(drop_1)
    drop_2 = Dropout(0.8, name='drop_2')(dense_2)
    output_action = Dense(NUMBER_OF_DISTINCT_ACTIONS, activation='softmax', name='output_action')(drop_2)

    return Model(actions, output_action)

def LSTM_MODEL_QUANTUM(NUMBER_OF_ACTIONS, NUMBER_OF_DISTINCT_ACTIONS, qnode, weight_shapes, n_qubits):
    # input layer with action sequences sliding window
    actions = Input(shape=(NUMBER_OF_ACTIONS,1), name='actions')

    # lstm layer
    lstm = LSTM(512, input_shape=(None,NUMBER_OF_ACTIONS), name='lstm')
    
    dense= Dense(8, activation='relu', name='dense_0')
    qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim= n_qubits, name="qlayer")
    
    # classification layer fully connected with dropout
    dense_1 = Dense(1024, activation='relu', name='dense_1')
    drop_1 = Dropout(0.8, name='drop_1')
    dense_2 = Dense(1024, activation='relu', name='dense_2')
    drop_2 = Dropout(0.8, name='drop_2')
    output_action = Dense(NUMBER_OF_DISTINCT_ACTIONS, activation='softmax', name='output_action')

    return tf.keras.models.Sequential([actions, lstm, dense, qlayer, dense_1, drop_1, dense_2, drop_2, output_action])

def LSTM_MODEL(NUMBER_OF_ACTIONS, NUMBER_OF_DISTINCT_ACTIONS):
    # input layer with action sequences sliding window
    actions = Input(shape=(NUMBER_OF_ACTIONS,1), name='actions')

    # lstm layer
    lstm = LSTM(512, name='lstm')(actions)
    
    # classification layer fully connected with dropout
    dense_1 = Dense(1024, activation='relu', name='dense_1')(lstm)
    drop_1 = Dropout(0.8, name='drop_1')(dense_1)
    dense_2 = Dense(1024, activation='relu', name='dense_2')(drop_1)
    drop_2 = Dropout(0.8, name='drop_2')(dense_2)
    output_action = Dense(NUMBER_OF_DISTINCT_ACTIONS, activation='softmax', name='output_action')(drop_2)

    return Model(actions, output_action)


def main(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument("--neuralNetworkType",
                        type=str,
                        default="LSTM_MODEL",
                        nargs="?",
                        help="Dataset file name")
    
    args = parser.parse_args()


    num_inputs = 5
    num_samples = 20

    directory = os.getcwd()

    print('Loading DATASET...')
    DATASET = directory + "\\data\\base_kasterenA_reduced.csv"
    print(DATASET)
    df_har = pd.read_csv(DATASET, parse_dates=[[0]], index_col=0, sep=' ', header=None)
    df_har.columns = ['date','sensor', 'action', 'event', 'activity']
    df_har.index.names = ["timestamp"]
    print('DATASET loaded')

    UNIQUE_ACTIONS = directory + '\\data\\unique_actionsA_reduced.json'
    unique_actions = json.load(open(UNIQUE_ACTIONS, 'r'))


    graph_to_retrofit = 'activities'
    retrofitted_embeddings = 'False'
    word2vec_window = 5
    iterations = 5
    embedding_size = 50
    trainable_embeddings = 'True'

    X_actions, y, unique_actions, tokenizer = prepare_x_y_action(df_har, unique_actions, num_inputs) 
    # create the embedding matrix for the embedding layer initialization from FILE
    if retrofitted_embeddings == 'True':
        VECTOR_FILE = directory +'\\data\\word2vec_models\\word2vec_retrofitted_' + str(graph_to_retrofit) + '_embedding_size_' + str(embedding_size) + '_iterations_' + str(iterations) + '_word2vec_window_' + str(word2vec_window)
    else:
        VECTOR_FILE = directory +'\\data\\word2vec_models\\word2vec_embedding_size_' + str(embedding_size) + '_iterations_' + str(iterations) + '_word2vec_window_' + str(word2vec_window)
    embedding_matrix, unknown_actions = create_action_embedding_matrix_from_file(tokenizer, VECTOR_FILE, embedding_size)



    #X_actions, y, unique_actions = prepare_x_y_corrected(df_har, unique_actions, num_inputs) 

    total_actions = len(unique_actions) + 1
    print(total_actions)

    total_examples = len(X_actions)
    test_per = 0.2
    limit = int(test_per * total_examples)
    X_actions_train = X_actions[limit:]
    X_actions_test = X_actions[:limit]
    y_train = y[limit:]
    y_test = y[:limit]
    #print(('Different actions:'< total_actions))
    print(('Total examples:', total_examples))
    print(('Train examples:', len(X_actions_train), len(y_train))) 
    print(('Test examples:', len(X_actions_test), len(y_test)))
    X_actions_train = np.array(X_actions_train)
    y_train = np.array(y_train)
    X_actions_test = np.array(X_actions_test)
    y_test = np.array(y_test)
    print('Shape (X,y):')
    print((X_actions_train.shape))

    n_qubits = 8
    dev = qml.device("default.qubit", wires=n_qubits)
    #dev = qml.device("qiskit.aer", wires=n_qubits)

    @qml.qnode(dev)
    def qnode_a(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    n_layers = 6
    weight_shapes = {"weights": (n_layers, n_qubits)}

    trainable_embeddings = True if trainable_embeddings == 'True' else False
    if args.neuralNetworkType == "LSTM_EMBEDDING_QUANTUM": 
        model = LSTM_EMBEDDING_MODEL_QUANTUM(5, len(unique_actions), embedding_matrix, embedding_size, qnode_a, weight_shapes, n_qubits, trainable_embeddings)

    elif args.neuralNetworkType == "LSTM_EMBEDDING": 
        model = LSTM_EMBEDDING_MODEL(5, len(unique_actions), embedding_matrix, embedding_size, trainable_embeddings)

    elif args.neuralNetworkType == "LSTM_MODEL_QUANTUM":
        model = LSTM_MODEL_QUANTUM(5, len(unique_actions), qnode_a, weight_shapes, n_qubits)
    elif args.neuralNetworkType == "LSTM_MODEL":
            model = LSTM_MODEL(5, len(unique_actions))


    ############################################

    tokenizer = Tokenizer(lower=False)
    tokenizer.fit_on_texts(unique_actions)
    action_index = tokenizer.word_index

    y_train_final = []
    y_test_final = []

    print(action_index)
    keys = action_index.keys()
    print(list(keys))
    print("--------------------------")
    for value in y_train.tolist():

        print(value)

        action = np.argmax(value, axis=0)
        print(list(keys)[action])
        y_train_final.append(int(action))
        print("--------------------------")
        
    for value in y_test.tolist():
        print(value)

        action = np.argmax(value, axis=0)
        print(list(keys)[action])
        y_test_final.append(int(action))
        print("--------------------------")

    ############################################

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy', 'mse', 'mae'])

    fitting = model.fit(X_actions_train.tolist(), 
                        y_train.tolist(), 
                        batch_size=64, 
                        epochs=1000, 
                        validation_data=(X_actions_test.tolist(), y_test.tolist()))

    predictions = model.predict(X_actions_test, 128)
    correct = [0] * 5
    prediction_range = 5
    for i, prediction in enumerate(predictions):
        correct_answer = y_test[i].tolist().index(1)       
        best_n = np.sort(prediction)[::-1][:prediction_range]
        for j in range(prediction_range):
            if prediction.tolist().index(best_n[j]) == correct_answer:
                for k in range(j,prediction_range):
                    correct[k] += 1 
    for i in range(prediction_range):
        print('acc_at_' + str(i+1)+":" + str((correct[i] * 1.0) / len(y_test)))


if __name__ == "__main__":
    main(sys.argv)