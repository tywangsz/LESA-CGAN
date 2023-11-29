
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from encode_label import end_cond

def MinMax(train_data):
    scaler = MinMaxScaler()
    num_instances, num_time_steps, num_features = train_data.shape
    train_data = np.reshape(train_data, (-1, num_features))
    train_data = scaler.fit_transform(train_data)
    train_data = np.reshape(train_data, (num_instances, num_time_steps, num_features))
    return train_data, scaler




def google_data_loading(seq_length):

    x = np.loadtxt(r'C:\Users\wsz\Desktop\encoder_decoder_transformer/DE_Combine.csv', delimiter=",", skiprows=1)
    x = np.array(x)

    # Build dataset
    dataX = []

    # Cut data by sequence length
    for i in range(0, len(x) - seq_length):
        _x = x[i:i + seq_length]
        dataX.append(_x)

    # Mix Data (to make it similar to i.i.d)
    idx = np.random.permutation(len(dataX))

    outputX = []
    for i in range(len(dataX)):
        outputX.append(dataX[idx[i]])

    X_train = np.stack(dataX)

    x_x, scaler = MinMax(X_train[:, :, :-1])


    x_x = np.dstack((x_x, X_train[:, :, -1]))

    return x_x, X_train, scaler


seq_length = 1

features_n = 81

noise_dim = seq_length * features_n
SHAPE = (seq_length, features_n)
hidden_dim = features_n * 4