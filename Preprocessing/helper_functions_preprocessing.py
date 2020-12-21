
import torch
from sklearn.model_selection import train_test_split

def save_train_test_data(Y, D, X, path, random_state = None):

    # split into A and B set
    dataset_size = Y.size()[0]
    a_index, b_index = train_test_split(range(dataset_size), test_size=0.5, random_state=random_state)
    a_index_train, a_index_test, b_index_train, b_index_test = train_test_split(a_index, b_index, test_size=0.2, random_state=random_state)
    Y_a_train, D_a_train, X_a_train = Y[a_index_train], D[a_index_train], X[a_index_train]
    Y_a_test, D_a_test, X_a_test = Y[a_index_test], D[a_index_test], X[a_index_test]
    Y_b_train, D_b_train, X_b_train = Y[b_index_train], D[b_index_train], X[b_index_train]
    Y_b_test, D_b_test, X_b_test = Y[b_index_test], D[b_index_test], X[b_index_test]


    #save preprocessed data
    torch.save(Y_a_train, path + 'Y_a_train')
    torch.save(Y_a_test, path + 'Y_a_test')
    torch.save(Y_b_train, path + 'Y_b_train')
    torch.save(Y_b_test, path + 'Y_b_test')

    torch.save(D_a_train, path + 'D_a_train')
    torch.save(D_a_test, path + 'D_a_test')
    torch.save(D_b_train, path + 'D_b_train')
    torch.save(D_b_test, path + 'D_b_test')

    torch.save(X_a_train, path + 'X_a_train')
    torch.save(X_a_test, path + 'X_a_test')
    torch.save(X_b_train, path + 'X_b_train')
    torch.save(X_b_test, path + 'X_b_test')