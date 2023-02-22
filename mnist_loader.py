import pandas as pd
import numpy as np


def load_data(trainfilename, testfilename):

    # Bring in training & test data via pandas
    train_df = pd.read_csv(trainfilename)
    test_df = pd.read_csv(testfilename)

    # Convert correct answer columns, to numpy arrays
    train_answrs = train_df.iloc[:, 0].to_numpy()
    test_answrs = test_df.iloc[:, 0].to_numpy()

    # Convert training answers, to column vector form (not test answers, network won't train using them)
    train_answrs_v = []
    for answr in train_answrs:
        vectr = np.zeros(shape=(10, 1))
        vectr[answr][0] = 1
        train_answrs_v.append(vectr)

    # Convert all instances to numpy arrays
    train_instncs = train_df.iloc[:, 1:].to_numpy()
    test_instncs = test_df.iloc[:, 1:].to_numpy()

    # Normalize/scale values in instances, down to 0-1, (currently 0-255)
    train_instncs_norm = [np.array(x) / 255 for x in train_instncs]
    test_instncs_norm = [np.array(x) / 255 for x in test_instncs]

    # Convert each instance, to column vector form
    train_instncs_v = []
    for instnc in train_instncs_norm:
        instnc_v = np.expand_dims(instnc, axis=0)
        train_instncs_v.append(instnc_v.T)

    test_instncs_v = []
    for instnc in test_instncs_norm:
        instnc_v = np.expand_dims(instnc, axis=0)
        test_instncs_v.append(instnc_v.T)


    # Pair instances, with corresponding answers:
    train_data = [ (train_instncs_v[i], train_answrs_v[i]) for i in range(0, len(train_answrs)) ]
    test_data = [ (test_instncs_v[i], test_answrs[i]) for i in range(0, len(test_answrs)) ]

    return train_data, test_data



