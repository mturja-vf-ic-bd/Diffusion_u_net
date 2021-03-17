from sklearn.model_selection import StratifiedKFold

from data_utilities.prepare_data import prepare_data_amy_with_temp_graph_data, prepare_data_v8


def generateValidationFolds(X, y, k=5):
    import pickle
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold = 0
    for train_index, test_index in skf.split(X, y):
        print((y[test_index] == 0).sum(), (y[test_index] == 1).sum())
        with open("fold_ids/fold_{}.p".format(fold), "wb") as f:
            pickle.dump({"X_train": X[train_index], "y_train": y[train_index],
                         "X_valid": X[test_index], "y_valid": y[test_index]}, f)
        fold = fold + 1


def generateTestSet(X, y, test_size=0.10, rnd=42):
    import pickle
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rnd, shuffle=True)
    with open("fold_ids/fold_5.p", "wb") as f:
        pickle.dump({"X_train": X_train,
                     "y_train": y_train,
                     "X_valid": X_test,
                     "y_valid": y_test}, f)


if __name__ == '__main__':
    import pickle
    label_map_ad = {"CN": 0, "SMC": 0, "LMCI": 1, "AD": 1}
    x, y, dx, g = prepare_data_v8(label_map_ad)
    data = generateTestSet(x.detach().cpu().numpy(), dx.detach().cpu().numpy())
    generateValidationFolds(data["X_train"], data["y_train"])

