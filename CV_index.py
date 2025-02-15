from sklearn.model_selection import KFold

def nfold_cv_index(data, folds):
    CV_nfold_train = [[],[],[],[],[]]
    CV_nfold_test = [[],[],[],[],[]]
    current_fold = 0

    kf =KFold(n_splits=folds, shuffle=True, random_state=100)

    for train_index, test_index in kf.split(data):
        CV_nfold_train[current_fold].append(train_index)
        CV_nfold_test[current_fold].append(test_index)
        current_fold += 1

    return CV_nfold_train, CV_nfold_test

