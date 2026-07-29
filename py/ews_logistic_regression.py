from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# Random seed for logistic regression
r_state = 42


def ews_logistic_regression(ews_data, signals,
                            standardise=False, do_pca=False, pca_components=7,
                            method="LR", print_output=False, **kwargs):
    x, y = ews_data[signals].values, ews_data["is_test"].values
    if type(signals) is str:
        x = x.reshape(-1, 1)

    scaler = None
    pca = None

    if standardise:
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
    if do_pca:
        pca = PCA(n_components=pca_components)
        pca.fit(x)
        x = pca.transform(x)

    if method == "LR":
        kwargs.setdefault('solver', "liblinear")
        lr_clf = LogisticRegression(random_state=r_state, **kwargs)
                                    #,solver="liblinear")
    elif method == "SGD":
        lr_clf = SGDClassifier(loss="log", random_state=r_state, **kwargs)
    else:
        print("Invalid method")
        return None

    lr_clf.fit(x, y)

    if do_pca:
        coefs = pca.inverse_transform(lr_clf.coef_.reshape(-1))
    else:
        coefs = lr_clf.coef_.reshape(-1)


    if standardise:

        w = lr_clf.coef_.reshape(-1)
        w0 = lr_clf.intercept_[0] - np.sum(w * scaler.mean_ / scaler.scale_)
        coefs = w/scaler.scale_
        coefs = dict(zip(signals, coefs))
        coefs["intercept"] = w0
        if print_output:
            print(scaler.scale_)
            print(scaler.mean_)
            print(w)
            print(coefs)

    else:
        coefs = dict(zip(signals, coefs))
        coefs["intercept"] = lr_clf.intercept_[0]
        if print_output:
            print(coefs)


    return coefs, lr_clf
