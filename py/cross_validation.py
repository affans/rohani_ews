import itertools
import numpy as np
import pandas as pd
# import ews
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import py.ews_logistic_regression as elr
import py.helper as h



def run_cross_validation(use_incidence, folder, aggregation_period=4,
                         use_parallel=False, n_jobs=1):

    # Select signals to be used in learning:
    signals = h.signals()

    # Read in parameters:
    params_df = pd.read_csv(folder+"/simulation_parameters.csv")
    n_models = params_df.shape[0]

    # Read in time series
    df = pd.concat(
        [pd.read_csv(folder+"/data_" + str(i) + ".csv")
         .assign(model=i).assign(is_test=int(params_df.loc[i, "R0_f"] == 1.0))
         for i in params_df.index.values])
    print("Read in files: ok")

    # Cross-validation
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]


    np.random.seed(3658042)
    n_chunks = 10
    rnd_models = np.random.permutation(params_df.index.values[:(n_models//2)])
    chunk_arr = np.array([chunk for chunk in chunks(rnd_models,
                                                    len(rnd_models)//n_chunks)])

    p_str = [100, 10, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    w_str = [52, 104, 156, 208, 260, 312]
    plist = []
    auc_list = []

    for w in w_str:
        ews_data = h.get_ews(df, params_df, agg=aggregation_period,
                             wtime=w, mv_method="exp",
                             use_incidence=use_incidence)
        ews_data = ews_data[ews_data.Time > 10 * 365]
        ews_data = ews_data[~ews_data.isin([np.inf, -np.inf, np.nan])
            .any(axis=1)]
        ews_data = ews_data[~ews_data[signals].isna().any(axis=1)]
        ews_g = ews_data.groupby("model")
        print("Calculate ews: ok")

        def get_auc_for_chunk(i, p):
            chunk = chunk_arr[i]
            models = np.concatenate((chunk, chunk + n_models // 2))
            train_models = ews_g.filter(lambda x: x.name not in models)
            test_models = ews_g.filter(lambda x: x.name in models)
            # print("test/train ok")

            coefs, lr_clf = elr.ews_logistic_regression(train_models,
                                                        standardise=True,
                                                        signals=signals,
                                                        do_pca=False,
                                                        penalty="l1",
                                                        C=p,
                                                        solver="liblinear")

            c2 = pd.Series(coefs)
            tdf = test_models[signals].dot(c2[signals]) + c2["intercept"]
            auc = roc_auc_score(test_models["is_test"], tdf)

            print(w, i, p)
            return pd.Series({"p": p, "w": w, "auc": auc})

        loop_pars = list(itertools.product(np.arange(n_chunks), p_str))

        if use_parallel:
            auc_list += Parallel(n_jobs=n_jobs)(delayed(get_auc_for_chunk)(*par)
                                                for par in loop_pars)
        else:
            auc_list += [get_auc_for_chunk(*par) for par in loop_pars]

    auc_cval = pd.concat(auc_list, axis=1).transpose()
    auc_g = auc_cval.groupby(["p", "w"]).agg({"auc": ["mean", "std"]})\
        .reset_index()
    auc_g.columns = ["p", "w", "auc", "std"]
    p_performance = auc_g

    p_performance.to_csv(folder+"/k-fold-cross-validation" +
                         h.incidence_filepath(use_incidence)
                         + ".csv")

    p_performance.to_csv("~/Dropbox/"+folder[7:-1]+"-k-fold-cross-validation" +
                         h.incidence_filepath(use_incidence)
                         + ".csv")
