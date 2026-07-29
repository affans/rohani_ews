import py.ews_logistic_regression as elr
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn import metrics
import py.helper as h



def run_classifier_training(use_incidence, use_cross_val, calculate_ews, folder,
                            save_df=False, aggregation_period=4,
                            training_signals=None, hyperparameters=(156,0.0001)):
    """

    :param use_incidence:
    :param use_cross_val:
    :param calculate_ews:
    :param folder:
    :param save_df:
    :param aggregation_period: multiplier of simulation sampling frequency
    :return:
    """

    if training_signals is None:
        # Select signals to be used in learning (hardcoded in helper)
        # default is use all
        signals = h.signals()
    else:
        signals = training_signals

    # Read cross-validation results
    if use_cross_val:
        w_min, c_min = h.read_cross_val(folder, use_incidence)
    else:
        # Hyperparameter values if use_cross_val==False
        # Option not used in full analysis
        w_min, c_min = hyperparameters


    # Read data
    params_df = pd.read_csv(folder+"/simulation_parameters.csv")

    if calculate_ews:
        # Read in time series
        df = pd.concat(
            [pd.read_csv(folder+"/data_" + str(i) + ".csv")
             .assign(model=i).assign(is_test=int(params_df.loc[i, "R0_f"] == 1.0))
             for i in params_df.index.values])
        # calculate EWS
        edf = h.get_ews(df, params_df, agg=aggregation_period, wtime=w_min,
                        mv_method="exp", use_incidence=use_incidence)
        edf.to_csv(folder + "/ews_data" +
                   h.incidence_filepath(use_incidence) + ".csv")

    else:
        edf = pd.read_csv(folder + "/ews_data" +
                          h.incidence_filepath(use_incidence) + ".csv", index_col=0)

    # Get data points used for training (last 10 years)
    ews_data = edf[edf.Time.between(10 * 365, 20*365)].copy()
    # Drop NAs
    ews_data = ews_data[~ews_data.isin([np.inf, -np.inf, np.nan]).any(axis=1)]
    ews_data = ews_data[~ews_data[signals].isna().any(axis=1)]

    # Train classifier
    coefs, lr_clf = elr.ews_logistic_regression(ews_data[ews_data["Time"] > 10],
                                                standardise=True,
                                                signals=signals, do_pca=False,
                                                penalty="l1", C=c_min,
                                                solver="liblinear")
    c2 = pd.Series(coefs)
    print(coefs)
    ews_data["all"] = ews_data[signals].dot(c2[signals]) \
                        + c2["intercept"]


    # Calculate AUC through time
    auc_dict = {signal: ews_data.groupby("Time").apply(
                            lambda x: roc_auc_score(x["is_test"], x[signal]))
                for signal in signals + ["timeseries"]}
    auc_dict["all"] = ews_data.groupby("Time").apply(
        lambda x: roc_auc_score(x["is_test"], x["all"]))

    # ROC for each individual EWS and all combined
    fpr_dict = {}
    tpr_dict = {}
    thresh_dict = {}
    roc_auc_dict = {}
    df_c_dict = {}
    for signal in signals + ["timeseries", "all"]:
        y_score = ews_data[signal].values
        y_true = ews_data["is_test"]
        fpr_dict[signal], tpr_dict[signal], thresh_dict[signal] = \
            metrics.roc_curve(y_true, y_score, pos_label=1)
        roc_auc_dict[signal] = roc_auc_score(y_true, y_score)
        df_c_dict[signal] = thresh_dict[signal][np.argmin(fpr_dict[signal]
                                                          - tpr_dict[signal])]

    # Write df_c_dict, coefs and auc_dict to files:
    pd.Series(df_c_dict).to_csv(folder + "/optimum_thresholds" +
                                h.incidence_filepath(use_incidence) + ".csv")

    pd.Series(coefs).to_csv(folder + "/ews_weights" +
                            h.incidence_filepath(use_incidence) + ".csv")

    pd.DataFrame(auc_dict).to_csv(folder + "/auc_time_series" +
                                  h.incidence_filepath(use_incidence) + ".csv")

    if save_df:
        ews_data.to_csv(folder + "/ews_data_test" +
                   h.incidence_filepath(use_incidence) + ".csv")

