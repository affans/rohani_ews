import numpy as np
import pandas as pd
from . import movingwindow as ts
#import pyximport; pyximport.install()
from . import kolmogorov_complexity
from . import entropy


def get_ews(x,windowsize, ac_lag, se = True, kc = True, method="old", mv_method="uniform"):
    '''
    Calculate early-warning signals from time series data.
    :param x: time series data (either 1-dimensional numpy array or list)
    :param windowsize: integer number of time points in moving window
    :param ac_lag: integer lag used to calculated the autocovariance, autocorrelation and decay time
    :param se: Boolean. If True calculate the moving Shannon entropy. This is significantly slower
    :param kc: Boolean. If True calculate the moving Kolmogorov complexity. This is around 10x slower
    :return: dict containing all early-warning signals and original time series
    '''

    if method == "old":
        y = np.array(x, dtype = 'float')

        # Mean:
        mu = ts.MovingWindowAverage(y, windowsize)
        # Variance:
        var = ts.MovingWindowAverage((y-mu)**2, windowsize)
        # Autocovariance
        cm = y-mu
        cm_lag = np.concatenate([np.zeros(ac_lag), cm[:-ac_lag]])
        var_lag = np.concatenate([np.zeros(ac_lag), var[:-ac_lag]])
        acov = ts.MovingWindowAverage(cm*cm_lag, windowsize)

        with np.errstate(divide='ignore', invalid='ignore'):
            # Autocorrelation:
            ac = acov / np.sqrt(var * var_lag)
            # Coefficient of variation:
            cov = np.sqrt(var)/mu
            # Index of dispersion:
            iod = var/mu
            # Correlation time:
            ct = -ac_lag/np.log(abs(ac))
            # Skewness:
            skew = ts.MovingWindowAverage(cm**3, windowsize)/(var**(3/2))
            # Kurtosis
            kurtosis = ts.MovingWindowAverage(cm**4, windowsize)/(var**2)

        out = {"timeseries": y, "mean": mu, "variance": var, 'coefficient_of_variation': cov,
               "index_of_dispersion": iod, "autocorrelation": ac, "decay_time": ct,
               "skewness": skew,
               "kurtosis": kurtosis, "autocovariance": acov}

    if method=="new":
        def mvw(x, w, weight="exp"):
            if weight == "exp":
                return x.ewm(halflife=w).mean()
            if weight == "uniform":
                return x.rolling(w).mean()
        w = windowsize

        y = pd.Series(x, dtype = 'float')


        out = pd.DataFrame({"timeseries": y})
        out["mean"] = mvw(y, w, mv_method)
        out["variance"] = mvw((y - out["mean"]) ** 2, w,
                            mv_method)  # ((y - y.ewm(halflife=w).mean()) ** 2).ewm(halflife=w).mean()
        out["standard_deviation"] = np.sqrt(out["variance"])

        out["coefficient_of_variation"] = out["standard_deviation"]/out["mean"]
        out["index_of_dispersion"] = out["variance"]/out["mean"]
        out["skewness"] = mvw((y - out["mean"]) ** 3, w, mv_method) / (
            out["variance"] ** (3 / 2))
        out["kurtosis"] = mvw((y - out["mean"]) ** 4, w, mv_method) / (
        out["variance"] ** (2))
        out["autocovariance"] = mvw((y - out["mean"]) * ((y - out["mean"]).shift(ac_lag)),
                                  w, mv_method)  # .ewm(w).mean()
        out["autocorrelation"] = out["autocovariance"] / (
        out["standard_deviation"] * out["standard_deviation"].shift(ac_lag))
        #out["acov2"] = mvw((y - out["mean"]) * ((y - out["mean"]).shift(2)), w,
        #                 mv_method)
        #out["ac2"] = out["acov2"] / (
        #    out["standard_deviation"] * out["standard_deviation"].shift(2))
        out["decay_time"] = -ac_lag / np.log(np.abs(out["autocorrelation"]))

        mu = out["mean"]


    # Kolmogorov complexity:
    if(kc == True):
        out["Kolmogorov_complexity"] = kolmogorov_complexity.CMovingKC(x,mu, windowsize)
        
    # Shannon entropy
    if(se == True):
        with np.errstate(divide='ignore', invalid='ignore'):
            out["Shannon_entropy"] = entropy.MovingEntropy(x, windowsize)


    return(out)



def output_csv(data,filename):
    import pandas
    d = pandas.DataFrame(data)
    d.to_csv(filename)
