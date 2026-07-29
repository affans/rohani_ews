# List of functions called in other files
import pandas as pd
import ews
from joblib import Parallel, delayed
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats


# Colour palette for plots

pal = ["#11a1b7", "#11b74c", "#f75a5b", "#a64ca6", "#f79e54","#ff8fad",  "#fee9ce"]

long_pal = ["#6ED228",
"#A028BE",
"#ED1E79",
"#FAAA00",
"#00468C",
"#F15A24",
"#28D2FA",
"#008CF0",
"#0A783C",
"#FD84FF",
"#961414"]


plot_colours = {"axis": "#737e83",
                "axis_text": "#47555c"}

#long_pal = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c",
#            "#fdbf6f", "#ff7f00", "#cab2d6"]

# Signals used in analysis
def signals():
    return ["mean", "index_of_dispersion", "autocorrelation",
            "coefficient_of_variation", "kurtosis", "skewness",
            "standard_deviation", "ac2"]  # , "sd_convexity"]


# Unique colour for each EWS
ews_colours = dict(zip(["Best fit to simulated data", "Best fit to pertussis"] +
                       signals(), long_pal))


def sig_labels():
    return {"ac2": "Autocorrelation (lag 2)",
            "autocorrelation": "Autocorrelation (lag 1)",
            "mean": "Mean",
            "index_of_dispersion": "Index of dispersion",
            "coefficient_of_variation": "Coefficient of variation",
            "standard_deviation": "Standard deviation",
            "skewness": "Skewness",
            "kurtosis": "Kurtosis",
            "sd_convexity": "SD convexity"}

def simulator_args(folder, use_pertussis=False):
    p = {"folder": folder,
         "rn": "epi_data",
         "N_i": 1e6,
         "N_f": 1e6,
         "N_rs": 10,
         "N_rd": 10,
         "v_i": 0.0,
         "v_f": 0.0,
         "v_rs": 10,
         "v_rd": 10,
         "R0_i": 0.5,
         "R0_f": 0.5,
         "R0_rs": 10,
         "R0_rd": 10.,
         "eta_i": 1 / 7,
         "eta_f": 1. / 7,
         "eta_rs": 10,
         "eta_rd": 10.,
         "rp_i": 0.1,
         "rp_f": 0.1,
         "rp_rs": 10,
         "rp_rd": 10,
         "ts": 7,
         "force": "none",
         "T": 20,
         "runs": 1,
         "rho": 0.0769,  # mumps
         "gamma": 0.1667,  # mumps
         "sa": 0.3,
         "rdis": 100,
         "seed": 1,
         # OU parameters:
         "R0_ramp": "bb",  # options: linear, fixed, bb or ou
         "bb_a": 1,
         "bb_v": 0.00001,
         "ou_d": 0.01,
         "ou_v": 0.00001,
         "ou_l": 0,
         "ou_u": 1}

    if use_pertussis:
        #  pertussis Rohani, P, Zhong, X, & King, A. A. (2010) Science
        p["rho"] = 0.125
        p["gamma"] = 0.0667
    return p

def lhs_space():
    """
    Set up parameter space for latin hypercube sampling
    :return:
    """
    return {"N_i": (5e4, 5.0e6),
            "eta_i": (1/30, 1),
            "rp_i": (0.01, 0.5),
            "R0_i": (0.1, 0.9),
            "bb_a": (1, 5)}


def incidence_filepath(x):
    return "_incidence" if x else ""


def read_cross_val(folder, use_incidence=True):
    """
    Get best hyperparameter combination (windowsize/half-life and penalty
    strength) from cross validation.

    :param folder: containing cross-validation results
    :param use_incidence: (boolean) Whether using case reports or incidence data
    :return: w_min, c_min
    """
    c_before_w = True

    p_performance = pd.read_csv(folder+"/k-fold-cross-validation" +
                                incidence_filepath(use_incidence)+".csv",
                                index_col=0)
    p_performance["test"] = np.sign(p_performance["auc"] + p_performance["std"]
                                    - p_performance["auc"].max()).astype(int)
    if c_before_w:
        # Minimise c_min before w_min
        c_min = p_performance[p_performance["test"] == 1]["p"].min()
        w_min = p_performance[(p_performance["test"] == 1)
                              & (p_performance["p"] == c_min)]["w"].min()
    else:
        # Minimise w_min before c_min
        w_min = p_performance[p_performance["test"] == 1]["w"].min()
        c_min = p_performance[(p_performance["test"] == 1) &
                              (p_performance["w"] == w_min)]["p"].min()

    return w_min, c_min


def get_ews(data, params_df, agg=4, wtime=200, mv_method="exp",
            use_parallel=False, nc=1, use_incidence=False):
    w = wtime//agg

    def mvw(x, weight="exp"):
        if weight == "exp":
            return x.ewm(halflife=w).mean()
        if weight == "uniform":
            return x.rolling(w).mean()

    def single_run_ews(sc):
        i = sc[0]
        g = sc[1]

        x = g.groupby(g.time // (7*agg) * 7*agg)["reported_cases"].sum()\
            .reset_index(drop=True).values

        # Convert to incidence
        if use_incidence:
            x = x/params_df.loc[i[1], "N_i"]

        e = pd.DataFrame(ews.get_ews(x, windowsize=w, ac_lag=1, se=False,
                                     kc=False, method="new",
                                     mv_method=mv_method))

        e["acov2"] = mvw((x-e["mean"])*((x-e["mean"]).shift(2)), mv_method)
        e["ac2"] = e["acov2"] / (
            e["standard_deviation"] * e["standard_deviation"].shift(2))

        e["sd_convexity"] = e["standard_deviation"] - \
            e["standard_deviation"].shift(1)
        e["run"] = i[0]
        e["model"] = i[1]
        e["is_test"] = g["is_test"].iloc[0]
        gg = g.groupby(g.time // (7*agg) * 7*agg)
        e["R0"] = gg["R0"].mean().values
        e["Time"] = gg["time"].mean().values
        return e

    dfg = data.groupby(["run", "model"])

    if use_parallel:
        edf_list = Parallel(n_jobs=nc)(delayed(single_run_ews)(sc)
                                       for sc in dfg)
    else:
        edf_list = [single_run_ews(sc) for sc in dfg]

    edf = pd.concat(edf_list)
    return edf


def lr_decision_function(x, coefs, intercept):
    return np.sum(x*coefs) + intercept


def lr_emergence_risk(df):
    return 1 / (1 + np.exp(-df))


def get_pop_size(state, time, us_dem):
    state_pop = us_dem.get_group(state)
    state_pop.index = state_pop["year"]
    post_time = np.array(
        [np.min([np.floor(i) + 1, state_pop.index.max()]) for i in time])
    pre_time = np.array(
        [np.min([np.floor(i), state_pop.index.max() - 1]) for i in time])
    post = state_pop.loc[post_time, "population.size"] * (time - pre_time)
    post.index = time
    pre = state_pop.loc[pre_time, "population.size"] * (time - post_time)
    pre.index = time
    return post - pre


def read_pertussis_files():
    tdf = pd.read_csv("./data/pertussis.51.12.csv", header=0, sep=",")
    tdf.index = tdf["YEAR"] + tdf["MONTH"] / 12
    states = tdf.columns[2:]  # .str.replace(".", " ")
    tdf = tdf.fillna(0)
    state_names = pd.read_csv("./data/states.csv", index_col=0)
    us_dem = pd.read_csv("./data/journal.pbio.1002172.s004.CSV").groupby(
        "state")
    return tdf, states, state_names, us_dem


def get_lr_pertussis(use_incidence=True, years=(1980, 2000),
                     significance_level=0.05):
    """
    Perform linear regression of log-transform of the US state-level pertussis
    case data with time. Significance of emergence is found by performing a one
    tail t-test for each state.
    :param use_incidence: boolean specifying whether to transform the case
    counts data into incidence, using US state-level population data.
    :param years: tuple specifying the period over which to perform the
    linear regression.
    :param significance_level: signifiance level for the t-test. States with
    p-values below significance_level are labelled emerging.
    :return: Dataframe with the output of the linear regression for each state
    """




    tdf, states, state_names, us_dem = read_pertussis_files()

    def get_lr_single_state(state):

        lr = LinearRegression(fit_intercept=True)

        # Cases
        x = np.log10(tdf[state].copy() + 1)

        if use_incidence:
            npop = get_pop_size(state_names.loc[state.replace(".", " "),
                                                "Abbreviation"],
                                x.index.values, us_dem)
            x = np.log10(1e0*(tdf[state].copy() + 1) / npop)

        x = x.loc[(x.index > years[0]) & (x.index < years[1])]
        y = x.values

        x_train = x.index.values.reshape(-1, 1)
        y_train = y
        lr.fit(x_train, y_train)

        # t-test:
        # sum of square residuals:
        ssr = np.sum((y_train - lr.predict(X=x_train)) ** 2)
        xbar = np.mean(x_train)
        ybar = np.mean(y_train)
        var_x = np.sum((x_train - xbar) ** 2)
        # Standard error
        se = np.sqrt(ssr / (var_x * (len(x_train) - 2)))
        t_score = lr.coef_ / se
        # One tail test
        p_value = (1 - stats.t.cdf(t_score, (len(x_train) - 2)))

        df = pd.DataFrame({"intercept": lr.intercept_,
                           "coef": lr.coef_[0],
                           "p_value": p_value[0],
                           "ybar": ybar,
                           "t_score": t_score}, index=[state])

        df["emerging"] = df["p_value"] < significance_level

        return df

    lr_df = pd.concat(
        [get_lr_single_state(s) for s in states])
    return lr_df
