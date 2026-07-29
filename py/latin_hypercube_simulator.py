import pandas as pd
from subprocess import call
import pyDOE
from joblib import Parallel, delayed
import numpy as np
import os


def do_lhs_simulations(default_params, v_params,
                       simulator="./SEIR-Simulator-0.2.5/seir_simulator_gamma",
                       curvature=None, include_covariates=False,
                       n_samples=5000, random_seed=533, logscale=("N_i"),
                       inverse=("eta_i", "eta_f"), rand_int_max=1048576,
                       num_cores=6):

    # function for bb_a (brownian bridge curvature)
    def f(x, a, b):
        if curvature == "concave":
            return 1 / b + x * (1 / a - 1 / b)
        elif curvature == "convex":
            return a+x*(b-a)
        elif x < 0.5:
            return 1/b+2*x*(1/a-1/b)
        else:
            return a+2*(x-0.5)*(b-a)

    np.random.seed(random_seed)
    output_folder = default_params["folder"]

    if include_covariates:
        v_params["eta_f"] = v_params["eta_i"]
        v_params["rp_f"] = v_params["rp_i"]

    # Create simulation parameters matrix using LHS
    lhs_matrix = pyDOE.lhs(len(v_params), samples=n_samples)
    par_list = []
    for i in range(0, lhs_matrix.shape[0]):
        params = default_params.copy()
        for j, p in enumerate(v_params):
            if p in logscale:
                params[p] = np.exp(np.log(v_params[p][0]) +
                                   lhs_matrix[i, j] * (np.log(v_params[p][1]) -
                                                       np.log(v_params[p][0])))
            elif p in inverse:
                params[p] = 1/(lhs_matrix[i, j]*(1/v_params[p][0] -
                                                 1/v_params[p][1])
                               + 1/v_params[p][1])
            elif p == "bb_a":
                params[p] = f(lhs_matrix[i, j], v_params[p][0], v_params[p][1])
            else:
                params[p] = v_params[p][0] + lhs_matrix[i, j]*(v_params[p][1] -
                                                               v_params[p][0])

        params["N_f"] = params["N_i"]

        if not include_covariates:
            params["eta_f"] = params["eta_i"]
            params["rp_f"] = params["rp_i"]

        params["seed"] = np.random.randint(rand_int_max)

        if i < n_samples/2:
            params["R0_ramp"] = "bb"
            params["R0_f"] = 1
        else:
            params["R0_ramp"] = "ou"
            params["R0_f"] = params["R0_i"]
        params["rn"] = "data_" + str(i)
        par_list = par_list + [pd.DataFrame(params, index=[i])]

    params_df = pd.concat(par_list).sort_index()

    # Set folder for output
    try:
        os.makedirs(output_folder)
    except FileExistsError:
        pass
    params_df.to_csv(output_folder + "/simulation_parameters.csv")

    # Perform simulations for each parameter set using parallel
    def parallel_wrapper(sc):
        row = sc[1]
        index = sc[0]
        pr = row.to_dict()
        pl = [k + '=' + str(pr[k]) for k in pr]
        comm = [simulator] + pl
        call(comm)
        print(index)
        return index

    Parallel(n_jobs=num_cores)(delayed(parallel_wrapper)(sc)
                               for sc in params_df.iterrows())
