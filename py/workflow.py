import py.classifier_training as ct
import py.cross_validation as cv
import py.latin_hypercube_simulator as lhs
import py.helper as h



f_root = "./data/backup-01-14"
dataset = ["", "-concave", "-convex", "-covar"]
curvature = [None, "concave", "convex", None]
include_covariates = [False, False, False, True]
seeds = [8010, 6424, 4512, 7656]



lhs_space = h.lhs_space()

for i, fds in enumerate(dataset):
    f = f_root + fds
    lhs.do_lhs_simulations(h.simulator_args(folder=f), lhs_space,
                           simulator="./SEIR-Simulator-0.2.5/"
                                     "seir_simulator_gamma",
                           include_covariates=include_covariates[i],
                           curvature=curvature[i], random_seed=seeds[i],
                           n_samples=10000)

    for ui in [0, 1]:
        if not (ui == 0 and i > 0):
            cv.run_cross_validation(use_incidence=ui, folder=f,
                                     aggregation_period=4)
            ct.run_classifier_training(use_incidence=ui, use_cross_val=True,
                                       calculate_ews=True,
                                       folder=f, save_df=True,
                                       aggregation_period=4,
                                       )
            print(f)

