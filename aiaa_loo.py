import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic

from aiaa_utils import get_dataset, test_interpolation
from aiaa_utils import plot_loo, plot_comparison_loo, split_output_input, plot_changing_error


def test_loo(inputs, outputs, model, speeds_test, num_samples, speeds=[]):
    # GET DATASET
    dataset = get_dataset(num_samples=num_samples, speeds=speeds)
    # EXECUTE TEST SEQUENCE
    df_mape = all_df_test_pred = pd.DataFrame([], columns=outputs)
    for speed in speeds_test:
        df_train, df_test = split_output_input(dataset, speeds_test=[speed])
        df_test_pred, list_mape = test_interpolation(inputs, outputs, model, df_train, df_test)
        df_mape = df_mape.append(pd.DataFrame([list_mape], columns=outputs))
        all_df_test_pred = pd.concat([all_df_test_pred, df_test_pred], sort=False)

    return df_mape, all_df_test_pred


############## MODELS ##############
svr = SVR(kernel='rbf', C=200, gamma="scale", epsilon=1e-10)
svr2 = SVR(kernel='rbf', C=200, epsilon=1e-10)
svr3 = SVR(kernel='rbf', C=20, gamma="scale", epsilon=1e-10)
svr4 = SVR(kernel='rbf', C=20, epsilon=1e-10)

gpr_rbf = GaussianProcessRegressor(kernel=RBF(length_scale_bounds=(3, 10)))
gpr_matern = GaussianProcessRegressor(kernel=Matern(length_scale_bounds=(3, 10), nu=5, length_scale=0.5))
gpr_quad = GaussianProcessRegressor(kernel=RationalQuadratic(length_scale_bounds=(3, 10), alpha_bounds=(3, 10)))

############## TRAINING SPECIFICATIONS ##############
outputs = ['Pressure_ratio', 'Polytropic_efficiency']
inputs = ['speeds_scaled', 'mass_flow_scaled']
speeds = np.arange(10000, 100000, 10000)

############## TEST SIMPLE LOO ##############
# num_samples = 25
# models = [svr, gpr_matern]
# list_results_models = []
# for model in models:
#     mape_results, list_df_test_pred = test_loo(inputs, outputs, model, speeds, num_samples)
#     print('MAPE mean: ', np.mean(mape_results.values))
#     list_results_models.append(mape_results)
# dict_legend = {'title': 'Models', 'tested_features': ['SVR', 'GPR']}
# plot_loo(list_results_models, speeds, outputs, dict_legend)

############## TEST KERNELS ##############
# num_samples = 25
# kernels = [gpr_rbf] #, gpr_matern
# speed = 30000
# list_results_kernels = []
# for kernel in kernels:
#     mape_results, list_df_test_pred = test_loo(inputs, outputs, kernel, speeds, num_samples)
#     print('MAPE mean: ', np.mean(mape_results.values))
#     list_results_kernels.append(list_df_test_pred)
#
# dict_legend = {'title': '', 'tested_features': ['True Values', 'RBF', 'Matern']}
# plot_comparison_loo(list_results_kernels, speed, outputs, dict_legend)


############## TEST SVR FEATURES ##############
num_samples = 25
speed = 30000
models = [svr, svr2, svr3, svr4]
list_results_models, list_results_structure = [], []
for model in models:
    mape_results, list_df_test_pred = test_loo(inputs, outputs, model, speeds, num_samples)
    print('MAPE mean: ', np.mean(mape_results.values))
    list_results_models.append(mape_results)
    list_results_structure.append(list_df_test_pred)
dict_legend = {'title': '', 'tested_features': ['C=200, scale', 'C=200, 1/n_features',
                                                'C=20, scale', 'C=20, 1/n_features']}
plot_loo(list_results_models, speeds, outputs, dict_legend)
dict_legend = {'title': '', 'tested_features': ['True Values', 'C=200, scale', 'C=200, 1/n_features',
                                                'C=20, scale', 'C=20, 1/n_features']}
plot_comparison_loo(list_results_structure, speed, outputs, dict_legend)

############## TEST NUM SAMPLES ##############
# list_num_samples = [5, 15, 20, 25]
# speed = 50000
# speeds = [20000, 30000, 40000, 50000, 60000, 70000]
# list_results_num_samples, list_mape_results = [], []
# for num_samples in list_num_samples:
#     mape_results, list_df_test_pred = test_loo(inputs, outputs, svr, speeds, num_samples, speeds=speeds)
#     list_results_num_samples.append(list_df_test_pred)
#     list_mape_results.append(mape_results)
#
# dict_legend = {'title': '', 'tested_features': ['True Values', '25', '5']}
# # plot_comparison_loo(list_results_num_samples, speed, outputs, dict_legend)
# plot_changing_error(list_mape_results, list_num_samples, outputs)

############## TEST NUM TRAINING ##############
# speed = [50000]
# all_speeds = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000]
# num_iterations = [1, 2, 3, 4, 5, 6]
# num_samples = 25
# list_results_num_samples, list_mape_results = [], []
# for num_iteration in num_iterations:
#     if num_iteration % 2:
#         all_speeds = all_speeds[:-1]
#     else:
#         all_speeds = all_speeds[1:]
#     print('SPEEDS', all_speeds)
#     mape_results, list_df_test_pred = test_loo(inputs, outputs, svr, speed, num_samples, speeds=all_speeds)
#     list_results_num_samples.append(list_df_test_pred)
#     list_mape_results.append(mape_results)
#
# dict_legend = {'title': '', 'tested_features': ['True Values', '25', '5']}
# plot_changing_error(list_mape_results, num_iterations, outputs)

############## TEST COMPARISON MODELS ##############
# num_samples = 25
# list_models = [svr, gpr_matern]
# speed = 30000
# list_results_models = []
# for model in list_models:
#     _, list_df_test_pred = test_loo(inputs, outputs, model, speeds, num_samples)
#     list_results_models.append(list_df_test_pred)
#
# dict_legend = {'title': '', 'tested_features': ['True Values', 'SVR', 'GPR']}
# plot_comparison_loo(list_results_models, speed, outputs, dict_legend)
