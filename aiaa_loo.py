import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic

from aiaa_utils import get_dataset, test_interpolation
from aiaa_utils import plot_loo, plot_comparison_loo


def test_loo(inputs, outputs, model, speeds_test, num_samples):
    # GET DATASET
    dataset = get_dataset(num_samples=num_samples)
    # EXECUTE TEST SEQUENCE
    df_mape = all_df_test_pred = pd.DataFrame([], columns=outputs)
    for speed in speeds_test:
        df_train, df_test_pred, list_mape = test_interpolation(inputs, outputs, dataset, [speed], model)
        df_mape = df_mape.append(pd.DataFrame([list_mape], columns=outputs))
        all_df_test_pred = pd.concat([all_df_test_pred, df_test_pred], sort=False)

    return df_mape, all_df_test_pred


############## MODELS ##############
svr = SVR(kernel='rbf', C=200, gamma='scale', epsilon=1e-10)
gpr_rbf = GaussianProcessRegressor(kernel=RBF(length_scale_bounds=(3, 10)))
gpr_matern = GaussianProcessRegressor(kernel=Matern(length_scale_bounds=(3, 10), nu=5, length_scale=0.5))
gpr_quad = GaussianProcessRegressor(kernel=RationalQuadratic(length_scale_bounds=(3, 10), alpha_bounds=(3, 10)))

############## TRAINING SPECIFICATIONS ##############
outputs = ['Pressure_ratio', 'Polytropic_efficiency']
inputs = ['speeds_scaled', 'mass_flow_scaled']
speeds = np.arange(10000, 100000, 10000)

############## TEST SIMPLE LOO ##############
# num_samples = 5
# models = [svr, gpr_rbf]
# list_results_models = []
# for model in models:
#     mape_results, list_df_test_pred = test_loo(inputs, outputs, model, speeds, num_samples)
#     print('MAPE mean: ', np.mean(mape_results.values))
#     list_results_models.append(mape_results)
# dict_legend = {'title': 'Models', 'tested_features': ['SVR', 'GPR']}
# plot_loo(list_results_models, speeds, outputs, dict_legend)

############## TEST KERNELS ##############
# num_samples = 5
# kernels = [gpr_rbf, gpr_matern, gpr_quad]
# speed = 30000
# list_results_kernels = []
# for kernel in kernels:
#     _, list_df_test_pred = test_loo(inputs, outputs, kernel, speeds, num_samples)
#     list_results_kernels.append(list_df_test_pred)
#
# dict_legend = {'title': '', 'tested_features': ['True Values', 'RBF', 'Matern', 'RQ']}
# plot_comparison_loo(list_results_kernels, speed, outputs, dict_legend)


############## TEST NUM SAMPLES ##############
list_num_samples = [5, 10, 25]
speed = 30000
list_results_num_samples = []
for num_samples in list_num_samples:
    _, list_df_test_pred = test_loo(inputs, outputs, gpr_matern, speeds, num_samples)
    list_results_num_samples.append(list_df_test_pred)

dict_legend = {'title': '', 'tested_features': ['True Values', '5', '10', '25']}
plot_comparison_loo(list_results_num_samples, speed, outputs, dict_legend)

############## TEST COMPARISON MODELS ##############
num_samples = 5
list_models = [svr, gpr_matern]
speed = 30000
list_results_models = []
for model in list_models:
    _, list_df_test_pred = test_loo(inputs, outputs, model, speeds, num_samples)
    list_results_models.append(list_df_test_pred)

dict_legend = {'title': '', 'tested_features': ['True Values', 'SVR', 'GPR']}
plot_comparison_loo(list_results_models, speed, outputs, dict_legend)
