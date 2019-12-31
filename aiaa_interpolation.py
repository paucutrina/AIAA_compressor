from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.svm import SVR

from aiaa_utils import get_dataset, test_interpolation, plot_predictions, split_output_input, plot_pca
from aiaa_utils import plot_scatter_interpolation, plot_specific_prediction, plot_test_speed

############## MODELS ##############
gpr = GaussianProcessRegressor(kernel=Matern(length_scale_bounds=(3, 10), nu=5, length_scale=0.5))
svr = SVR(kernel='rbf', C=200, gamma=0.5, epsilon=1e-10)

############## TRAINING SPECIFICATIONS ##############
outputs = ['Pressure_ratio', 'Polytropic_efficiency']
inputs = ['speeds_scaled', 'mass_flow_scaled']
speeds_test = [70000]
num_samples = 25
models = [svr]
extrapolation = True

dataset = get_dataset(num_samples=num_samples)
for model in models:
    df_train, df_test = split_output_input(dataset, speeds_test=speeds_test, extrapolation=extrapolation)
    df_test_pred, list_mape = test_interpolation(inputs, outputs, model, df_train, df_test)
    # plot_pca(dataset[inputs])

    if extrapolation:
        # dict_legend = {'tested_features': ['Train values', 'Test values', 'Predicted values']}
        # plot_predictions(df_train, df_test_pred, speeds_test, outputs, dict_legend)
        # dict_legend = {'tested_features': ['Test values', 'Predicted values']}
        # plot_specific_prediction(df_test_pred, speeds_test, outputs, dict_legend)
        # dict_legend = {'tested_features': ['Test values']}
        # plot_specific_prediction(df_test_pred, speeds_test, outputs, dict_legend)
        dict_legend = {'tested_features': ['Test values']}
        plot_test_speed(df_test_pred, outputs)

    plot_scatter_interpolation(df_train, df_test_pred, outputs)



