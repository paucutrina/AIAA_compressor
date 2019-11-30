from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVR

from aiaa_utils import get_dataset, test_interpolation, plot_predictions


############## MODELS ##############
gpr = GaussianProcessRegressor(kernel=RBF(length_scale_bounds=(3, 5)), alpha=1e-10, random_state=0)
svr = SVR(kernel='rbf', C=200, gamma='scale', epsilon=1e-10)

############## TRAINING SPECIFICATIONS ##############
outputs = ['Pressure_ratio', 'Polytropic_efficiency']
inputs = ['speeds_scaled', 'mass_flow_scaled']
speeds_test = [30000]
num_samples = 10
models = [svr, gpr]

dataset = get_dataset(num_samples=num_samples)
for model in models:
    df_train, df_test_pred, list_mape = test_interpolation(inputs, outputs, dataset, speeds_test, model)
    dict_legend = {'tested_features': ['Train values', 'Test values', 'Predicted values']}
    plot_predictions(df_train, df_test_pred, speeds_test, outputs, dict_legend)



