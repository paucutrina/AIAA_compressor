import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

from aiaa_utils import get_dataset, LearningCurve

"""
Cross validation: 5
"""

############## TRAINING SPECIFICATIONS ##############
outputs = ['Pressure_ratio']
inputs = ['speeds_scaled', 'mass_flow_scaled']
num_samples = 25
num_simulations = 5
start_num_test = 40

############## MODELS ##############
gpr = GaussianProcessRegressor(kernel=RBF(length_scale_bounds=(3, 5)), alpha=1e-10, random_state=0)
svr = SVR(kernel='rbf', C=200, gamma='scale', epsilon=1e-10)
model = MultiOutputRegressor(gpr)

############## GET DATASET ##############
dataset = get_dataset(num_samples=num_samples)

############## TEST SIZE TO SIMULATE ##############
train_size = np.linspace(start_num_test,
                         len(dataset[inputs]) - len(dataset[inputs]) * 0.2, num=num_simulations).astype(int)

############## RESULTS LEARNING CURVE ##############
learning_curve_model = LearningCurve(model, dataset[inputs], dataset[outputs], 'neg_mean_squared_error', train_size)
learning_curve_model.plotLearningCurve()





