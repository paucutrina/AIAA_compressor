import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from matplotlib import lines
import matplotlib.pyplot as plt
import copy
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import learning_curve
from aiaa_constants import Cte


# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.linear_model import BayesianRidge
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import GradientBoostingRegressor  # SVM regressor
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.ensemble import AdaBoostRegressor


def mean_absolute_percentage_error(y_true, y_pred):
    lista = [(abs(y_true[num_pred] - pred)) / y_true[num_pred] for num_pred, pred in enumerate(y_pred)]
    mape = np.mean(lista) * 100
    return mape


def coefficient_determination(y_true, y_pred):
    # sum of square of residuals
    ssr = np.sum((y_true - y_pred) ** 2)
    #  total sum of squares
    mean_true = np.mean(y_true)
    sst = np.sum((y_true - mean_true) ** 2)
    # R2 score
    r2_score = 1 - (ssr / sst)
    return r2_score


def getError(test_y, pred_y):
    metrics = ['mape', 'r2', 'rmse']
    mape = mean_absolute_percentage_error(test_y, pred_y)
    rmse = np.sqrt(mean_squared_error(test_y, pred_y))
    r2 = coefficient_determination(test_y, pred_y)  # r2 = r2_score(test_y, pred_y)
    return pd.DataFrame([[mape, r2, rmse]], columns=metrics)


def get_dataset(num_samples):
    # IMPORT DATA
    dataset_original = pd.read_csv("data/small_dataset.csv", delimiter=",")
    dataset_original = dataset_original.drop(columns=['Unnamed: 0'])
    new_dataset = pd.DataFrame([], columns=['Rotational_Speed', 'Mass_flow', 'Pressure_ratio', 'Polytropic_efficiency'])
    # DATA REDUCTION
    speeds = np.arange(10000, 100000, 10000)
    for num_speed, speed in enumerate(speeds):
        speed_data = dataset_original.loc[dataset_original['Rotational_Speed'] == speed]
        jump = int(len(speed_data) / num_samples)
        idx_desired_rows = np.arange(1 + int(num_speed * len(speed_data)),
                                     len(speed_data) + int(num_speed * len(speed_data)), jump)
        reduced_data = speed_data.loc[idx_desired_rows]
        new_dataset = pd.concat([new_dataset, reduced_data])
    new_dataset = new_dataset.reset_index(drop=True)
    # NORMALIZATION DATA
    scaler = MinMaxScaler()
    input_data = pd.DataFrame(scaler.fit_transform(new_dataset[['Rotational_Speed', 'Mass_flow']]),
                              columns=['speeds_scaled', 'mass_flow_scaled'])
    clean_data = pd.concat([input_data, new_dataset[['Rotational_Speed']]], axis=1)
    clean_data = pd.concat([clean_data, new_dataset[['Pressure_ratio', 'Polytropic_efficiency']]], axis=1)
    clean_data = clean_data.sample(frac=1)
    return clean_data


def test_interpolation(inputs, outputs, dataset, speeds_test, model):
    # MULTI-OUTPUT MODEL
    if len(inputs) > 1:
        model = MultiOutputRegressor(model)
    # SPLIT INTO TRAIN AND TEST
    df_test = dataset[dataset['Rotational_Speed'].isin(speeds_test)].reset_index()
    df_train = dataset[~dataset['Rotational_Speed'].isin(speeds_test)].reset_index()
    # MODEL FIT
    model = model.fit(df_train[inputs], df_train[outputs])
    # PREDICTION
    pred_y = model.predict(df_test[inputs])
    # EVALUATION
    df_predict = pd.DataFrame(pred_y, columns=outputs)
    list_mape = []
    for output in outputs:
        results_pres = getError(df_test[output].values, df_predict[output].values)
        print(results_pres)
        list_mape.append(results_pres.mape.values[0])
        # plot_predictions(output, df_train, df_test, df_predict, [speed])
        # plot_predicted_true(df_test, df_predict, output)
    df_renamed_predict = df_predict.rename({'Pressure_ratio': 'Pressure_ratio_pred',
                                            'Polytropic_efficiency': 'Polytropic_efficiency_pred'}, axis='columns')
    df_test_pred = pd.concat([df_test, df_renamed_predict], axis='columns')
    return df_train, df_test_pred, list_mape


def sort_test_predicted_to_plot(df_test, df_predict):
    current_test = copy.copy(df_test).sort_values('mass_flow_scaled')
    current_pred = df_predict.reindex(list(current_test.index.values))
    return current_test, current_pred


def add_train_in_plot(ax, x, y, plot_prop, idx_plot):
    ax.plot(x, y, plot_prop['line_style'] + plot_prop['list_markers'][idx_plot], c=plot_prop['list_colors'][idx_plot],
            linewidth=plot_prop['line_width'], markerfacecolor=plot_prop['marker_color'], markersize=3)
    return ax


def plot_predictions(train, df_test_pred, speeds_test, outputs, dict_legend):
    speeds_train = train.Rotational_Speed.unique().tolist()
    plot_prop = {'line_style': '--', 'line_width': 0.4, 'font_size': 14,
                 'marker_color': "None", 'x_feature': 'mass_flow_scaled',
                 'list_colors': ['green', 'red', 'black', 'blue', 'purple', 'brown'],
                 'list_markers': ['^', 's', 'o', '*', 'X', 'D']}

    output_pred = [output_name + Cte.IDX_PRED for output_name in outputs]

    if 'tested_features' in dict_legend.keys():
        for num_output, output in enumerate(outputs):
            fig, ax = plt.subplots()
            if 'Train values' in dict_legend['tested_features']:
                for train_speed in speeds_train:
                    current_train = train[train['Rotational_Speed'] == train_speed].sort_values(plot_prop['x_feature'])
                    ax = add_train_in_plot(ax, current_train[plot_prop['x_feature']], current_train[output],
                                           plot_prop, idx_plot=0)

            for speed in speeds_test:
                df_test_pred_speed = df_test_pred.loc[df_test_pred['Rotational_Speed'] == speed]
                # Sort df to plot the results
                df_test_pred_speed = copy.copy(df_test_pred_speed).sort_values(plot_prop['x_feature'])
                if 'Test values' in dict_legend['tested_features']:
                    ax = add_train_in_plot(ax, df_test_pred_speed[plot_prop['x_feature']], df_test_pred_speed[output],
                                           plot_prop, idx_plot=1)
                if 'Predicted values' in dict_legend['tested_features']:
                    ax = add_train_in_plot(ax, df_test_pred_speed[plot_prop['x_feature']],
                                           df_test_pred_speed[output_pred[num_output]],
                                           plot_prop, idx_plot=2)

            ax.set_xlabel(plot_prop['x_feature'].replace('_', ' ').capitalize(), fontsize=plot_prop['font_size'])
            ax.set_ylabel(output.replace('_', ' '), fontsize=plot_prop['font_size'])
            handles = [lines.Line2D([0], [0], marker=plot_prop['list_markers'][num_feature], ls=plot_prop['line_style'],
                                    c=plot_prop['list_colors'][num_feature], markerfacecolor=plot_prop['marker_color'])
                       for num_feature in range(len(dict_legend['tested_features']))]
            ax.legend(handles, dict_legend['tested_features'], loc='best')
            plt.show()


class LearningCurve:
    def __init__(self, clf, X, Y, score, train_size):
        self.clf = clf
        self.X = X
        self.Y = Y
        self.score = score
        self.train_size = train_size

    def plotLearningCurve(self):
        train_sizes, train_scores, test_scores = \
            learning_curve(self.clf, self.X, self.Y,
                           train_sizes=self.train_size,
                           scoring=self.score, cv=5, shuffle=True, random_state=10)  # !!cv=5

        train_scores_mean = -train_scores.mean(1)
        test_scores_mean = -test_scores.mean(1)
        if self.score == 'neg_mean_squared_error':
            gap = test_scores_mean - train_scores_mean
        else:
            gap = None

        plt.figure()
        plt.plot(self.train_size, np.sqrt(test_scores_mean), '-o', markerfacecolor="None", label=' Test')
        plt.plot(self.train_size, np.sqrt(train_scores_mean), '-s', markerfacecolor="None", label=' Train')

        plt.xlabel("Number of training samples", fontsize=14)
        plt.ylabel('RMSE', fontsize=14)
        plt.legend(title='Simulation', loc="best")
        plt.show()
        return gap, train_sizes

    def plotGap(self, gap):
        plt.figure()
        plt.plot(self.train_size, gap, '-v', markerfacecolor="None", label='GAP Test-Train')
        plt.xlabel("Train size")
        plt.legend(loc="best")
        plt.ylabel('MSE')
        plt.show()


def plot_loo(list_df_mape, speeds, outputs, dict_legend):
    markers = ['--s', '--o']
    colors = ['red', 'black']

    for num_out, output in enumerate(outputs):
        plt.figure()
        for num_feature, df_mape in enumerate(list_df_mape):
            plt.plot(speeds, df_mape[output].values, markers[num_feature], c=colors[num_feature], linewidth=1,
                     markerfacecolor="None", label=dict_legend['tested_features'][num_feature])
        plt.title(output.replace('_', ' '))
        plt.ylabel('MAPE [%]', fontsize=14)
        plt.xlabel('Speed [rpm]', fontsize=14)
        plt.legend(title=dict_legend['title'])
        plt.show()


def plot_predicted_true(df_test, df_pred, output):
    current_test, current_pred = sort_test_predicted_to_plot(df_test, df_pred)

    plt.figure()
    plt.plot(current_test[output], current_pred[output], '--s', c='black', linewidth=1, markerfacecolor="None")
    plt.ylabel('Prediction ' + output.replace('_', ' '), fontsize=14)
    plt.xlabel('Real Value ' + output.replace('_', ' '), fontsize=14)

    minimal = min(min(current_test[output].values), min(current_pred[output].values))
    maximal = max(max(current_test[output].values), max(current_pred[output].values))
    plt.xlim(minimal, maximal)
    plt.ylim(minimal, maximal)

    plt.show()


def plot_comparison_loo(list_test_pred, speed, outputs, dict_legend):
    output_pred = [output_name + '_pred' for output_name in outputs]
    list_colors = ['black', 'blue', 'red', 'green', 'purple', 'brown']
    list_markers = ['*', 's', 'o', '^', 'X', 'D']
    line_style = '--'
    x_feature = 'mass_flow_scaled'
    line_width = 1
    marker_face_color = "None"
    font_size = 14
    for num_output, output in enumerate(outputs):
        true_value_plotted = False
        fig, ax = plt.subplots()
        for num_test, df_results in enumerate(list_test_pred):
            # We study just one speed
            df_result_speed = df_results.loc[df_results['Rotational_Speed'] == speed]
            # Sort df to plot the results
            df_result_speed = copy.copy(df_result_speed).sort_values(x_feature)
            # Plot true values
            if not true_value_plotted:
                ax.plot(df_result_speed[x_feature], df_result_speed[output],
                        line_style + list_markers[0], c=list_colors[0], linewidth=line_width,
                        markerfacecolor=marker_face_color)
                true_value_plotted = True
            # Plot predicted values
            ax.plot(df_result_speed[x_feature], df_result_speed[output_pred[num_output]],
                    line_style + list_markers[num_test + 1], c=list_colors[num_test + 1],
                    linewidth=line_width, markerfacecolor=marker_face_color)

        if 'tested_features' in dict_legend.keys() and 'title' in dict_legend.keys():
            ax.legend(dict_legend['tested_features'], loc='best', title=dict_legend['title'])
        ax.set_ylabel(output.replace('_', ' '), fontsize=font_size)
        ax.set_xlabel(x_feature.replace('_', ' ').capitalize(), fontsize=font_size)
        plt.show()
