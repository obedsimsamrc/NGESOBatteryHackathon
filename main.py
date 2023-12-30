from data.combine_data import PrepareModel
import pycaret
from pycaret.regression import *
from gplearn.genetic import SymbolicRegressor
import pandas as pd
import os

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)

train_instance = PrepareModel(test_or_train='train')

merged_train_df = train_instance.merge_dataframes(include_dyn_market=True,
                                                  include_gen=False,
                                                  include_day_ahead=False)

train_df = train_instance.add_additional_datetime_features(df=merged_train_df, datetime_col="UTC_Settlement_DateTime")

dx_periods = [("05/08/2022", "31/08/2022"), ("01/11/2022", "30/11/2022"), ("02/02/2023", "28/02/2023"),
              ("01/06/2023", "01/07/2023")]
train_df = train_instance.add_ffr_or_dx_service_identifier(df=train_df, dx_periods=dx_periods)

train_df = train_instance.add_additional_lagged_features(df=train_df, cols=["disp_ffr_percent_sum",
                                                                            "disp_dcl_percent_sum",
                                                                            "disp_dch_percent_sum",
                                                                            ])

train_df = train_instance.remove_extreme_battery_outputs(df=train_df, no_of_stds=4, save_df_as_csv=False)

# cols = ["battery_output", "disp_ffr_percent_sum",
#         "disp_ffr_percent_mean", "disp_ffr_percent_kWh_per_kW_sum", "dayofweek",
#         "in_dx", "f_sum", 'month', 'week_of_year', 'hour', "delta_freq_skew",
#         "disp_dcl_percent_sum", "disp_dch_percent_sum",
#         "disp_dch_percent_std", "disp_dcl_percent_std",
#         "cleared_volume_dcl", "cleared_volume_dch", "disp_dch_percent_mean",
#         ]
# train_df = train_df[cols]
train_df = train_df[(train_df["battery_output"] > 0.03) | (train_df["battery_output"] < 0.022)]

s = setup(train_df, target='battery_output', session_id=123)

# import RegressionExperiment and init the class
exp = RegressionExperiment()

# init setup on exp
exp.setup(train_df, target='battery_output',
          session_id=123,
          # log_experiment=True,
          feature_selection=True,
          remove_outliers=True,
          outliers_threshold=0.2,
          normalize=True,
          # transformation=True,
          # pca=True,
          low_variance_threshold=0.1,
          categorical_features=['year', 'season', 'is_winter', 'month', 'week_of_year', 'day', 'dayofweek', 'hour',
                                'minute', 'is_wknd', 'is_working_hr', 'is_lunch_hr', 'EFA Block Count',
                                'EFA HH Count', 'in_dx'],
          # categorical_features=["in_dx", 'EFA Block Count', 'EFA HH Count'],
          feature_selection_method="classic",
          remove_multicollinearity=True,
          imputation_type="iterative",
          numeric_iterative_imputer="lightgbm",
          train_size=0.70,
          fold_shuffle=True,
          # profile=True,
          # n_features_to_select=8
          )

# compare baseline models
# best = compare_models(sort="MAE",
#                       # include=['lightgbm', 'et', 'rf']
#                       )

# train model
base_lightgbm_model = create_model('lightgbm', bagging_fraction=0.7, bagging_freq=7, feature_fraction=0.5,
                                   min_child_samples=41, min_split_gain=0.3, n_estimators=70,
                                   n_jobs=-1, num_leaves=12, random_state=123, reg_alpha=3, folds=6,
                                   reg_lambda=0.0001, return_train_score=True)
current_tuned_lightgbm_model = create_model('lightgbm', bagging_fraction=0.8120116072779331, bagging_freq=0,
                                            feature_fraction=0.6900876194974394, folds=6,
                                            learning_rate=0.041101820776352826, min_child_samples=47,
                                            min_split_gain=0.7142327352477369, n_estimators=188, n_jobs=-1,
                                            num_leaves=50, random_state=123, reg_alpha=4.586356369488243e-09,
                                            reg_lambda=9.096335249691464e-06, return_train_score=True)
# interpret model
# interpret_model(base_lightgbm_model)

# tune model
tuned_lightgbm = tune_model(base_lightgbm_model, n_iter=50, optimize="MAE", early_stopping=True,
                            choose_better=True, return_train_score=True, search_library='optuna')
# print(tuned_lightgbm)

model_plot_selection = current_tuned_lightgbm_model
plot_model(model_plot_selection, plot='feature', save=True)
plot_model(model_plot_selection, plot='error', save=True)
plot_model(model_plot_selection, plot='residuals', save=True)
plot_model(model_plot_selection, plot='learning', save=True)
# plot_model(model_plot_selection, plot='cooks', save=True)

# y_pred = predict_model(lightgbm_model, train_df)
# y_pred = y_pred[["UTC_Settlement_DateTime", "battery_output", "prediction_label"]]
# y_pred.to_csv("check.csv")


test_instance = PrepareModel(test_or_train='test', concat_x_train_rows=47)

merged_test_df = test_instance.merge_dataframes(include_dyn_market=True,
                                                include_gen=False)

merged_test_df["UTC_Settlement_DateTime"] = pd.to_datetime(merged_test_df["UTC_Settlement_DateTime"],
                                                           format="%Y-%m-%d %H:%M:%S")

test_df = test_instance.add_ffr_or_dx_service_identifier(df=merged_test_df, dx_periods=dx_periods)

test_df = test_instance.add_additional_datetime_features(df=test_df, datetime_col="UTC_Settlement_DateTime")

test_df = test_instance.add_additional_lagged_features(df=test_df, cols=["disp_ffr_percent_sum",
                                                                         "disp_dcl_percent_sum",
                                                                         "disp_dch_percent_sum",
                                                                         ])

test_df = test_instance.remove_extreme_battery_outputs(df=test_df, no_of_stds=0, save_df_as_csv=False)

# Assuming 'test_df' is your DataFrame and 'cols' is the list of column names
# cols_to_exclude = ["battery_output"]
# # Using list comprehension to select columns not in 'cols_to_exclude'
# test_df = test_df[[col for col in cols if col not in cols_to_exclude]]

predicted_df = predict_model(current_tuned_lightgbm_model, data=test_df)

predicted_df = predicted_df[["prediction_label"]]
predicted_df.rename(columns={"prediction_label": "battery_output",
                             "index": "id"}, inplace=True)
predicted_df.index = range(39457, 39457 + len(predicted_df))

# predicted_df.to_csv(os.path.join(os.path.dirname(__file__), "results", "predicted_results_5.csv").replace("\\", "/"))
