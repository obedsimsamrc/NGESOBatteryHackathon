from data.combine_data import PrepareModel
import pycaret
from pycaret.regression import *
from gplearn.genetic import SymbolicRegressor
from ngboost import NGBRegressor
import xgboost as xgb
import pandas as pd
import os
from pprint import pprint

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
                                                                            "delta_freq_sum",
                                                                            "disp_ffr_percent_mean",
                                                                            "disp_ffr_percent_max",
                                                                            "disp_ffr_percent_min",
                                                                            "cleared_volume_dcl",
                                                                            "cleared_volume_dch",
                                                                            ])

train_df = train_instance.remove_extreme_battery_outputs(df=train_df,
                                                         no_of_stds=4,
                                                         save_df_as_csv=False)

train_df = train_df[(train_df["battery_output"] > 0.03) | (train_df["battery_output"] < 0.022)]
train_df = train_df[(train_df["UTC_Settlement_DateTime"] > pd.Timestamp(day=1, month=6, year=2022)) |
                    (train_df["UTC_Settlement_DateTime"] < pd.Timestamp(day=30, month=4, year=2022))]

s = setup(train_df, target='battery_output', session_id=123)

# import RegressionExperiment and init the class
exp = RegressionExperiment()

# init setup on exp
exp.setup(train_df, target='battery_output',
          session_id=123,
          # log_experiment=True,
          feature_selection=True,
          remove_outliers=True,
          outliers_threshold=0.1,
          normalize=True,
          # transformation=True,
          # pca=True,
          low_variance_threshold=0.05,
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
          # n_features_to_select=15
          )

# compare baseline models
# best = compare_models(sort="MAE",
#                       n_select=7,
#                       # include=['lightgbm', 'et', 'rf', 'xgboost']
#                       )

# # train model
base_lightgbm_model = create_model('lightgbm', bagging_fraction=0.6057704349876807, bagging_freq=0,
                                   feature_fraction=0.5073337434152808, folds=6,
                                   learning_rate=0.025770968863290385, min_child_samples=71,
                                   min_split_gain=0.04263967056800268, n_estimators=299, n_jobs=-1,
                                   num_leaves=149, random_state=123,
                                   reg_alpha=1.4572034433187538e-07,
                                   reg_lambda=7.524019133093036e-06)
# current_tuned_lightgbm_model = create_model('lightgbm', bagging_fraction=0.8120116072779331, bagging_freq=0,
#                                             feature_fraction=0.6900876194974394, folds=6,
#                                             learning_rate=0.041101820776352826, min_child_samples=47,
#                                             min_split_gain=0.7142327352477369, n_estimators=200, n_jobs=-1,
#                                             num_leaves=60, random_state=123, reg_alpha=4.586356369488243e-09,
#                                             reg_lambda=9.096335249691464e-06, return_train_score=True)
tuned_xgb_model = create_model('xgboost', colsample_bytree=0.7152711949378981, learning_rate=0.07885051584357601,
                               max_depth=7, min_child_weight=2, n_estimators=270, reg_aplha=1.6787875695108925e-09,
                               reg_lambda=5.793719570818451, scale_pos_weight=39.48401443540499,
                               subsample=0.7157796582340025)

# xgb_model = create_model('xgboost', return_train_score=True)
br_model = create_model('br', return_train_score=True)
rf_model = create_model('rf', return_train_score=True)
lr_model = create_model('lr', return_train_score=True)
# en_model = create_model('en', return_train_score=True)
# ridge_model = create_model('ridge', return_train_score=True)
# lasso_model = create_model('lasso', return_train_score=True)
# huber_model = create_model('huber', return_train_score=True)


# interpret model
# interpret_model(current_tuned_lightgbm_model)
# interpret_model(current_tuned_lightgbm_model, plot='correlation', feature='disp_ffr_percent_sum')
# interpret_model(current_tuned_lightgbm_model, plot='correlation', feature='disp_dch_percent_sum')
# interpret_model(current_tuned_lightgbm_model, plot='correlation', feature='disp_ffr_percent_mean')
# models()

# tune model
# tuned_lightgbm = tune_model(current_tuned_lightgbm_model, n_iter=100, optimize="MAE", early_stopping=True,
#                             choose_better=True, return_train_score=True, search_library='optuna',
#                             # custom_grid={'n_estimators': [100, 200, 300, 400, 500], 'num_leaves': [40, 50, 60, 70]}
#                             )
# print(tuned_lightgbm)
# tuned_xgb = tune_model(xgb_model, n_iter=50, optimize="MAE", early_stopping=True,
#                        choose_better=True, search_library='optuna', return_train_score=True)
# pprint(vars(tuned_xgb))

# boosted_lightgbm = ensemble_model(tuned_lightgbm, optimize="MAE", method='Boosting', choose_better=True,
#                                   return_train_score=True)

# stacked_model_best = stack_models(estimator_list=[xgb_model, base_lightgbm_model, br_model, rf_model, lr_model],
#                                   optimize="MAE", return_train_score=True, meta_model=base_lightgbm_model)
stacked_model = stack_models(estimator_list=[tuned_xgb_model, base_lightgbm_model, br_model, rf_model, lr_model],
                             optimize="MAE", return_train_score=True, meta_model=base_lightgbm_model)

# model_plot_selection = stacked_model
# plot_model(model_plot_selection, plot='error', save=True)
# plot_model(model_plot_selection, plot='residuals', save=True)
# plot_model(model_plot_selection, plot='learning', save=True)
# plot_model(model_plot_selection, plot='feature', save=True)
# plot_model(model_plot_selection, plot='cooks', save=True)

# y_pred = predict_model(current_tuned_lightgbm_model, train_df)
# y_pred = y_pred[["UTC_Settlement_DateTime", "battery_output", "prediction_label"]]
# y_pred.to_csv("check.csv")

# test_instance = PrepareModel(test_or_train='test', concat_x_train_rows=47)
#
# merged_test_df = test_instance.merge_dataframes(include_dyn_market=True,
#                                                 include_gen=False)
#
# merged_test_df["UTC_Settlement_DateTime"] = pd.to_datetime(merged_test_df["UTC_Settlement_DateTime"],
#                                                            format="%Y-%m-%d %H:%M:%S")
#
# test_df = test_instance.add_ffr_or_dx_service_identifier(df=merged_test_df, dx_periods=dx_periods)
#
# test_df = test_instance.add_additional_datetime_features(df=test_df, datetime_col="UTC_Settlement_DateTime")
#
# test_df = test_instance.add_additional_lagged_features(df=test_df, cols=["disp_ffr_percent_sum",
#                                                                          "disp_dcl_percent_sum",
#                                                                          "disp_dch_percent_sum",
#                                                                          "delta_freq_sum",
#                                                                          "disp_ffr_percent_mean",
#                                                                          "disp_ffr_percent_max",
#                                                                          "disp_ffr_percent_min",
#                                                                          "cleared_volume_dcl",
#                                                                          "cleared_volume_dch",
#                                                                          ])
#
# test_df = test_instance.remove_extreme_battery_outputs(df=test_df, no_of_stds=0, save_df_as_csv=False)
#
# # Assuming 'test_df' is your DataFrame and 'cols' is the list of column names
# # cols_to_exclude = ["battery_output"]
# # # Using list comprehension to select columns not in 'cols_to_exclude'
# # test_df = test_df[[col for col in cols if col not in cols_to_exclude]]
#
# predicted_df = predict_model(stacked_model, data=test_df)
#
# predicted_df = predicted_df[["prediction_label"]]
# predicted_df.rename(columns={"prediction_label": "battery_output",
#                              "index": "id"}, inplace=True)
# predicted_df.index = range(39457, 39457 + len(predicted_df))
#
# predicted_df.to_csv(os.path.join(os.path.dirname(__file__), "results",
#                                  "predicted_results_12_stacked.csv").replace("\\", "/"))
