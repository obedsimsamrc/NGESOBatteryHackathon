from data.combine_data import PrepareModel
import pycaret
from pycaret.regression import *
from pycaret.regression import RegressionExperiment
from gplearn.genetic import SymbolicRegressor
import pandas as pd
import os

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)

train_instance = PrepareModel(test_or_train='train')

merged_train_df = train_instance.merge_dataframes(include_dyn_market=True,
                                                  include_gen=False)

train_df = train_instance.add_additional_datetime_features(df=merged_train_df, datetime_col="UTC_Settlement_DateTime")

dx_periods = [("05/08/2022", "31/08/2022"), ("01/11/2022", "30/11/2022"), ("02/02/2023", "28/02/2023"),
              ("01/06/2023", "01/07/2023")]
train_df = train_instance.add_ffr_or_dx_service_identifier(df=train_df, dx_periods=dx_periods)

train_df = train_instance.add_additional_lagged_features(df=train_df, cols=["temperature_2mNewcastle upon Tyne_weather",
                                                                            "windspeed_10mNewcastle upon Tyne_weather",
                                                                            ])

train_df = train_instance.remove_extreme_battery_outputs(df=train_df, no_of_stds=2, save_df_as_csv=True)

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
          pca=True,
          # low_variance_threshold=0.1,
          categorical_features=['year', 'season', 'is_winter', 'month', 'week_of_year', 'day', 'dayofweek', 'hour',
                                'minute', 'is_wknd', 'is_working_hr', 'is_lunch_hr', 'EFA Block Count', 'EFA HH Count',
                                'in_dx'],
          feature_selection_method="classic",
          remove_multicollinearity=True,
          imputation_type="iterative",
          numeric_iterative_imputer="lightgbm",
          train_size=0.7,
          fold_shuffle=True,
          profile=True
          )

# compare baseline models
# best = compare_models(sort="MAE",
#                       include=['lightgbm', 'et', 'rf']
#                       )

# train model
# lightgbm_model = create_model('lightgbm', bagging_fraction=0.7, bagging_freq=7, feature_fraction=0.5,
#                               min_child_samples=41, min_split_gain=0.3, n_estimators=130,
#                               n_jobs=-1, num_leaves=50, random_state=123, reg_alpha=3,
#                               reg_lambda=0.0001, return_train_score=True)

# tune model
# tuned_huber = tune_model(huber_model, n_iter=10, optimize="MAE", choose_better=True)
# tuned_lightgbm = tune_model(lightgbm_model, n_iter=25, optimize="MAE", choose_better=True)

# print(tuned_lightgbm)

# plot_model(best, plot='feature', save=True)
# plot_model(best, plot='error', save=True)

test_instance = PrepareModel(test_or_train='test')

merged_test_df = test_instance.merge_dataframes(include_dyn_market=True,
                                                include_gen=False)

merged_test_df["UTC_Settlement_DateTime"] = pd.to_datetime(merged_test_df["UTC_Settlement_DateTime"],
                                                           format="%Y-%m-%d %H:%M:%S")

test_df = test_instance.add_ffr_or_dx_service_identifier(df=merged_test_df, dx_periods=dx_periods)

test_df = test_instance.add_additional_datetime_features(df=test_df, datetime_col="UTC_Settlement_DateTime")

test_df = test_instance.add_additional_lagged_features(df=test_df, cols=["temperature_2mNewcastle upon Tyne_weather",
                                                                         "windspeed_10mNewcastle upon Tyne_weather",
                                                                         ])

test_df = test_instance.remove_extreme_battery_outputs(df=test_df, no_of_stds=0, save_df_as_csv=True)

#
# predicted_df = predict_model(lightgbm_model, data=test_df)
#
# predicted_df = predicted_df[["prediction_label"]]
# predicted_df.rename(columns={"prediction_label": "battery_output",
#                              "index": "id"}, inplace=True)
# predicted_df.index = range(39457, 39457 + len(predicted_df))

# predicted_df.to_csv(os.path.join(os.path.dirname(__file__), "results", "predicted_results_2.csv").replace("\\", "/"))
