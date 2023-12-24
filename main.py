from data.combine_data import PrepareModel
import pycaret
from pycaret.regression import *
from pycaret.regression import RegressionExperiment
import pandas as pd
import os

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)

train_instance = PrepareModel(test_or_train='train')

merged_train_df = train_instance.merge_dataframes(save_merged_df_as_csv=False,
                                                  include_dyn_market=True,
                                                  include_gen=False)

train_df = train_instance.add_additional_datetime_features(df=merged_train_df, datetime_col="UTC_Settlement_DateTime")

train_df = train_instance.add_additional_lagged_features(df=train_df, cols=["temperature_2mLeeds_weather",
                                                                            "windspeed_10mLeeds_weather",
                                                                            # "DINO4Dinorwig",
                                                                            # "KLGLW1KilgalliochWindFarm"
                                                                            ])

# Remove all battery_output occurrences of 2 std
# Calculate the mean and standard deviation of the specified column
mean_value = train_df["battery_output"].mean()
std_dev = train_df["battery_output"].std()

# Define the threshold for culling
lower_bound = mean_value - 2 * std_dev
upper_bound = mean_value + 2 * std_dev

# Keep only the rows within the specified range
train_df = train_df[(train_df["battery_output"] >= lower_bound) &
                    (train_df["battery_output"] <= upper_bound)]

s = setup(train_df, target='battery_output', session_id=123)

# import RegressionExperiment and init the class
exp = RegressionExperiment()

# init setup on exp
exp.setup(train_df, target='battery_output',
          session_id=123,
          # log_experiment=True,
          feature_selection=True,
          remove_outliers=True,
          # outliers_threshold=0.1,
          normalize=True,
          transformation=True,
          pca=True,
          low_variance_threshold=0.1,
          categorical_features=['year', 'season', 'is_winter', 'month', 'week_of_year', 'day', 'dayofweek', 'hour',
                                'minute', 'is_wknd', 'is_working_hr', 'is_lunch_hr', 'EFA Block Count', 'EFA HH Count'],
          feature_selection_method="classic",
          remove_multicollinearity=True,
          imputation_type="iterative",
          numeric_iterative_imputer="lightgbm",
          train_size=0.7,
          )

# compare baseline models
# best = compare_models(sort="MAE",
#                       include=['lightgbm', 'gbr', 'rf']
#                       )

# train model
lightgbm_model = create_model('lightgbm', bagging_fraction=0.8, bagging_freq=3, feature_fraction=1.0,
                              learning_rate=0.005, min_child_samples=96, min_split_gain=0.5,
                              n_estimators=210, n_jobs=-1, num_leaves=80, random_state=123,
                              reg_alpha=0.001, reg_lambda=1e-06,
                              # return_train_score=True,
                              )

# tune model
# tuned_huber = tune_model(huber_model, n_iter=10, optimize="MAE", choose_better=True)
# tuned_lightgbm = tune_model(lightgbm_model, n_iter=15, optimize="MAE", choose_better=True)

# print(tuned_huber)

plot_model(lightgbm_model, plot='feature', save=True)
plot_model(lightgbm_model, plot='error', save=True)


test_instance = PrepareModel(test_or_train='test')

merged_test_df = test_instance.merge_dataframes(save_merged_df_as_csv=False,
                                                include_dyn_market=True,
                                                include_gen=False)
merged_test_df["UTC_Settlement_DateTime"] = pd.to_datetime(merged_test_df["UTC_Settlement_DateTime"],
                                                           format="%Y-%m-%d %H:%M:%S")
test_df = test_instance.add_additional_datetime_features(df=merged_test_df, datetime_col="UTC_Settlement_DateTime")

test_df = test_instance.add_additional_lagged_features(df=test_df, cols=["temperature_2mLeeds_weather",
                                                                         "windspeed_10mLeeds_weather"])

predicted_df = predict_model(lightgbm_model, data=test_df)

predicted_df = predicted_df[["UTC_Settlement_DateTime", "prediction_label"]]
# predicted_df.to_csv(os.path.join(os.path.dirname(__file__), "results", "predicted_results.csv").replace("\\", "/"))

