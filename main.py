from data.combine_data import PrepareModel
import pycaret
from pycaret.regression import *
from pycaret.regression import RegressionExperiment
import pandas as pd

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)

train_instance = PrepareModel(test_or_train='train')

merged_train_df = train_instance.merge_dataframes(save_merged_df_as_csv=False,
                                                  include_dyn_market=True,
                                                  include_gen=True)

train_df = train_instance.add_additional_datetime_features(df=merged_train_df, datetime_col="UTC_Settlement_DateTime")

train_df = train_instance.add_additional_lagged_features(df=train_df, cols=["temperature_2mLeeds_weather",
                                                                            "windspeed_10mLeeds_weather",
                                                                            # "DINO4Dinorwig",
                                                                            # "KLGLW1KilgalliochWindFarm"
                                                                            ])

# Create a SoC column
change_in_soc = train_df["battery_output"] / 35 / 2

s = setup(train_df, target='battery_output', session_id=123)

# import RegressionExperiment and init the class
exp = RegressionExperiment()

# init setup on exp
exp.setup(train_df, target='battery_output',
          session_id=123,
          # log_experiment=True,
          feature_selection=True,
          remove_outliers=True,
          outliers_threshold=0.05,
          # normalize=True,
          transformation=True,
          # pca=True,
          low_variance_threshold=0.1,
          categorical_features=['year', 'season', 'is_winter', 'month', 'week_of_year', 'day', 'dayofweek', 'hour',
                                'minute', 'is_wknd', 'is_working_hr', 'is_lunch_hr', 'EFA Block Count', 'EFA HH Count'],
          feature_selection_method="classic",
          remove_multicollinearity=True,
          imputation_type="iterative",
          numeric_iterative_imputer="lightgbm",
          )

# compare baseline models
best = compare_models(sort="MAE",
                      include=['huber', 'gbr']
                      )

# interpret_model(best)
# dashboard(best)
# print(best)
# check ML logs
# get_logs()
# reg1 = get_current_experiment()

# train model
huber_model = create_model('huber',
                           fold=5,
                           alpha=0.01,
                           epsilon=1.1,
                           # return_train_score=True
                           )

plot_model(huber_model, plot='feature', save=True)
plot_model(huber_model, plot='error', save=True)

# tune model
tuned_huber = tune_model(huber_model, n_iter=10, optimize="MAE", choose_better=True)

# print(tuned_huber)
