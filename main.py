from data.combine_data import PrepareModel
import pycaret
from pycaret.regression import *
from pycaret.regression import RegressionExperiment
import pandas as pd
import matplotlib.pyplot as plt

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)

train_instance = PrepareModel(test_or_train='train')

merged_train_df = train_instance.merge_dataframes(save_merged_df_as_csv=False)

train_df = train_instance.add_additional_datetime_features(df=merged_train_df, datetime_col="UTC_Settlement_DateTime")

train_df = train_instance.add_additional_weather_features(df=train_df, weather_cols=["temperature_2mLeeds_weather"])

s = setup(train_df, target='battery_output', session_id=123)

# import RegressionExperiment and init the class
exp = RegressionExperiment()

# init setup on exp
exp.setup(train_df, target='battery_output', session_id=123,
          # log_experiment=True,
          feature_selection=True,
          remove_outliers=True,
          normalize=True,
          feature_selection_method="classic",
          remove_multicollinearity=True)

# compare baseline models
# best = compare_models(sort="MAE", include=['huber', 'gbr', 'br', 'en', 'lasso'])

# interpret_model(best)
# print(best)

# train model
huber_model = create_model('huber', fold=5, alpha=0.9, epsilon=1.2, return_train_score=True)

# plot_model(huber_model, plot='learning', save=True)

# tune model
tuned_huber = tune_model(huber_model, n_iter=15, optimize="MAE", choose_better=True)

print(tuned_huber)

