from data.combine_data import PrepareModel
import pycaret
from pycaret.regression import *
from pycaret.regression import RegressionExperiment
import pandas as pd
import matplotlib.pyplot as plt

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)

train_instance = PrepareModel(test_or_train='train')

merged_train_df = train_instance.merge_dataframes(save_merged_df_as_csv=False)

train_df = train_instance.add_additional_datetime_features(df=merged_train_df, datetime_col="UTC_Settlement_DateTime")

train_df = train_instance.add_additional_weather_features(df=train_df, weather_cols=["temperature_2mLeeds_weather"])

s = setup(train_df, target='battery_output', session_id=123)

# import RegressionExperiment and init the class
exp = RegressionExperiment()

# init setup on exp
exp.setup(train_df, target='battery_output', session_id=123, feature_selection=True, remove_outliers=True,
          normalize=True, feature_selection_method="classic", remove_multicollinearity=True)

# compare baseline models
best = compare_models(sort="MAE")

# plot_model(best, plot='feature', save=True)
# plt.show()

# interpret_model(best)
