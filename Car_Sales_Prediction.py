
# Had developed with limited understanding of how learning works
'''import pandas as pd
import numpy  as np

# load data
df = pd.read_excel(r"E:\Python Projects\V2_CAR SALES 2 YEARS.xlsx")

# ensure month-start sate column is date-time
df["Attribute"] = pd.to_datetime(df["Attribute"])

# derive year and month for future requirement
df['Year'] = df["Attribute"].dt.year
df['Month'] = df["Attribute"].dt.month

# sort as per the time series for next operation
df = df.sort_values(["Company", "Model" ,"Attribute"]).reset_index(drop=True)

#The lag feature

lags = [1,2,3]

for lag in lags:
    df[f"value_lag_{lag}"] = (
        df.groupby(["Company", "Model"]) ["Value"]
        .shift(lag)
    )

df["rolling_avg_3"]= (
    df.groupby(["Company", "Model"]) ["Value"]
    .shift(1)
    .rolling(3)
    .mean()
)

df["VALUE_LAG_12"] = df.groupby(
    ["Company", "Model"]
    )["Value"].shift(12)

df["has_lag_12"] = (df["VALUE_LAG_12"]> 0 ).astype(int)

df["Value_lag_12_filled"] = df["VALUE_LAG_12"].fillna(
    df.groupby(["Company", "Month"])["Value"].transform("mean")
)

df["month_num"] = df["Attribute"].dt.month
df["month_sin"] = np.sin(2*np.pi*df["month_num"]/12)
df["month_cos"] = np.cos(2*np.pi*df["month_num"]/12)

df = df.dropna().reset_index(drop=True)

df["series_id"] = (
    df.groupby(["Company", "Model"])
      .ngroup()
)

feature_cols = [
    "value_lag_1",
    "value_lag_2",
    "value_lag_3",
    "rolling_avg_3",
    "month_sin",
    "month_cos",
    "series_id"
]

x = df[feature_cols]
y = df["Value"]

split_idx = int(len(df)*0.8)

x_train = x.iloc[:split_idx]
x_test = x.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

y_pre_naive = x_test["value_lag_1"]

from sklearn.metrics import mean_absolute_error

mae_naive = mean_absolute_error(y_test, y_pre_naive)
print("mae_naive:" , mae_naive)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred_lr = lr.predict(x_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
print("Linear Regression MAE: " , mae_lr) 

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators = 300,
    max_depth = 10,
    random_state = 42  
)
rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

avg_sales = y_test.mean()
print(avg_sales)

within_20pct = ((abs(y_test - y_pre_naive) / y_test) <= 0.20).mean()
print(within_20pct)

print("Random Forest MAE:", mae_rf)
print(f"Naive MAE: {mae_naive:.2f}")
print(f"Linear Regression MAE: {mae_lr:.2f}")




# January Tester
jan_mask = df.iloc[split_idx:]["Month"] == 1

mae_naive_jan = mean_absolute_error(
    y_test[jan_mask],
    y_pre_naive[jan_mask]
)

mae_lr_jan = mean_absolute_error(
    y_test[jan_mask],
    y_pred_lr[jan_mask]
)

print(mae_naive_jan,mae_lr_jan)

within_20pct = ((abs(y_test - y_pred_lr) / y_test) <= 0.20).mean()
print(within_20pct) '''







'''
# 2nd version of the above code after developing a better undertanding of learning
import pandas as pd
import numpy  as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from pathlib import Path

# load data
DATA_FILE = Path(r"E:\Python Projects\V2_CAR SALES 2 YEARS.xlsx")
df = pd.read_excel(DATA_FILE)

# ensure month-start sate column is date-time
df["Attribute"] = pd.to_datetime(df["Attribute"])

# derive year and month for future requirement
df['Year'] = df["Attribute"].dt.year
df['Month'] = df["Attribute"].dt.month

# sort as per the time series for next operation
df = df.sort_values(["Company", "Model" ,"Attribute"]).reset_index(drop=True)

#The lag feature

lags = [1,2,3]

for lag in lags:
    df[f"value_lag_{lag}"] = (
        df.groupby(["Company", "Model"]) ["Value"]
        .shift(lag)
    )

df["rolling_avg_3"]= (
    df.groupby(["Company", "Model"]) ["Value"]
    .shift(1)
    .rolling(3)
    .mean()
)

df["VALUE_LAG_12"] = df.groupby(
    ["Company", "Model"]
    )["Value"].shift(12)

df["has_lag_12"] = (df["VALUE_LAG_12"]> 0 ).astype(int)

df["Value_lag_12_filled"] = df["VALUE_LAG_12"].fillna(
    df.groupby(["Company", "Month"])["Value"].transform("mean")
)

df["month_num"] = df["Attribute"].dt.month
df["month_sin"] = np.sin(2*np.pi*df["month_num"]/12)
df["month_cos"] = np.cos(2*np.pi*df["month_num"]/12)

df = df.dropna().reset_index(drop=True)

df["series_id"] = (
    df.groupby(["Company", "Model"])
      .ngroup()
)


# Lifecycle learning logic

def get_lifecycle_state(grp, idx):
    lag_1 = grp.iloc[idx - 1]["Value"] if  idx - 1 >= 0 else 0
    lag_2 = grp.iloc[idx - 2]["Value"] if  idx - 2 >= 0 else 0

    future_vals = grp.iloc[idx:]["Value"].values

    if lag_1 > 0 and lag_2 > 0:
        return "Running"
    
    if lag_1 == 0 and lag_2 == 0 and np.any(future_vals > 0):
        return "Pre_Launch_Or_Gap"
    
    if lag_1 == 0 and lag_2 == 0 and not np.any(future_vals > 0):
        return "Discontinued"

    return "Unknown"

df["lifecycle_state"] = None

for (company, model), grp in df.groupby(["Company", "Model"]):
    grp = grp.sort_values("Attribute").reset_index()
    for i in range(len(grp)):
        state = get_lifecycle_state(grp, i)
        df.loc[grp.loc[i , "index"], "lifecycle"] = state
df["eligible_for_prediction"] = (
    df["lifecycle_state"].isin(["Running", "Unknown"])
)

feature_cols = [
    "value_lag_1",
    "value_lag_2",
    "value_lag_3",
    "rolling_avg_3",
    "month_sin",
    "month_cos",
    "series_id"
]

x = df[feature_cols]
y = df["Value"]

split_idx = int(len(df)*0.8)

x_train = x.iloc[:split_idx]
x_test = x.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

y_pre_naive = x_test["value_lag_1"]

from sklearn.metrics import mean_absolute_error

mae_naive = mean_absolute_error(y_test, y_pre_naive)
print("mae_naive:" , mae_naive)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred_lr = lr.predict(x_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
print("Linear Regression MAE: " , mae_lr) 

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators = 300,
    max_depth = 10,
    random_state = 42  
)
rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

avg_sales = y_test.mean()
print(avg_sales)

within_20pct = ((abs(y_test - y_pre_naive) / y_test) <= 0.20).mean()
print(within_20pct)

within_20pct = ((abs(y_test - y_pred_lr) / y_test) <= 0.20).mean()
print(within_20pct)

within_20pct = ((abs(y_test - y_pred_rf) / y_test) <= 0.20).mean()
print(within_20pct)

print("Random Forest MAE:", mae_rf)
print(f"Naive MAE: {mae_naive:.2f}")
print(f"Linear Regression MAE: {mae_lr:.2f}")




# January Tester
jan_mask = df.iloc[split_idx:]["Month"] == 1

mae_naive_jan = mean_absolute_error(
    y_test[jan_mask],
    y_pre_naive[jan_mask]
)

mae_lr_jan = mean_absolute_error(
    y_test[jan_mask],
    y_pred_lr[jan_mask]
)

print(mae_naive_jan,mae_lr_jan)

within_20pct = ((abs(y_test - y_pred_lr) / y_test) <= 0.20).mean()
print(within_20pct) 

'''

# 3rd and the version that actually responds to the API calls

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
from datetime import datetime
#from dateutil.relativedelta import relativedelta
#from openpyxl import load_workbook


#====================================================
# config
#====================================================

#DATA_FILE = Path(r"E:\Python Projects\Car Sales Site\CAR SALES PQ.xlsx")
#OUTPUT_FILE = Path(R"E:\Python Projects\Car Sales Site\CAR SALES PREDICTOR.xlsx")
#SHEET_NAME = "prediction_history"

DATA_FILE = Path("CAR SALES PQ.xlsx")

Naive_Weight = .6
LR_Weight = .4

#====================================================
# Data Preparation
#====================================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["Attribute"] = pd.to_datetime(df["Attribute"])
    df["Year"] = df["Attribute"].dt.year
    df["Month"] = df["Attribute"].dt.month

    df = df.sort_values(["Company", "Model", "Attribute"]).reset_index(drop = True)

    # lag features

    for lag in [1,2,3]:
        df[f"value_lag_{lag}"] = (
            df.groupby(["Company", "Model"])["Value"].shift(lag)
        )

    df["rolling_avg_3"] = (
        df.groupby(["Company", "Model"])["Value"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    # Debug code i want run later to check the number of months being considered to calculate the average of the new month

#    df["debug_window"] = (
#    df.groupby(["Company", "Model"])["Month"]
#      .shift(1)
#      .rolling(3)
#      .apply(lambda x: list(x), raw=False)
#)

#print(
#    df[
#        ["Company", "Model", "Month", "Values", "rolling_avg_3"]
#    ]
#    .sort_values(["Company", "Model", "Month"])
#    .to_string(index=False)
#)

    # Seasonality
    df["month_num"] = df["Attribute"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"]/12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"]/12)

    # series ID
    df["series_id"] = (
        df.groupby(["Company", "Model"])
        .ngroup()
    )

    return df

#====================================================
# Lifecycle Logic
#====================================================

def get_lifecycle_state(grp, idx):
    lag_1 = grp.iloc[idx - 1]["Value"] if idx - 1 >= 0 else 0
    lag_2 = grp.iloc[idx - 2]["Value"] if idx - 2 >= 0 else 0 

    future_vals = grp.iloc[idx:]["Value"].values

    if lag_1 > 0 and lag_2 > 0:
        return "Running"
    
    if lag_1 == 0 and lag_2 == 0 and np.any(future_vals > 0):
        return "Pre_Launch_Or_Gap"
    
    if lag_1 ==0 and lag_2 == 0 and not np.any(future_vals > 0):
        return "Discontinued"

    return "Unknown"

def apply_lifecycle(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lifecycle_state"] = None

    for (company, model), grp in df.groupby(["Company", "Model"]):
        grp = grp.sort_values("Attribute").reset_index()
        for i in range(len(grp)):
            state = get_lifecycle_state(grp, i)
            df.loc[grp.loc[i, "index"], "lifecycle_state"] = state
        
    df["eligible_for_prediction"] = df["lifecycle_state"].isin(
        ["Running", "Unknown"]
    )
    
    return df

#====================================================
# Core Prediction Function (This is the key output)
#====================================================

def generate_next_month_prediction(cutoff_date: str):

    """
    Generates predictions for cutoff_date + 1 month
    using Naive + Linear Regression hybrid
    """
    df = pd.read_excel(DATA_FILE)
    
    #print(df.columns.tolist())


    # Load & prepare data
    df = pd.read_excel(DATA_FILE)
    df = prepare_features(df)

    cutoff_date = pd.to_datetime(cutoff_date)
    prediction_month_date = cutoff_date

    # Filter data up to cutoff
    df = df[df["Attribute"] <= cutoff_date].copy()

    df = apply_lifecycle(df)

    feature_cols = [
        "value_lag_1",
        "value_lag_2",
        "value_lag_3",
        "rolling_avg_3",
        "month_sin",
        "month_cos",
        "series_id"
    ]

    # Train LR on all available Past Data
    train_df = df.dropna().copy()

    if train_df.empty:
        return[]
    
    x_train = train_df[feature_cols]
    y_train = train_df["Value"]

    lr = LinearRegression()
    lr.fit(x_train, y_train)

    predictions = []

    #generate one prediction per company model pair
    for (company, model), grp in df.groupby(["Company", "Model"]):
        grp = grp.sort_values("Attribute")
        last_row = grp.iloc[-1]

        if not last_row["eligible_for_prediction"]:
            continue

    # Build next month row
        next_month_features = {
            "value_lag_1": last_row["Value"],
            "value_lag_2": grp.iloc[-2]["Value"] if len(grp) > 1 else last_row["Value"],
            "value_lag_3": grp.iloc[-3]["Value"] if len(grp) > 2 else last_row["Value"],
            "rolling_avg_3" : grp.tail(3)["Value"].mean(),
            "month_sin" : np.sin(2 * np.pi * prediction_month_date.month/12),
            "month_cos" : np.cos(2 * np.pi * prediction_month_date.month/12),
            "series_id": last_row["series_id"]
        }

        x_next = pd.DataFrame([next_month_features])

        naive_pred = next_month_features["value_lag_1"]
        lr_pred = lr.predict(x_next)[0]

        final_pred = (
            Naive_Weight*naive_pred + LR_Weight * lr_pred
        )

        predictions.append({
            "company": company,
            "model" : model,
            "predicted_for_month" : int(prediction_month_date.strftime("%Y%m")),
            "predictions_made_on" : cutoff_date.strftime("%Y-%m-%d"),
            "naive_prediction" : round(naive_pred, 2),
            "lr_prediction" : round(lr_pred, 2),
            "final_prediction" : round(final_pred, 2),
        })

    return predictions

#====================================================
# Excel Append (Storage Layer)
#====================================================

'''
def append_predictions_to_excel(predictions:list):

    df_new = pd.DataFrame(predictions)

    df_new = df_new[
        [
            "predictions_made_on",
            "predicted_for_month",
            "company",
            "model",
            "naive_prediction",
            "lr_prediction",
            "final_prediction"
        ]
    ]

    if not OUTPUT_FILE.exists():
        df_new.to_excel(
            OUTPUT_FILE,
            sheet_name= SHEET_NAME,
            index=False
        )
        print("New prediction history file created.")
        return
    
    df_existing = pd.read_excel(OUTPUT_FILE, sheet_name=SHEET_NAME)

    df_combined = pd.concat([df_existing, df_new], ignore_index = True)

    df_combined = df_combined.drop_duplicates(
        subset=[
            "predictions_made_on",
            "predicted_for_month",
            "company",
            "model",
        ],
        keep="first"
    )

    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl",
        mode="a",
        if_sheet_exists= "overlay"
    ) as writer:

        #book =load_workbook(OUTPUT_FILE)
        #writer = book

        #start_row = (
        #    book[SHEET_NAME].max_row
        #    if SHEET_NAME in book.sheetnames
        #    else 0
        #)

        df_combined.to_excel(
            writer,
            sheet_name=SHEET_NAME,
            index=False
        )
    print("Predictions appended (duplicates removed if any).")
'''
#====================================================
# One time backfill engine
#====================================================
'''
def backfill_historical_predictions():

    df = pd.read_excel(DATA_FILE)
    #df_long = reshape_wide_to_long(df_raw)

    df["Attribute"] = pd.to_datetime(df["Attribute"])

    all_months = sorted(df["Attribute"].unique())

    print(f"Total months available: {len(all_months)}")

    for cutoff_date in all_months[3:-1]:

        cutoff_str = pd.to_datetime(cutoff_date).strftime("%Y-%m-%d")

        print(f"Generating prediction using cutoff: {cutoff_str}")

        preds = generate_next_month_prediction(cutoff_str)

        append_predictions_to_excel(preds)

    print("Backfill completed.")

#if __name__ == "__main__":

    #backfill_historical_predictions()
#    pass
'''
# ====================================================
# LOCAL TEST / MANUAL RUN
# ====================================================
if __name__ == "__main__":

    # Set the month you want to predict
    cutoff_date = "2026-01-01"  # <-- change whenever needed

    preds = generate_next_month_prediction(cutoff_date)

    #append_predictions_to_excel(preds)

    print(f"{len(preds)} predictions generated and saved.")
