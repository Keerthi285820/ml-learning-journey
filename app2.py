import streamlit as st
import pandas as pd
import numpy as np
st.title("📊 Data Drift Detection System")
st.write("Upload training data and new data to detect drift")
train_file = st.file_uploader("Upload Training Dataset", type=["csv"])
test_file = st.file_uploader("Upload New Dataset", type=["csv"])
if train_file and test_file:
    df_train = pd.read_csv(train_file)
    df_new = pd.read_csv(test_file)
    st.subheader("Training Data Preview")
    st.write(df_train.head())
    st.subheader("New Data Preview")
    st.write(df_new.head())
    num_cols = df_train.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df_train.select_dtypes(include=['object']).columns
    drift_results = []

    for col in num_cols:
        old_mean = df_train[col].mean()
        new_mean = df_new[col].mean()
        diff = abs(new_mean - old_mean)

        drift = "Yes" if diff > 5 else "No"

        drift_results.append({
            "Feature": col,
            "Type": "Numerical",
            "Old Mean": round(old_mean, 2),
            "New Mean": round(new_mean, 2),
            "Difference": round(diff, 2),
            "Drift Detected": drift
        })

    for col in cat_cols:
        old_dist = df_train[col].value_counts(normalize=True)
        new_dist = df_new[col].value_counts(normalize=True)

        drift = "Yes" if not old_dist.equals(new_dist) else "No"

        drift_results.append({
            "Feature": col,
            "Type": "Categorical",
            "Old Mean": "-",
            "New Mean": "-",
            "Difference": "-",
            "Drift Detected": drift
        })

    drift_df = pd.DataFrame(drift_results)
    st.subheader("📈 Drift Report")
    st.dataframe(drift_df)
    total_drift = drift_df["Drift Detected"].value_counts()
    st.subheader("📊 Summary")
    st.write(total_drift)

    if "Yes" in total_drift:
        st.error("⚠️ Data Drift Detected! Model retraining recommended.")
    else:
        st.success("✅ No significant drift detected.")
