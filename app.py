import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved models, scaler, and metadata
scaler = joblib.load('scaler.joblib')
models = {
    "SVR": joblib.load('svr_model.joblib'),
    "KNN": joblib.load('knn_model.joblib'),
    "Polynomial Regression": joblib.load('poly_model.joblib'),
    "Lasso": joblib.load('lasso_model.joblib'),
    "Ridge": joblib.load('ridge_model.joblib'),
    "ElasticNet": joblib.load('elastic_model.joblib')
}
poly_features = joblib.load('poly_features.joblib')
metadata = joblib.load('metadata.joblib')

feature_columns = metadata['feature_columns']
categorical_columns = metadata['categorical_columns']
category_levels = metadata['category_levels']
original_columns = metadata['original_columns']

# Load dataset to get feature ranges and modes
df = pd.read_csv("CarPrice_Assignment.csv")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('price')

# Top 10 numeric features based on importance
important_numeric_features = [
    'horsepower', 'curbweight', 'enginesize', 'citympg', 'highwaympg',
    'stroke', 'peakrpm', 'boreratio', 'wheelbase', 'compressionratio'
]

# Top categorical features
important_categorical_features = [
    'fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation',
    'enginetype', 'cylindernumber', 'fuelsystem'
]

# Calculate ranges for numeric features and median for filling
numeric_ranges = {col: (float(df[col].min()), float(df[col].max())) for col in important_numeric_features}
numeric_medians = df[numeric_cols].median()

# Mode for categorical columns
categorical_modes = {col: df[col].mode()[0] for col in categorical_columns}

st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title("🚗 Car Price Prediction App")
st.markdown("Choose a regression model and adjust the car features below to predict its price.")

# Sidebar: Model selection
st.sidebar.header("Select Regression Model")
model_name = st.sidebar.selectbox("Choose Model", list(models.keys()))
model = models[model_name]

# Sidebar: Input features
st.sidebar.header("Set Car Features")

# Numeric inputs with sliders for better UX
st.sidebar.subheader("Numeric Features")
user_input_numeric = {}
for col in important_numeric_features:
    min_val, max_val = numeric_ranges[col]
    median_val = numeric_medians[col] if col in numeric_medians else (min_val + max_val) / 2
    user_input_numeric[col] = st.sidebar.slider(
        label=col.replace('_', ' ').title(),
        min_value=min_val,
        max_value=max_val,
        value=float(median_val),
        step=0.01,
        help=f"Set the value for {col.replace('_', ' ')}"
    )

# Categorical inputs
st.sidebar.subheader("Categorical Features")
user_input_categorical = {}
for col in important_categorical_features:
    options = category_levels[col]
    default_cat = categorical_modes[col] if col in categorical_modes else options[0]
    user_input_categorical[col] = st.sidebar.selectbox(
        label=col.replace('_', ' ').title(),
        options=options,
        index=options.index(default_cat),
        help=f"Select the {col.replace('_', ' ')}"
    )

# Merge inputs
user_input = {**user_input_numeric, **user_input_categorical}

# Prepare input DataFrame
input_df = pd.DataFrame(columns=original_columns)

# Fill numeric columns
for col in numeric_cols:
    input_df.at[0, col] = user_input[col] if col in user_input else numeric_medians[col]

# Fill categorical columns
for col in categorical_columns:
    input_df.at[0, col] = user_input[col] if col in user_input else categorical_modes[col]

# Show user inputs for transparency
with st.expander("🔍 View Input Features", expanded=True):
    st.write(input_df.T)

# Predict button
if st.button("Predict Price"):
    # One-hot encode without drop_first to get all dummies
    input_encoded = pd.get_dummies(input_df, drop_first=False)

    # Mimic training one-hot encoding (drop_first=True) by dropping the first dummy for each categorical
    for cat_col in categorical_columns:
        # Dummy columns for this cat feature during training (alphabetical)
        training_dummies = sorted([col for col in feature_columns if col.startswith(cat_col + '_')])
        input_dummies = sorted([col for col in input_encoded.columns if col.startswith(cat_col + '_')])
        if training_dummies and input_dummies:
            # Drop the same dummy as training (first alphabetically)
            dummy_to_drop = training_dummies[0]
            if dummy_to_drop in input_encoded.columns:
                input_encoded.drop(columns=[dummy_to_drop], inplace=True)

    # Add missing columns as zeros
    missing_cols = set(feature_columns) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0

    # Ensure column order matches training features
    input_encoded = input_encoded[feature_columns]

    # Scale features
    input_scaled = scaler.transform(input_encoded)

    # Polynomial features transform if needed
    if model_name == "Polynomial Regression":
        input_scaled = poly_features.transform(input_scaled)

    # Predict
    predicted_price = model.predict(input_scaled)[0]

    # Display result prominently
    st.markdown("---")
    st.markdown("### Predicted Car Price:")
    st.markdown(f"<h2 style='color:#E63946;'>${predicted_price:,.2f}</h2>", unsafe_allow_html=True)

    # Show model details & coefficients for linear models
    st.markdown("### Model Details")
    st.write(f"Model: **{model_name}**")

    if model_name in ["Lasso", "Ridge", "ElasticNet"]:
        coef = model.coef_
        coef_df = pd.DataFrame({"Feature": feature_columns, "Coefficient": coef})
        coef_df = coef_df[coef_df['Coefficient'].abs() > 1e-4].sort_values(by='Coefficient', ascending=False).reset_index(drop=True)
        with st.expander("📈 Model Coefficients", expanded=False):
            st.dataframe(coef_df)

else:
    st.info("Adjust car features and click **Predict Price** to get the predicted car price.")
