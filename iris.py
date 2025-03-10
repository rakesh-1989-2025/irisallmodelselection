import pandas as pd
import numpy as np

import joblib

import streamlit as st

def load_models():
    logistic_reg_binary_scalar_data = joblib.load('logistics_binary_scalar.joblib')
    log_reg_binary_model = logistic_reg_binary_scalar_data['model']
    scalar_binary = logistic_reg_binary_scalar_data['scaler']
    
    logistic_reg_ovr_multiclass_scalar_data = joblib.load('logistics_ovr_multi_scalar.joblib')
    log_reg_ovr_multiclass_model = logistic_reg_ovr_multiclass_scalar_data['model']
    scalar_multiclass = logistic_reg_ovr_multiclass_scalar_data['scaler']
    
    logistic_reg_multinomial_multiclass_scalar_data = joblib.load('logistics_multinomial_multi_scalar.joblib')
    log_reg_multinomial_multiclass_model = logistic_reg_multinomial_multiclass_scalar_data['model']
    
    svc_binary_scalar_data = joblib.load('svm(c)_binary_scalar.joblib')
    svc_binary_model = svc_binary_scalar_data['model']
    svc_binary_scalar = svc_binary_scalar_data['scaler']

    svc_multiclass_scalar_data = joblib.load('svm(c)_multi_scalar.joblib')
    svc_multiclass_model = svc_multiclass_scalar_data['model']
    svc_multiclass_scalar = svc_multiclass_scalar_data['scaler']

    return log_reg_binary_model, log_reg_ovr_multiclass_model, log_reg_multinomial_multiclass_model, svc_binary_model, svc_multiclass_model, scalar_binary, scalar_multiclass, svc_binary_scalar, svc_multiclass_scalar



def preprocessing_input_data(data, scalar):
    df = pd.DataFrame([data])

    df_scaled = scalar.transform(df)

    return df_scaled



def predict_data(data, model, scalar):
    df_scaled = preprocessing_input_data(data, scalar)

    return model.predict(df_scaled)



def main():
    st.title("Iris Flower Species Classification App")
    st.write("Enter the data to get a prediction for the Iris Flower Species")

    log_reg_binary_model, log_reg_ovr_multiclass_model, log_reg_multinomial_multiclass_model, svc_binary_model, svc_multiclass_model, scalar_binary, scalar_multiclass, svc_binary_scalar, svc_multiclass_scalar = load_models()

    sepal_length = st.slider("Sepal Length (cm)", 0.0, 10.0, 5.0)
    sepal_width = st.slider("Sepal Width (cm)", 0.0, 10.0, 5.0)
    petal_length = st.slider("Petal Length (cm)", 0.0, 10.0, 5.0)
    petal_width = st.slider("Petal Width (cm)", 0.0, 10.0, 5.0)

    model_options = [
        "Logistic Regression Binary Classification",
        "Logistic Regression OVR Multiclass Classification",
        "Logistic Regression Multinomial Multiclass Classification",
        "SVM(C) Binary Classification",
        "SVM(C) Multiclass Classification"
    ]

    selected_model = st.selectbox("Select a model", model_options)

    if st.button("Classify the Species"):
        user_data = {
            "sepal length (cm)": sepal_length,
            "sepal width (cm)": sepal_width,
            "petal length (cm)": petal_length,
            "petal width (cm)": petal_width
        }

        if selected_model == "Logistic Regression Binary Classification":
            prediction = predict_data(user_data, log_reg_binary_model, scalar_binary)
        elif selected_model == "Logistic Regression OVR Multiclass Classification":
            prediction = predict_data(user_data, log_reg_ovr_multiclass_model, scalar_multiclass)
        elif selected_model == "Logistic Regression Multinomial Multiclass Classification":
            prediction = predict_data(user_data, log_reg_multinomial_multiclass_model, scalar_multiclass)
        elif selected_model == "SVM(C) Binary Classification":
            prediction = predict_data(user_data, svc_binary_model, svc_binary_scalar)
        elif selected_model == "SVM(C) Multiclass Classification":
            prediction = predict_data(user_data, svc_multiclass_model, svc_multiclass_scalar)

        if prediction == 0:
            prediction = "Setosa"
        elif prediction == 1:
            prediction = "Versicolor"
        elif prediction == 2:
            prediction = "Virginica"

        st.success(f"Prediction using {selected_model}: {prediction}")



if __name__ == "__main__":
    main()