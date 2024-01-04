import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
import statsmodels.api as sm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('DataTesla.csv')

# Function to perform model fitting
def fit_models(data):
    
    print(data.columns)
    
    X = data.drop('Tesla Stock Price', axis=1).select_dtypes(np.number)
    y = data['Tesla Stock Price']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit models
    xgb_model = XGBRegressor().fit(X_train, y_train)
    lasso_model = Lasso().fit(X_train, y_train)

    # For PC Regression
    pca = PCA(n_components=min(X_train.shape))
    X_train_pca = pca.fit_transform(X_train)
    pc_regression_model = sm.OLS(y_train, sm.add_constant(X_train_pca)).fit()

    return xgb_model, lasso_model, pc_regression_model, X_test, y_test

# Function to retrieve log-likelihood results
def get_log_likelihood_results(models, X_test, y_test):
    results = {}
    predictions = {}

    for name, model in models.items():
        if name in ['OLS', 'PC Regression']:
            results[name] = model.llf
        else:
            # Calculate likelihood 
            model_predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, model_predictions)
            likelihood = -0.5 * len(y_test) * np.log(2 * np.pi * mse) - 0.5 * len(y_test)
            results[name] = likelihood
            predictions[name] = model_predictions

    return results, predictions

# Sidebar with filters
st.sidebar.title("Filtres")

# Sidebar for financial data
st.sidebar.header("Données financières")
selected_financial_metrics = st.sidebar.multiselect('Sélectionnez les métriques financières', ['Tesla Stock Price', 'NASDAQ Composite', 'S&P 500', 'Oil Price'])

# Sidebar for death data
st.sidebar.header("Données de décès")
selected_death_metrics = st.sidebar.multiselect('Sélectionnez les métriques de décès', ['Deaths', 'Deaths Lag 1', 'Deaths Lag 7', 'Deaths Lag 30'])

# Sidebar for Twitter data
st.sidebar.header("Données Twitter")
selected_twitter_metrics = st.sidebar.multiselect('Sélectionnez les métriques Twitter', ['Tweets of Elon Musk', 'Tweet with mention Tesla', 'Tweet Likes', 'Retweet'])

# Filter data
filtered_data = df[selected_financial_metrics + selected_death_metrics + selected_twitter_metrics]

# Page for Data Visualization
def data_visualization_page():
    st.title("Visualisation des données temporelles")

    # Financial Metrics
    for metric in selected_financial_metrics:
        if metric in selected_financial_metrics:
            fig_financial = go.Figure()
            fig_financial.add_trace(go.Scatter(x=df['Date'], y=df[metric], mode='lines', name=metric))
            fig_financial.update_layout(title=f"Évolution de {metric} au fil du temps", xaxis_title="Date", yaxis_title="Valeur")
            st.plotly_chart(fig_financial)

    # Deaths and Tweets Normalization and Plotting
    st.title("Évolution des décès et de l'activité Twitter normalisée")
    normalized_data = pd.DataFrame()

    # Plot all selected Death Metrics
    if selected_death_metrics:
        fig_deaths = go.Figure()
        for metric in selected_death_metrics:
            normalized_metric = metric + ' (normalized)'
            max_value = df[metric].max()
            if max_value != 0:
                normalized_data[normalized_metric] = df[metric] / max_value
                fig_deaths.add_trace(go.Scatter(x=df['Date'], y=normalized_data[normalized_metric], mode='lines', name=normalized_metric))
        fig_deaths.update_layout(title="Évolution des décès normalisés au fil du temps", xaxis_title="Date", yaxis_title="Valeur normalisée")
        st.plotly_chart(fig_deaths)

    # Normalize and plot all selected Twitter Metrics
    if selected_twitter_metrics:
        fig_twitter = go.Figure()
        for metric in selected_twitter_metrics:
            normalized_metric = metric + ' (normalized)'
            max_value = df[metric].max()
            if max_value != 0:
                normalized_data[normalized_metric] = df[metric] / max_value
                fig_twitter.add_trace(go.Scatter(x=df['Date'], y=normalized_data[normalized_metric], mode='lines', name=normalized_metric))
        fig_twitter.update_layout(title="Évolution de l'activité Twitter normalisée au fil du temps", xaxis_title="Date", yaxis_title="Valeur normalisée")
        st.plotly_chart(fig_twitter)

# Page for Estimation Results
def estimation_results_page(models, X_test, y_test):
    st.title("Estimation Results")

    # Retrieve log-likelihood results
    results, predictions = get_log_likelihood_results(models, X_test, y_test)

    # Display the results
    st.write("Log-Likelihood Results:")
    st.write(results)

    # Display the best model
    best_model = pd.Series(results).idxmax()
    best_llf = pd.Series(results).max()
    st.write(f'Best model is: {best_model} with Log-likelihood of: {best_llf}')

    # Plot predictions for the best model
    st.write("Plotting Predictions for the Best Model:")
    fig, ax = plt.subplots()
    tmp_df = pd.DataFrame({'Predictions': predictions[best_model], 'Realized': y_test}).reset_index(drop=True)
#     tmp_df.plot(ax=ax)
    st.line_chart(tmp_df)

# Main Streamlit App
def main():
    st.sidebar.title("Dashboard Navigation")
    page = st.sidebar.selectbox("Select a page", ["Data Visualization", "Estimation Results"])

    if page == "Data Visualization":
        data_visualization_page()
    elif page == "Estimation Results":
        filtered_data = pd.read_csv('DataTesla.csv')
        xgb_model, lasso_model, pc_regression_model, X_test, y_test = fit_models(filtered_data)
        models = {'XGBoost': xgb_model, 'Lasso': lasso_model, 'PC Regression': pc_regression_model}
        estimation_results_page(models, X_test, y_test)

if __name__ == "__main__":
    main()

