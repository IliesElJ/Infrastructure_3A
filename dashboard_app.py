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
import shap

st.set_option('deprecation.showPyplotGlobalUse', False)
name_1, name_2, name_3, name_4, name_5 = "Thomas", "Arthur", "Ilies", "Ismaïl", "Florent"
df = pd.read_csv('datasets/DataTesla.csv')


# Function to perform model fitting
def fit_models(data):
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


# Introductory text
st.markdown(f"""
# Bienvenue sur notre Dashboard!

Ce projet Python d'Infrastructures et Systèmes Logiciels a été réalisé avec passion et dévouement par notre équipe 
dynamique composé de **{name_1}**, **{name_2}**, **{name_3}**, **{name_4}** et **{name_5}**.

Nous sommes fiers de vous présenter notre travail sur l'analyse des données de Tesla. Ce dashboard interactif vous 
permet d'explorer diverses métriques financières, de décès, et des données Twitter liées à Tesla, ainsi que les 
résultats de nos modèles prédictifs. 

Plongez dans l'univers des données et découvrez les insights cachés derrière les chiffres!

---

**Instructions:**
Utilisez la barre latérale pour naviguer à travers les différentes sections du dashboard et interagir avec les visualisations.

Bonne exploration!
""", unsafe_allow_html=True)


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


st.sidebar.image("sticker.png", use_column_width=True)

# Sidebar with filters
st.sidebar.title("Filtres")

# Sidebar for financial data
st.sidebar.header("Données financières")
selected_financial_metrics = st.sidebar.multiselect('Sélectionnez les métriques financières',
                                                    ['Tesla Stock Price', 'NASDAQ Composite', 'S&P 500', 'Oil Price'])

# Sidebar for death data
st.sidebar.header("Données de décès")
selected_death_metrics = st.sidebar.multiselect('Sélectionnez les métriques de décès',
                                                ['Deaths', 'Deaths Lag 1', 'Deaths Lag 7', 'Deaths Lag 30'])

# Sidebar for Twitter data
st.sidebar.header("Données Twitter")
selected_twitter_metrics = st.sidebar.multiselect('Sélectionnez les métriques Twitter',
                                                  ['Tweets of Elon Musk', 'Tweet with mention Tesla', 'Tweet Likes',
                                                   'Retweet'])

# Filter data
filtered_data = df[selected_financial_metrics + selected_death_metrics + selected_twitter_metrics]


# Page for Data Visualization
def data_visualization_page():
    # Financial Metrics
    for metric in selected_financial_metrics:
        st.title("Visualisation des données temporelles")
        if metric in selected_financial_metrics:
            fig_financial = go.Figure()
            fig_financial.add_trace(go.Scatter(x=df['Date'], y=df[metric], mode='lines', name=metric))
            fig_financial.update_layout(title=f"Évolution de {metric} au fil du temps", xaxis_title="Date",
                                        yaxis_title="Valeur")
            st.plotly_chart(fig_financial)

            st.write(plot_markdown(metric))

    # Deaths and Tweets Normalization and Plotting
    normalized_data = pd.DataFrame()

    # Plot all selected Death Metrics
    if selected_death_metrics:
        st.title("Évolution des décès liés au accidents")
        fig_deaths = go.Figure()
        for metric in selected_death_metrics:
            normalized_metric = metric + ' (normalized)'
            max_value = df[metric].max()
            if max_value != 0:
                normalized_data[normalized_metric] = df[metric] / max_value
                fig_deaths.add_trace(go.Scatter(x=df['Date'], y=normalized_data[normalized_metric], mode='lines',
                                                name=normalized_metric))
            st.write(plot_markdown(metric))
        fig_deaths.update_layout(title="Évolution des décès normalisés au fil du temps", xaxis_title="Date",
                                 yaxis_title="Valeur normalisée")
        st.plotly_chart(fig_deaths)

    # Normalize and plot all selected Twitter Metrics
    if selected_twitter_metrics:
        print("Selected Twitter Metrics:", selected_twitter_metrics)  # Debug print
        st.title("Évolution de l'activité Twitter normalisée")
        fig_twitter = go.Figure()
        for metric in selected_twitter_metrics:
            max_value = df[metric].max()
            print(f"Max value for {metric}: {max_value}")  # Debug print
            if max_value != 0:
                normalized_metric = metric + ' (normalized)'
                normalized_data[normalized_metric] = df[metric] / max_value
                fig_twitter.add_trace(go.Scatter(x=df['Date'], y=normalized_data[normalized_metric], mode='lines',
                                                 name=normalized_metric))
            else:
                print(f"Skipped {metric} due to zero max value")  # Debug print

        st.write(plot_markdown(metric))
        fig_twitter.update_layout(title="Évolution de l'activité Twitter normalisée au fil du temps",
                                  xaxis_title="Date", yaxis_title="Valeur normalisée")
        st.plotly_chart(fig_twitter)


def plot_markdown(metrics, nombre=None):
    a = None
    if metrics == 'Tesla Stock Price':
        a = "Tesla est un pionnier dans le domaine des véhicules électriques. Chaque avancée technologique, " \
            "chaque nouvelle annonce, peut faire trembler le marché. Tesla ouvre de nouveaux horizons pour " \
            "l'industrie automobile, comme avec le Tesla Semi pour le transport de longue distance. "

    elif metrics == 'NASDAQ Composite':
        a = "Le NASDAQ, un miroir fidèle de l'économie technologique, reflète les hauts et les bas des entreprises " \
            "comme Tesla. Les fluctuations du NASDAQ sont un indicateur crucial pour les investisseurs qui scrutent " \
            "l'avenir de Tesla. "
    elif metrics == 'S&P 500':
        a = "Le S&P500 est un panier d’action contenant les 500 plus grosse capitalisation américaine. Il reflète " \
            "donc l’état de l’économie globale des Etats-Unis. A la différence du NASDAQ le S&P prend en compte un " \
            "plus large pans de l’économie. "

    elif metrics == 'Oil Price':
        a = "Le prix du pétrole, change constamment et influence profondément le marché des véhicules électriques. " \
            "Quand le prix du pétrole monte, l'intérêt pour les véhicules électriques comme ceux de Tesla augmente, " \
            "et vice versa. Le WTI permet donc de mieux prendre en compte la composante énergétique du prix de " \
            "l’action Tesla "

    elif metrics == 'Deaths Lag 1' or 'Deaths Lag 7' or 'Deaths Lag 30':
        a = "Ces événements tragiques, souvent sous les projecteurs des médias, éveillent des questions sur la " \
            "sécurité des technologies de pointe de Tesla, notamment l'Autopilot. Les investisseurs sont toujours " \
            "attentifs aux risques et peuvent réagir rapidement à ces nouvelles, entraînant des fluctuations parfois " \
            "significatives du prix de l'action. Ces incidents soulèvent des débats essentiels sur l'avenir de la " \
            "conduite automatisée et la responsabilité des constructeurs automobiles dans l'ère de l'innovation. " \
            "Tesla, en tant que leader dans ce domaine, se trouve souvent au centre de ces discussions, ce qui peut " \
            "affecter la confiance des investisseurs et donc la valorisation boursière de l'entreprise. La manière " \
            "dont Tesla répond à ces défis, par des améliorations de sécurité et une communication transparente, " \
            "est donc cruciale pour maintenir la confiance et stabiliser son action en bourse. (Les lag correspondent " \
            "au nombre de mort pas jour, par semaine et par mois) "

    elif metrics == 'Tweets of Elon Musk':
        a = "Chaque tweet d'Elon Musk est comme un levier puissant qui peut soulever ou abaisser le cours de " \
            "l'action Tesla en un instant. Que ce soit par ses annonces audacieuses, ses réflexions futuristes ou " \
            "même ses plaisanteries légères, chaque tweet est scruté, analysé, et souvent agit comme un " \
            "catalyseur sur le marché boursier. Nous avons pu notamment le constaté sur le marché des " \
            "cryptomonnaies avec le DogeCoin. Cette influence directe et parfois imprévisible d'Elon Musk sur le " \
            "marché est un facteur crucial pour comprendre et anticiper les mouvements de l'action Tesla. "

    elif metrics == 'Tweet with mention Tesla' or 'Tweet Likes':
        a = "Dans le monde numérique d'aujourd'hui, chaque like et retweet d'un tweet concernant Tesla forme une " \
            "vague qui se propage à travers le marché boursier. Ces interactions numériques sont des baromètres " \
            "puissants de l'intérêt et de la perception des consommateurs envers Tesla. Une accumulation rapide " \
            "de likes et de retweets, surtout lorsqu'ils portent sur des innovations, des succès ou des " \
            "partenariats stratégiques, peut signaler un accroissement de l'enthousiasme et de la confiance " \
            "envers l'entreprise, influençant positivement le cours de l'action. Inversement, une réaction " \
            "négative ou mitigée sur les réseaux sociaux, particulièrement en réponse à des controverses ou des " \
            "défis, peut susciter des inquiétudes chez les investisseurs et peser sur la valeur de l'action. Ces " \
            "données permettent donc de mieux appréhender les évènements de haute volatilité tout aussi bien que " \
            "le ressenti de l’opinion public pour Tesla. "

    return a


# Page for Estimation Results
def estimation_results_page(models, x_test, y_test):
    if not models:
        # Display a blank page with a gentle prompt
        st.write("Veuillez sélectionner des modèles pour afficher les résultats.")
        return

    st.title("Résultats d'Estimation")

    # Retrieve log-likelihood results
    results, predictions = get_log_likelihood_results(models, x_test, y_test)

    # Display the results using Markdown for better formatting
    st.markdown("""
    ### Résultats de la Vraisemblance Logarithmique

    La vraisemblance mesure à quel point un modèle spécifique est probable ou plausible étant donné les données 
    observées. En utilisant la vraisemblance, on peut comparer différents modèles pour voir lequel correspond le 
    mieux à la réalité complexe des données. Un modèle avec une forte vraisemblance capte les nuances et les 
    tendances cachées dans les données, permettant ainsi des prédictions plus précises et fiables. Cet outil 
    statistique est d’autant plus important pour la finance car les modèles déterminent les décisions 
    d’investissement.
    """, unsafe_allow_html=True)

    st.table(pd.DataFrame([results], index=['Log vraisemblance']))

    # Display the best model
    best_model = pd.Series(results).idxmax()
    best_llf = pd.Series(results).max()
    st.markdown(f"**Le modèle {best_model} est le plus performant** avec une log-vraisemblance de {round(best_llf, 5)}.")

    # Plot predictions for the best model
    st.markdown("#### Graphe des Prédictions vs Valeurs Réalisées")
    fig, ax = plt.subplots()
    tmp_df = pd.DataFrame({'Predictions': predictions[best_model], 'Realized': y_test}).reset_index(drop=True)
    st.line_chart(tmp_df)

    # SHAP values
    st.title('Analyse des Valeurs SHAP')
    explainer = shap.TreeExplainer(models['XGBoost'])
    shap_values = explainer.shap_values(x_test)
    st.markdown("""
    Les valeurs SHAP permettent de décomposer la contribution de chaque caractéristique (comme le prix du 
    pétrole, les tweets, etc.) à la prédiction finale. Les valeurs SHAP nous aident à comprendre non seulement 
    quelles données sont les plus importantes, mais aussi comment elles s'assemblent pour prédire le prix. Cette 
    compréhension fine permet aux analystes de mieux interpréter les prédictions de leur modèle. Les valeurs de 
    SHAP sont propres au modèle, une même variable peut avoir une valeur SHAP différentes selon le modèle ou les 
    composants de la régression. Ainsi, les valeurs SHAP permettent d’affiner la compréhension de notre modèle.
    """)

    # Feature Importance Plot
    st.markdown('#### Importance des Caractéristiques Basée sur les Valeurs SHAP')
    shap.summary_plot(shap_values, x_test, plot_type="bar")
    st.pyplot(bbox_inches='tight')
    plt.clf()

    # Summary Plot
    st.markdown('#### Résumé des Valeurs SHAP')
    shap.summary_plot(shap_values, x_test)
    st.pyplot(bbox_inches='tight')
    plt.clf()

    # Dependence Plot for a specific feature
    feature_names = x_test.columns.tolist()  # List of feature names
    st.markdown("""
    #### Plot de Dépendance SHAP

    Ce graphique représente un **plot de dépendance SHAP**. Les plots de dépendance SHAP montrent l'effet qu'une seule 
    caractéristique du dataset a sur la **prédiction du modèle**. Chaque point sur le graphique représente une observation. 
    L'axe horizontal représente la **valeur de la caractéristique**, tandis que l'axe vertical montre **l'impact de cette 
    caractéristique sur la prédiction du modèle**. En d'autres termes, cela nous aide à comprendre comment la valeur d'une 
    caractéristique spécifique influence la sortie du modèle. C'est un outil puissant pour **l'interprétation des modèles 
    de machine learning**, offrant une compréhension plus profonde des relations entre les caractéristiques et les 
    prédictions du modèle.
    """, unsafe_allow_html=True)
    selected_feature = st.selectbox('Sélectionner un élément pour le graphe de dépendance', feature_names)
    st.markdown(f'#### Dependence Plot pour {selected_feature}')
    shap.dependence_plot(selected_feature, shap_values, x_test)
    st.pyplot(bbox_inches='tight')
    plt.clf()


# Main Streamlit App
def main():
    st.sidebar.title("Dashboard Navigation")
    page = st.sidebar.selectbox("Select a page", ["Data Visualization", "Estimation Results"])

    if page == "Data Visualization":
        data_visualization_page()
    elif page == "Estimation Results":
        processed_data = pd.read_csv('datasets/DataTesla.csv')
        xgb_model, lasso_model, pc_regression_model, X_test, y_test = fit_models(processed_data)
        models = {'XGBoost': xgb_model, 'Lasso': lasso_model, 'PC Regression': pc_regression_model}
        estimation_results_page(models, X_test, y_test)


if __name__ == "__main__":
    main()
