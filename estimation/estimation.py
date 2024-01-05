import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
import statsmodels.api as sm


class ModelFitter:
    """
    A class used to fit various statistical and machine learning models to a given dataset.

    Attributes
    ----------
    best_llf : float
        The best log-likelihood value obtained from the fitted models.
    best_model : str
        The name of the best performing model based on log-likelihood.
    data : DataFrame
        The dataset used for model fitting.
    target : str
        The name of the target variable in the dataset.
    models : dict
        A dictionary to store instances of different models.
    results : dict
        A dictionary to store the results (log-likelihood values) of the fitted models.
    predictions : dict
        A dictionary to store the predictions made by each model.
    target_test : Series
        The test target data.

    Methods
    -------
    fit_models():
        Fits a range of models to the data and stores their results and predictions.
    get_results():
        Returns the results of the model fitting.
    retrieve_max_llf_model():
        Identifies and prints the model with the highest log-likelihood.
    plot_predictions():
        Plots the predictions of the models.
    """

    def __init__(self, data, target):
        """
        Constructs all the necessary attributes for the ModelFitter object.

        Parameters
        ----------
        data : DataFrame
            The dataset to be used for model fitting.
        target : str
            The name of the target variable in the dataset.
        """
        self.best_llf = None
        self.best_model = None
        self.data = data
        self.target = target
        self.models = {
            'XGBoost': None,
            'OLS': None,
            'RandomForest': None,
            'Lasso': None,
            'PC Regression': None
        }
        self.results = {}
        self.predictions = {}
        self.target_test = {}

    def fit_models(self):
        """
        Fits various models to the dataset and stores their results and predictions.

        The method splits the data into training and test sets, fits different models,
        and calculates the log-likelihood for each model. It stores the fitted models,
        their predictions, and the log-likelihood values.
        """
        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Fit models
        self.models['XGBoost'] = XGBRegressor().fit(X_train, y_train)
        self.models['OLS'] = sm.OLS(y_train, sm.add_constant(X_train)).fit()
        self.models['RandomForest'] = RandomForestRegressor().fit(X_train, y_train)
        self.models['Lasso'] = Lasso().fit(X_train, y_train)

        self.y_test = y_test

        # For PC Regression
        pca = PCA(n_components=min(X_train.shape))
        X_train_pca = pca.fit_transform(X_train)
        self.models['PC Regression'] = sm.OLS(y_train, sm.add_constant(X_train_pca)).fit()

        # Store results
        for name, model in self.models.items():
            if name in ['OLS', 'PC Regression']:
                self.results[name] = model.llf
            else:
                # Calculate likelihood
                predictions = model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                likelihood = -0.5 * len(y_test) * np.log(2 * np.pi * mse) - 0.5 * len(y_test)
                self.results[name] = likelihood
                self.predictions[name] = predictions

    def get_results(self):
        """
        Returns the results of the fitted models.

        Returns
        -------
        dict
            A dictionary containing the log-likelihood values of the fitted models.
        """
        return self.results

    def retrieve_max_llf_model(self):
        """
        Identifies the best performing model based on log-likelihood.

        This method finds the model with the highest log-likelihood value and updates
        the best_model and best_llf attributes accordingly. It also prints the name
        of the best model and its log-likelihood.
        """
        results = self.get_results()
        self.best_model = pd.Series(results).idxmax()
        self.best_llf = pd.Series(results).max()
        print(f'Best model is: {self.best_model} with Log-likelihood of: {self.best_llf}')

    def plot_predictions(self):
        """
        Plots the predictions made by each model.

        This method generates a plot for each model's predictions against the actual
        values. It raises an exception if called before models are fitted.
        """

        if not hasattr(self, 'best_model'):
            raise Exception("Model not fitted")

        n_models = len(self.predictions.items())
        fig, ax = plt.subplots(nrows=n_models)
        fig.set_size_inches(20, 15)

        for _ in range(n_models):
            model = list(self.predictions.keys())[_]
            tmp_df = pd.DataFrame({'Predictions': self.predictions[model], 'Realized': self.y_test}).reset_index(
                drop=True)
            tmp_df.plot(ax=ax[_])
