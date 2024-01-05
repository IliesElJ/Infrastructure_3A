# Project Overview

The project is a comprehensive analysis and forecasting tool for Tesla stock prices. It is structured into three main parts, each contributing to the overall goal of understanding and predicting the stock's behavior.

## 1. Data Scraping

In the `Processing` folder, you'll find the initial step of the project, which involves scraping essential data from various sources:

### a. Financial Data
- **Objective:** Obtain financial data related to Tesla stock, S&P 500 and NASDAQ 100.
- **Date Range:** 01/01/2021 to 29/12/2022.
- **Implementation:** Utilizes web scraping techniques to collect crucial financial information.

### b. Twitter Data
- **Objective:** Gather data related to tweets mentioning either Twitter or Elon Musk.
- **Implementation:** Scrapes Twitter for relevant tweets, providing insights into the social media sentiment surrounding Tesla.

### c. Tesla Car Deaths Data
- **Objective:** Collect data about confirmed deaths in Tesla cars.
- **Implementation:** Extracts information regarding fatalities in Tesla vehicles, contributing to a holistic view of the company's impact.

## 2. Data Processing

After collecting the diverse datasets, the project moves to the `Data Processing` stage, where the information is consolidated and structured for further analysis:

### DataTesla.csv
- **Purpose:** A unified CSV file housing all processed data.
- **Contents:** Merged financial, Twitter, and Tesla car deaths data for comprehensive analysis.

## 3. Estimation

In this phase, the project employs various estimation models to understand and predict Tesla stock prices.

### ModelFitter Class
A dedicated Python class, `ModelFitter`, has been created to facilitate the testing of multiple predictive models. The class includes several models : OLS, RandomForest, PC Regression, Lasso and XGBoost.


## 4. Visualization

The file `dashboard_app.py` file focuses on creating insightful and informative visual representations of the data:

### Streamlit Dashboard
- **Objective:** Provide clear visualizations to enhance understanding.
- **Implementation:** Utilizes Streamlit to create and deploy a dashboard with two pages:
    - **Data Visualization Page:** Allows users to select various metrics and visualize them.
    - **Estimation Results Page:** Displays the results of the estimation using models defined in the `ModelFitter` class.

## How to Use

1. **Data Scraping:**
   - Navigate to the `Processing` folder.
   - Execute the relevant scripts for financial data, Twitter data, and Tesla car deaths data scraping.

2. **Data Processing:**
   - Open the `Processing` folder.
   - Run the script for consolidating data (`ConsolidateData.py` or similar).
   - Locate the resulting `DataTesla.csv` file.

3. **Estimation:**
   - Access the `Estimation` section.
   - Execute estimation models and algorithms (provide specific instructions if needed).

4. **Visualization:**
   - Explore the `Visualization` directory.
   - Run visualization scripts or notebooks to generate charts and graphs.

