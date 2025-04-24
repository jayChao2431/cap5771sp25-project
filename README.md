# cap5771sp25-project

### G20Climate-Economy Dynamics: Machine Learning Models and Interactive Visualization Dashboard

**Summary**

This project utilizes machine learning and interactive dashboards to investigate the interconnections between carbon emissions, energy sustainability, and economic growth among G20 countries from 2000 to 2020. By employing extensive datasets sourced from Kaggle, the research implements predictive modeling techniques to estimate per capita CO₂ emissions (CO2_per_capita) and GDP growth rates (gdp_growth). Additionally, classification models are utilized to categorize nations into high and low carbon emitters as well as different tiers of energy efficiency, thereby providing valuable insights into global sustainability trends. The machine learning framework incorporates both regression and classification methodologies, which are refined through cross-validation and feature selection processes. The findings are presented through an interactive dashboard, developed using Plotly Dash or Streamlit, which facilitates the exploration of dynamic correlations—such as the relationship between GDP per capita and emissions—and allows for the simulation of scenario-based outcomes. By integrating data science with environmental economics, this project establishes a scalable framework for evaluating climate policies and their socio-economic implications, highlighting the importance of data-driven approaches to decarbonization.

**Data Sources**

I utilized the open datasets from Kaggle:

[Global Data on Sustainable Energy](https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy)

This dataset presents an analysis of sustainable energy indicators and various relevant factors across all countries from the year 2000 to 2020.

[CO2 Emission by countries Year wise](https://www.kaggle.com/datasets/moazzimalibhatti/co2-emission-by-countries-year-wise-17502022)

This dataset will facilitate predictions regarding global warming.

[G20 Countries&#39; Currency Exchange Rates against USD](https://www.kaggle.com/datasets/mohamedharris/g20-countries-currency-exchange-rates-against-usd)

This dataset includes the currency exchange rates of G20 countries in relation to the US dollar, thereby facilitating a comprehensive understanding of the macroeconomic landscape.

**Project Goals**

Collect and preprocess data from multiple sources.

Perform Exploratory Data Analysis (EDA) to identify trends and insights.

Develop an interactive tool (dashboard or recommendation engine or Conversational agent) to visualize findings.

Utilize Python for data wrangling, visualization, and modeling.

**Interactive Dashboard**

GDP Predictions(G20)

**Methodology**

This project follows the **CRISP-DM (Cross Industry Standard Process for Data Mining)** framework:

**1. Data Collection & Preprocessing**

Acquire datasets from Kaggle

Data Cleaning and Preprocessing

**2. Exploratory Data Analysis (EDA)**

Visualization Analysis

**3. Feature Engineering & Modeling**

Machine learning skills

**Repository Structure**

```
cap5771sp25-project/
├── README.md
├── Data/
│   ├── CO2_emission_by_countries.csv
│   ├── G20_Exchange_Rates.csv
│   ├── global_data_on_sustainable_energy.csv
│   ├── New_CO2_emission_by_countries.csv
│   ├── New_G20_Exchange_Rates.csv
│   ├── New_global_data_on_sustainable_energy.csv
│   ├── Gdp_test_set.csv
│   ├── Final_merged_data.csv
├── Report/
│   ├── Reports/Milestone1
│   ├── Reports/Milestone2
│   ├── Reports/Milestone3
├── Scripts/
│   ├── Co2_data_Cleaning.ipynb
│   ├── Exchange_rate.ipynb
│   ├── final_combination.ipynb
│   ├── Renewable_energy.ipynb
│   ├── Milestone2.ipynb
│   ├── evaluate.ipynb
│   ├── gdp_dashboard.py
│   ├── Co2_emission_classification.py
│   ├── energy_efficiency_classifier.py
```

**Tech Stack**

**Programming Language:** Python

**Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Plotly, re ,mplcursors

**Data Storage:** SQLite (if necessary for structured queries)

**Version Control:** GitHub,Git

**Computing Resource: HiperGator (University of Florida's supercomputer)**

**Demo Video : [Demo Video](https://youtu.be/E9T_1hrhq1Q?si=uQHr1jYKa4NfocxH)**
