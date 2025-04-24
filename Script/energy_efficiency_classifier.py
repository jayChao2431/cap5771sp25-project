import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
# Import DecisionTreeClassifier to check model type
from sklearn.tree import DecisionTreeClassifier
import os  # Import os to check file paths
import warnings  # To suppress warnings if needed
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Suppress specific warnings if they clutter the output (optional)
# warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
# warnings.filterwarnings('ignore', category=FutureWarning)

# --- User's Feature Engineering Function ---


def create_new_features(df):
    """
    Create new features for climate and energy data analysis.
    This function creates features that provide unique information while avoiding redundancy.
    Requires base columns: 'CO2 emission (Tons)', 'gdp_per_capita', 'Population(2022)',
                           'Primary energy consumption per capita (kWh/person)',
                           'Area(Square kilometre)', 'Exchange_Rate', 'gdp_growth',
                           'Country', 'Year', 'Renewables (% equivalent primary energy)'
    """
    # Create a copy to avoid modifying the original DataFrame
    df_new = df.copy()

    # --- Check for required base columns ---
    required_base_cols = [
        'CO2 emission (Tons)', 'gdp_per_capita', 'Population(2022)',
        'Primary energy consumption per capita (kWh/person)',
        'Area(Square kilometre)', 'Exchange_Rate', 'gdp_growth',
        'Country', 'Year', 'Renewables (% equivalent primary energy)'
    ]
    missing_base_cols = [
        col for col in required_base_cols if col not in df_new.columns]
    if missing_base_cols:
        st.error(
            f"❌ Feature Engineering Error: Base data CSV is missing required columns for create_new_features: {', '.join(missing_base_cols)}")
        st.stop()  # Stop execution if base columns are missing

    # --- Feature Creation ---
    # Handle potential division by zero or NaN in denominators
    # CO2_per_GDP: Higher values indicate more carbon-intensive economies
    denominator_co2_gdp = (df_new['gdp_per_capita']
                           * df_new['Population(2022)'])
    df_new['CO2_per_GDP'] = np.where(
        denominator_co2_gdp != 0, df_new['CO2 emission (Tons)'] / denominator_co2_gdp, 0)

    # GDP_per_energy: Higher values indicate more economic value generated per unit of energy
    denominator_gdp_energy = df_new[
        'Primary energy consumption per capita (kWh/person)']
    df_new['GDP_per_energy'] = np.where(
        denominator_gdp_energy != 0, df_new['gdp_per_capita'] / denominator_gdp_energy, 0)

    # CO2_per_area: Higher values indicate higher emission density
    denominator_co2_area = df_new['Area(Square kilometre)']
    df_new['CO2_per_area'] = np.where(
        denominator_co2_area != 0, df_new['CO2 emission (Tons)'] / denominator_co2_area, 0)

    # Real Purchasing Power GDP
    denominator_ppp = df_new['Exchange_Rate']
    df_new['Real_Purchasing_Power_GDP'] = np.where(
        denominator_ppp != 0, df_new['gdp_per_capita'] / denominator_ppp, 0)

    # Ensure data is sorted by Country and Year for rolling/pct_change calculations
    df_new = df_new.sort_values(by=['Country', 'Year'])

    # Exchange Rate Volatility (3-year rolling std dev)
    df_new['Exchange_Rate_Volatility'] = df_new.groupby('Country')['Exchange_Rate'].transform(
        # Use min_periods=2
        lambda x: x.rolling(window=3, min_periods=2).std())

    # Economic External Sensitivity
    df_new['Economic_External_Sensitivity'] = df_new['Exchange_Rate_Volatility'] * \
        df_new['gdp_per_capita']

    # Exchange-Rate Adjusted GDP Growth
    exchange_rate_pct_change = df_new.groupby(
        'Country')['Exchange_Rate'].pct_change()
    df_new['Exchange_Adjusted_GDP_Growth'] = df_new['gdp_growth'] - \
        exchange_rate_pct_change

    # Time-based features
    # CO2_growth_rate: Year-over-year change in emissions
    df_new['CO2_growth_rate'] = df_new.groupby(
        'Country')['CO2 emission (Tons)'].pct_change()

    # GDP_growth_per_capita: Year-over-year change in GDP per capita
    df_new['GDP_growth_per_capita'] = df_new.groupby(
        'Country')['gdp_per_capita'].pct_change()

    # --- FIX for CO2_trend calculation ---
    # Get the first non-NaN CO2 value for each country (requires df sorted by Year)
    first_co2 = df_new.groupby('Country')['CO2 emission (Tons)'].transform(
        lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan
    )

    # Calculate trend relative to the first value
    # Handle cases where first_co2 is 0 or NaN
    df_new['CO2_trend'] = np.where(
        # Condition: first value is valid and non-zero
        (pd.notna(first_co2)) & (first_co2 != 0),
        (df_new['CO2 emission (Tons)'] - first_co2) /
        first_co2,  # Calculate trend
        # Default value if first value is 0 or NaN (or use np.nan if preferred)
        0
    )
    # Ensure the result is numeric, fill potential NaNs from calculation if needed
    df_new['CO2_trend'] = pd.to_numeric(
        df_new['CO2_trend'], errors='coerce').fillna(0)
    # --- End of FIX ---

    # Renewable_adoption_rate: Year-over-year change in renewable energy adoption
    df_new['Renewable_adoption_rate'] = df_new.groupby(
        'Country')['Renewables (% equivalent primary energy)'].pct_change()

    # --- NaN Handling ---
    # Fill NaNs created by rolling/pct_change, potentially with 0 or using ffill/bfill within groups
    # Using ffill (forward fill) then bfill (backward fill) within each country group after sorting
    group_cols_for_ffill = [
        'Exchange_Rate_Volatility', 'Economic_External_Sensitivity',
        'Exchange_Adjusted_GDP_Growth', 'CO2_growth_rate',
        'GDP_growth_per_capita', 'CO2_trend', 'Renewable_adoption_rate'
    ]
    for col in group_cols_for_ffill:
        if col in df_new.columns:  # Check if column exists before filling
            # Important: Ensure group operations respect the original grouping
            df_new[col] = df_new.groupby('Country', group_keys=False)[
                col].apply(lambda x: x.ffill().bfill())

    # Replace infinite values that might result from division by zero followed by pct_change
    df_new.replace([np.inf, -np.inf], 0, inplace=True)

    # Final check for any remaining NaNs in the engineered columns and fill with 0
    # (Could use mean/median imputation instead if appropriate)
    engineered_cols = [
        'CO2_per_GDP', 'GDP_per_energy', 'CO2_per_area', 'Real_Purchasing_Power_GDP',
        'Exchange_Rate_Volatility', 'Economic_External_Sensitivity', 'Exchange_Adjusted_GDP_Growth',
        'CO2_growth_rate', 'GDP_growth_per_capita', 'CO2_trend', 'Renewable_adoption_rate'
    ]
    # Add the base columns that are part of the final feature set if they might have NaNs
    final_feature_cols_to_check = list(set(engineered_cols + [
        'CO2_per_capita', 'gdp_per_capita', 'Access to clean fuels for cooking',
        'gdp_growth', 'CO2 emission (Tons)'
    ]))

    for col in final_feature_cols_to_check:
        if col in df_new.columns:
            if df_new[col].isnull().any():
                # st.sidebar.warning(f"Filling remaining NaNs in '{col}' with 0.") # Optional warning
                df_new[col].fillna(0, inplace=True)

    return df_new


class EnergyEfficiencyClassifier:
    def __init__(self,
                 model_path=None,
                 data_path=None):
        """
        Initialize the Energy Efficiency Classifier application.
        Loads base data, performs feature engineering, fits scaler, and loads model.

        Args:
            model_path (str): Path to the pre-trained classification model file.
            data_path (str): Path to the CSV file containing the BASE training data
                             BEFORE feature engineering.
        """
        # Use default paths if none provided
        if model_path is None:
            # Default to looking in the current directory
            model_path = 'best_energy_efficiency_classifier.pkl'

        if data_path is None:
            # Default to looking in the Data subdirectory
            data_path = os.path.join('Data', 'Final_merged_data.csv')

        st.set_page_config(
            page_title="Energy Efficiency Classifier",  # English Title
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        self._add_custom_styling()

        # --- Define the 10 features the MODEL expects ---
        # This list must match the features used when 'best_energy_efficiency_classifier.pkl' was trained.
        self.feature_names = [
            'CO2_growth_rate', 'CO2_per_capita', 'gdp_per_capita',
            'GDP_per_energy', 'Real_Purchasing_Power_GDP', 'high_emission',
            'CO2_per_area', 'Access to clean fuels for cooking',
            'gdp_growth', 'CO2 emission (Tons)'
        ]
        # English Info
        st.sidebar.info(
            f"Model expects 10 features: {', '.join(self.feature_names)}")

        # --- Load BASE data ---
        # Use the data_path passed during initialization
        # English Info
        st.sidebar.info(f"Loading base data from {data_path}...")
        if not os.path.exists(data_path):
            # English Error - Show path
            st.sidebar.error(f"❌ Base data file not found at: {data_path}")
            st.sidebar.error(
                "Ensure the 'Data' subfolder exists and contains 'Final_merged_data.csv'.")
            st.stop()
        try:
            base_df = pd.read_csv(data_path)
            # Convert relevant columns to numeric, coercing errors
            numeric_cols = ['CO2 emission (Tons)', 'gdp_per_capita', 'Population(2022)',
                            'Primary energy consumption per capita (kWh/person)',
                            'Area(Square kilometre)', 'Exchange_Rate', 'gdp_growth',
                            'Renewables (% equivalent primary energy)']
            for col in numeric_cols:
                if col in base_df.columns:
                    base_df[col] = pd.to_numeric(base_df[col], errors='coerce')
            # English Success
            st.sidebar.success("✅ Base data loaded successfully.")
        except Exception as e:
            # English Error
            st.sidebar.error(
                f"❌ Error loading base data CSV from {data_path}: {e}")
            st.stop()

        # --- Perform Feature Engineering ---
        st.sidebar.info("Performing feature engineering...")  # English Info
        # create_new_features function includes checks for required base columns
        engineered_df = create_new_features(base_df)
        # English Success
        st.sidebar.success("✅ Feature engineering complete.")

        # Add 'high_emission' column if it's expected by the model but not created by engineering
        # This assumes 'high_emission' might be in the base_df or needs default handling
        if 'high_emission' in self.feature_names and 'high_emission' not in engineered_df.columns:
            if 'high_emission' in base_df.columns:
                # English Info
                st.sidebar.info("Adding 'high_emission' from base data.")
                engineered_df['high_emission'] = base_df['high_emission'].fillna(
                    0)  # Example fillna
            else:
                # English Warning
                st.sidebar.warning(
                    "Model expects 'high_emission' but it's not in base data or engineered features. Adding default column (0).")
                engineered_df['high_emission'] = 0  # Default value if missing

        # --- Fit the Scaler on Engineered Data ---
        self.scaler = self._fit_scaler(
            engineered_df)  # Pass the engineered data

        # --- Load the Model ---
        # Model is expected in the same directory as the script
        self.model = self._load_model(model_path)

    def _add_custom_styling(self):
        """Add custom CSS for improved UI"""
        # (CSS code remains the same)
        st.markdown("""
        <style>
            .main { padding: 2rem; background-color: #f9f9f9; }
            .stMetric { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
            h1, h2, h3 { color: #0e1117; font-weight: bold; }
            .stAlert { padding: 1rem; margin-bottom: 1rem; }
            .stSidebar .stAlert { padding: 0.5rem; margin-top: 0.5rem; }
            /* Style for feature importance plot */
            .feature-importance-container {
                margin-top: 2rem;
                padding: 1rem;
                background-color: #ffffff;
                border-radius: 0.5rem;
                border: 1px solid #e0e0e0;
            }
        </style>
        """, unsafe_allow_html=True)

    # --- Fits scaler using the ENGINEERED data and the 10 expected features ---
    def _fit_scaler(self, engineered_df: pd.DataFrame):
        """
        Fit the StandardScaler object using the engineered data and the 10 features
        expected by the model.
        """
        st.sidebar.info("Fitting scaler on engineered data...")  # English Info
        try:
            # Verify the 10 required feature columns exist AFTER engineering
            missing_model_features = [
                col for col in self.feature_names if col not in engineered_df.columns]
            if missing_model_features:
                # English Error
                st.sidebar.error(
                    f"❌ Post-Engineering Error: Engineered data is missing columns the model expects: {', '.join(missing_model_features)}")
                st.sidebar.error(
                    # English Hint
                    f"👉 Check the 'create_new_features' function output and the 'self.feature_names' list ({self.feature_names}).")
                st.stop()

            # Initialize and fit the scaler using only the 10 features model expects
            scaler = StandardScaler()
            # Ensure data being fit is numeric
            scaler.fit(engineered_df[self.feature_names].apply(
                pd.to_numeric, errors='coerce').fillna(0))
            # English Success
            st.sidebar.success("✅ Scaler fitted successfully on 10 features!")
            return scaler

        except Exception as e:
            st.sidebar.error(f"❌ Error fitting scaler: {e}")  # English Error
            st.stop()

    def _load_model(self, model_path: str):
        """ Load the pre-trained machine learning model. """
        # (Function remains the same, but added English warning)
        if not os.path.exists(model_path):
            # English Error
            st.sidebar.error(f"❌ Model file not found: {model_path}")
            st.stop()
        try:
            model = joblib.load(model_path)
            if not (hasattr(model, 'predict') and hasattr(model, 'predict_proba')):
                # English Error
                st.sidebar.error(
                    "❌ Loaded object is not a valid classifier model.")
                st.stop()
            # --- Check if model has feature_importances_ attribute ---
            if not hasattr(model, 'feature_importances_'):
                # English Warning
                st.sidebar.warning(
                    "⚠️ Loaded model type does not directly support feature_importances_ (e.g., it might not be a tree-based model). Feature importance plot will not be available.")
            # English Success
            st.sidebar.success("✅ Model loaded successfully!")
            return model
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {e}")  # English Error
            st.stop()

    # --- Creates widgets for ALL 10 features ---
    def _create_input_widgets(self) -> dict:
        """ Create input widgets for all 10 features expected by the model. """
        input_values = {}
        st.sidebar.header("Adjust Input Features:")  # English Header

        # Configuration for ALL 10 features (with English labels/help text)
        feature_configs = {
            'CO2_growth_rate': {'label': 'CO2 Growth Rate (%)', 'type': 'number_input', 'min': -20.0, 'max': 20.0, 'default': 0.0, 'step': 0.1, 'help': "Estimated annual % change in CO2 emissions."},
            'CO2_per_capita': {'label': 'CO2 per Capita (tons)', 'type': 'number_input', 'min': 0.0, 'max': 50.0, 'default': 8.0, 'step': 0.1, 'help': "CO2 emissions per person."},
            'gdp_per_capita': {'label': 'GDP per Capita ($)', 'type': 'number_input', 'min': 500.0, 'max': 150000.0, 'default': 40000.0, 'step': 1000.0, 'help': "Gross Domestic Product per person."},
            'GDP_per_energy': {'label': 'GDP per Unit Energy ($/kWh)', 'type': 'number_input', 'min': 0.0, 'max': 20.0, 'default': 5.0, 'step': 0.1, 'help': "Estimated economic output per kWh energy consumed."},
            'Real_Purchasing_Power_GDP': {'label': 'Real PPP GDP (Est.)', 'type': 'number_input', 'min': 100.0, 'max': 150000.0, 'default': 45000.0, 'step': 1000.0, 'help': "Estimated GDP adjusted for purchasing power."},
            'high_emission': {'label': 'High Emission Category (0=Low, 1=High)', 'type': 'selectbox', 'options': [0, 1], 'default': 0, 'help': "Is the country considered a high emitter (based on CO2/capita)?"},
            'CO2_per_area': {'label': 'CO2 per Area (tons/km²)', 'type': 'number_input', 'min': 0.0, 'max': 5000.0, 'default': 100.0, 'step': 10.0, 'help': "Estimated CO2 emissions relative to land area."},
            'Access to clean fuels for cooking': {'label': 'Access to Clean Fuels (%)', 'type': 'number_input', 'min': 0.0, 'max': 100.0, 'default': 85.0, 'step': 1.0, 'help': "Percentage of population with access to clean cooking fuels."},
            'gdp_growth': {'label': 'GDP Growth Rate (%)', 'type': 'number_input', 'min': -15.0, 'max': 15.0, 'default': 2.0, 'step': 0.1, 'help': "Annual percentage change in GDP."},
            'CO2 emission (Tons)': {'label': 'Total CO2 Emissions (Million Tons)', 'type': 'number_input', 'min': 0.0, 'max': 12000.0, 'default': 500.0, 'step': 100.0, 'help': "Total annual CO2 emissions in million tons."}
        }

        # Create input widgets in the sidebar for ALL 10 features
        for feature in self.feature_names:
            if feature in feature_configs:
                config = feature_configs[feature]
                label = config.get('label', feature)  # Use English label
                help_text = config.get('help', None)  # Use English help text
                widget_key = f"input_{feature}"  # Unique key for each widget

                if config['type'] == 'number_input':
                    input_values[feature] = st.sidebar.number_input(
                        label,
                        min_value=config['min'],
                        max_value=config['max'],
                        value=config['default'],
                        step=config['step'],
                        help=help_text,
                        key=widget_key
                    )
                elif config['type'] == 'selectbox':
                    input_values[feature] = st.sidebar.selectbox(
                        label,
                        options=config['options'],
                        index=config['options'].index(config['default']),
                        help=help_text,
                        key=widget_key
                    )
            else:
                # English Warning
                st.sidebar.warning(
                    f"Configuration missing for feature: {feature}")
        return input_values

    # --- Prepares data using the FULL 10 feature list ---
    def _prepare_input_data(self, input_values: dict) -> np.ndarray:
        """
        Prepare and scale input data using the scaler fitted on engineered data.
        Uses the full 10 feature list expected by the model.
        """
        try:
            # Convert the dictionary of input values to a DataFrame
            # The keys in input_values should match self.feature_names
            input_df = pd.DataFrame([input_values])

            # Ensure columns are in the exact order expected by the scaler/model
            input_df = input_df[self.feature_names]

            # Ensure all input data is numeric before scaling
            input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)

            # --- Use transform with the fitted scaler ---
            input_scaled = self.scaler.transform(input_df)

            return input_scaled
        except KeyError as e:
            # English Error
            st.error(
                f"❌ Feature Mismatch Error preparing input: Input feature '{e}' missing or mistyped. Check input widgets and feature_names.")
            return None
        except ValueError as e:
            # English Error
            st.error(
                f"❌ Data Type Error preparing input: Could not convert input to numeric. Check values. Details: {e}")
            return None
        except Exception as e:
            st.error(f"❌ Error preparing input data: {e}")  # English Error
            return None

    def _predict(self, input_scaled: np.ndarray):
        """
        Make prediction and calculate probabilities using the 10 scaled features.
        Handles the mapping from 3 classes to 2 output classes.
        """
        try:
            if input_scaled is None:
                # English Error
                st.error("❌ Prediction Error: Input data is missing.")
                return None
            if input_scaled.ndim == 1:
                input_scaled = input_scaled.reshape(1, -1)

            n_features = input_scaled.shape[1]
            if n_features != 10:
                # English Error
                st.error(
                    f"❌ Internal Error: Prepared data has {n_features} features, but model expects 10.")
                return None

            # Make predictions
            prediction = self.model.predict(input_scaled)[0]
            prediction_proba = self.model.predict_proba(input_scaled)[0]

            # --- Logic to map 3 classes to 2 output classes ---
            final_prediction = 0 if prediction in [0, 1] else 1

            # Adjust probabilities
            if len(prediction_proba) == 3:
                adjusted_proba = [
                    # Probability for new Low (0)
                    prediction_proba[0] + prediction_proba[1],
                    # Probability for new High (1)
                    prediction_proba[2]
                ]
            elif len(prediction_proba) == 2:
                # English Warning
                st.warning(
                    "Model predicted only 2 classes, expected 3 for probability adjustment.")
                adjusted_proba = prediction_proba
                final_prediction = prediction  # Assume model's 0/1 matches our 0/1
            else:
                # English Error
                st.error(
                    f"❌ Unexpected number of probabilities ({len(prediction_proba)}) from model.")
                return None

            return {'prediction': final_prediction, 'probabilities': adjusted_proba}

        except AttributeError:
            # English Error
            st.error(
                "❌ Prediction Error: Model object missing 'predict' or 'predict_proba'.")
            return None
        except ValueError as e:
            if "features" in str(e) and "expecting" in str(e):
                # English Error
                st.error(
                    f"❌ Prediction Error: Feature count mismatch. Input has {input_scaled.shape[1]}, model expects different count. Details: {e}")
            else:
                st.error(f"❌ Prediction value error: {e}")  # English Error
            return None
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")  # English Error
            return None

    def _display_results(self, prediction_result: dict):
        """ Display classification results. """
        # (Function remains the same, but uses English labels)
        class_labels = {0: "Low", 1: "High"}  # English Labels
        # English Header
        st.subheader("⚡ Energy Efficiency Classification Result")
        if prediction_result is None:
            st.warning("Prediction could not be completed.")  # English Warning
            return
        prediction = prediction_result['prediction']
        probabilities = prediction_result['probabilities']
        if prediction in class_labels:
            predicted_label = class_labels[prediction]
            if prediction == 0:  # Low Efficiency
                st.metric(label="Predicted Efficiency Level", value=predicted_label,
                          delta="Potential for Improvement", delta_color="inverse")  # English Metric Labels
                # English Message
                st.error(
                    f"Based on the inputs, the predicted energy efficiency is **{predicted_label}**.", icon="📉")
            else:  # High Efficiency
                st.metric(label="Predicted Efficiency Level", value=predicted_label,
                          delta="Efficient Performance", delta_color="normal")  # English Metric Labels
                # English Message
                st.success(
                    f"Based on the inputs, the predicted energy efficiency is **{predicted_label}**.", icon="🏆")
            st.write("---")
            st.write("#### Probability Distribution")  # English Header
            col1, col2 = st.columns(2)
            # Display adjusted probabilities
            with col1:
                # English Label
                st.info(
                    f"Probability of **Low** Efficiency: **{probabilities[0]:.2%}**")
            with col2:
                # English Label
                st.success(
                    f"Probability of **High** Efficiency: **{probabilities[1]:.2%}**")
        else:
            # English Error
            st.error(
                f"❌ Internal Error: Unexpected prediction value ({prediction}).")

    # --- NEW: Display Feature Importance Plot ---
    def _display_feature_importance(self):
        """Display the model's feature importance plot"""
        st.write("---")
        st.subheader("📊 Model Feature Importance")  # English Header
        st.markdown("This chart shows how much each input feature contributes to the model's prediction. Features with higher importance have a greater impact on the classification (Low/High efficiency).")  # English Description

        # Check if the model has the feature_importances_ attribute
        if hasattr(self.model, 'feature_importances_'):
            try:
                importances = self.model.feature_importances_
                # Create a DataFrame of feature names and importance scores, then sort
                feature_importance_df = pd.DataFrame({
                    'Feature': self.feature_names,  # English Column Name
                    'Importance': importances      # English Column Name
                }).sort_values(by='Importance', ascending=False)

                # Create the plot
                fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size
                ax.barh(
                    feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
                ax.set_xlabel('Importance Score')  # English Label
                # English Title
                ax.set_title(
                    'Importance of Features for Energy Efficiency Classification')
                ax.invert_yaxis()  # Show most important features at the top
                plt.tight_layout()  # Adjust layout to prevent labels overlapping

                # Display the plot in Streamlit
                st.pyplot(fig)

                # (Optional) Display the table data
                # st.dataframe(feature_importance_df)

            except Exception as e:
                # English Error
                st.error(f"❌ Could not plot feature importances: {e}")
        else:
            # English Warning
            st.warning(
                "⚠️ The currently loaded model does not directly provide feature importance information (it might not be a tree-based model).")

    def run(self):
        """ Main application run method """
        st.title("⚡ Interactive Energy Efficiency Classifier")  # English Title
        st.markdown(  # English Description
            """
            This tool predicts whether a country's energy efficiency profile is **Low** or **High** based on various economic and environmental indicators (including engineered features).
            Adjust the input features in the sidebar to see how they influence the classification.
            *(Note: 'Low' efficiency here combines the original 'Low' and 'Medium' categories from the model training.)*
            """
        )
        st.markdown("---")

        if not self.model or not self.scaler:
            # English Error
            st.error("App init failed: Model or scaler not properly initialized.")
            st.stop()

        input_values = self._create_input_widgets()
        input_scaled = self._prepare_input_data(input_values)

        # Display prediction results in the main panel
        main_col, _ = st.columns([2, 1])  # Create columns to control width
        with main_col:
            if input_scaled is not None:
                prediction_result = self._predict(input_scaled)
                if prediction_result:
                    self._display_results(prediction_result)
                else:
                    # English Warning
                    st.warning(
                        "Prediction failed. See errors above or in sidebar.")
            else:
                # English Warning
                st.warning(
                    "Data preparation failed. See errors above or in sidebar.")

            # --- Display feature importance plot in the main panel ---
            self._display_feature_importance()

        st.sidebar.markdown("---")
        st.sidebar.markdown(
            "💡 **Tip:** Experiment with different input values, observe changes in prediction and probabilities, and compare with the feature importance chart.")  # English Tip


def main():
    """ Main entry point for the Streamlit application """
    try:
        # Define file paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(
            script_dir, 'best_energy_efficiency_classifier.pkl')
        data_path = os.path.join(script_dir, 'Data', 'Final_merged_data.csv')

        # Check for files in current directory first
        if not os.path.exists(model_path):
            # If not in script_dir, try current working directory
            model_path = 'best_energy_efficiency_classifier.pkl'

        if not os.path.exists(data_path):
            # If not in script_dir/Data, try ./Data
            data_path = os.path.join('Data', 'Final_merged_data.csv')

            # If still not found, try just current directory
            if not os.path.exists(data_path):
                data_path = 'Final_merged_data.csv'

        # Verify files exist
        if not os.path.exists(model_path):
            st.error(f"Fatal Error: Model file not found at: {model_path}")
            st.info(
                "Please place the model file 'best_energy_efficiency_classifier.pkl' in the same directory as this script.")
            st.stop()

        if not os.path.exists(data_path):
            st.error(f"Fatal Error: Data file not found at: {data_path}")
            st.info("Please ensure that 'Final_merged_data.csv' is available either in a 'Data' subfolder or in the same directory as this script.")
            st.stop()

        # Initialize and run application
        classifier_app = EnergyEfficiencyClassifier(
            model_path=model_path, data_path=data_path)
        classifier_app.run()

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        import traceback
        st.error(traceback.format_exc())


if __name__ == "__main__":
    main()
