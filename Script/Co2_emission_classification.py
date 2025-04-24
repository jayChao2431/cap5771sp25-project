import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Interactive Emission Classifier",
    page_icon="🧪",
    layout="wide"
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    h1, h2, h3 {
        color: #0e1117;
    }
    .stAlert {
        padding: 1rem;
    }
    .feature-info {
        font-size: 0.85rem;
        color: #6c757d;
        font-style: italic;
    }
    .impact-high {
        color: #dc3545;
        font-weight: bold;
    }
    .impact-medium {
        color: #fd7e14;
        font-weight: bold;
    }
    .impact-low {
        color: #28a745;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Main title and description
st.title("🧪 Interactive Emission Classifier")
st.markdown(
    "Adjust the parameters below to see how different factors affect a country's emission classification.")

# Function to extract scaler from model if possible


def try_extract_scaler_from_model(model):
    """Attempt to extract the scaler from various model types"""
    # Check if model is a pipeline with a scaler
    if hasattr(model, 'named_steps') and any(step.startswith('scale') for step in model.named_steps):
        for step_name in model.named_steps:
            if step_name.startswith('scale'):
                return model.named_steps[step_name], True

    # Check if model is a GridSearchCV or similar with best_estimator_
    if hasattr(model, 'best_estimator_'):
        if hasattr(model.best_estimator_, 'named_steps') and any(step.startswith('scale') for step in model.best_estimator_.named_steps):
            for step_name in model.best_estimator_.named_steps:
                if step_name.startswith('scale'):
                    return model.best_estimator_.named_steps[step_name], True

    # No scaler found in model
    return None, False

# Function to normalize input data without a scaler


def normalize_features(input_df, feature_ranges):
    """Normalize features to reasonable ranges based on domain knowledge"""
    normalized_df = input_df.copy()

    for column in normalized_df.columns:
        if column in feature_ranges:
            feature_range = feature_ranges[column]
            min_val = feature_range['min']
            max_val = feature_range['max']

            # Skip normalization if min == max to avoid division by zero
            if min_val == max_val:
                continue

            # Apply min-max scaling to [-1, 1] range, which is common for many ML models
            normalized_df[column] = 2 * \
                ((normalized_df[column] - min_val) / (max_val - min_val)) - 1

    return normalized_df


# Load the model
try:
    model = joblib.load(
        '/Users/chaotzuchieh/Documents/GitHub/cap5771sp25-project/Script/best_classifier_high_emission.pkl')
    st.sidebar.success("✅ Model loaded successfully!")

    # Try to extract scaler from model
    scaler, has_scaler = try_extract_scaler_from_model(model)

    if has_scaler:
        st.sidebar.success("✅ Scaler extracted from model!")
    else:
        st.sidebar.info(
            "ℹ️ No scaler found in model. Will use feature normalization.")

except FileNotFoundError:
    st.sidebar.error(
        "❌ Model file not found! Please ensure 'best_classifier_high_emission.pkl' is in the correct directory.")
    st.stop()  # Stop the app if the model isn't found
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {e}")
    st.stop()

# Get the feature names expected by the model
if hasattr(model, 'feature_names_in_'):
    feature_names = list(model.feature_names_in_)
elif hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'feature_names_in_'):
    feature_names = list(model.best_estimator_.feature_names_in_)
else:
    # If the model doesn't store feature names, use a predefined list of important features
    feature_names = [
        'CO2_per_capita',
        'Energy_per_CO2',
        'Renewable_ratio',
        'gdp_per_capita',
        'Primary energy consumption per capita (kWh/person)',
        'Access to clean fuels for cooking',
        'Access to electricity (% of population)'
    ]
    st.sidebar.warning(
        f"⚠️ Model doesn't store feature names. Using predefined features.")

# Define feature ranges, defaults, and influence on emissions
feature_ranges = {
    'CO2_per_capita': {
        'min': 0, 'max': 30, 'default': 5, 'unit': 'tons',
        'description': 'Annual carbon dioxide emissions per person',
        # positive means higher values = higher emissions
        'impact': 'high', 'direction': 'positive'
    },
    'Energy_per_CO2': {
        'min': 0, 'max': 15, 'default': 3, 'unit': 'kWh/kg',
        'description': 'Energy generated per unit of CO2 emissions',
        # negative means higher values = lower emissions
        'impact': 'high', 'direction': 'negative'
    },
    'Renewable_ratio': {
        'min': 0, 'max': 1, 'default': 0.25, 'unit': 'ratio',
        'description': 'Proportion of energy from renewable sources',
        'impact': 'high', 'direction': 'negative'
    },
    'gdp_per_capita': {
        'min': 500, 'max': 100000, 'default': 15000, 'unit': 'USD',
        'description': 'Gross Domestic Product per person',
        # relationship with emissions is complex
        'impact': 'medium', 'direction': 'mixed'
    },
    'Primary energy consumption per capita (kWh/person)': {
        'min': 1000, 'max': 100000, 'default': 25000, 'unit': 'kWh',
        'description': 'Annual energy consumption per person',
        'impact': 'medium', 'direction': 'positive'
    },
    'Access to clean fuels for cooking': {
        'min': 0, 'max': 100, 'default': 70, 'unit': '%',
        'description': 'Percentage of population with access to clean cooking fuels',
        'impact': 'medium', 'direction': 'negative'
    },
    'Access to electricity (% of population)': {
        'min': 20, 'max': 100, 'default': 90, 'unit': '%',
        'description': 'Percentage of population with access to electricity',
        'impact': 'low', 'direction': 'mixed'
    },
    'Population(2022)': {
        'min': 100000, 'max': 1500000000, 'default': 50000000, 'unit': 'people',
        'description': 'Total population',
        'impact': 'low', 'direction': 'positive'
    },
    'Latitude': {
        'min': -90, 'max': 90, 'default': 0, 'unit': 'degrees',
        'description': 'Geographic latitude coordinate',
        'impact': 'low', 'direction': 'mixed'
    },
    'Longitude': {
        'min': -180, 'max': 180, 'default': 0, 'unit': 'degrees',
        'description': 'Geographic longitude coordinate',
        'impact': 'low', 'direction': 'mixed'
    }
}

# Organize features by their impact level
high_impact_features = [
    f for f in feature_names if f in feature_ranges and feature_ranges[f]['impact'] == 'high']
medium_impact_features = [
    f for f in feature_names if f in feature_ranges and feature_ranges[f]['impact'] == 'medium']
low_impact_features = [
    f for f in feature_names if f in feature_ranges and feature_ranges[f]['impact'] == 'low']
other_features = [f for f in feature_names if f not in feature_ranges]

# Create tabs for different ways to interact with the model
tab1, tab2, tab3 = st.tabs(
    ["🎛️ Key Parameters", "📊 Advanced Settings", "ℹ️ About"])

# Initialize input_values dictionary
input_values = {}

with tab1:
    st.header("Key Emission Factors")
    st.markdown(
        "These factors have the strongest influence on emission classification:")

    # Create two columns for the key parameters
    col1, col2 = st.columns(2)

    with col1:
        # Display high impact features in the first column
        for feature in high_impact_features:
            if feature in feature_ranges:
                feature_info = feature_ranges[feature]

                # Create slider with appropriate range
                input_values[feature] = st.slider(
                    f"{feature} ({feature_info['unit']})",
                    min_value=float(feature_info['min']),
                    max_value=float(feature_info['max']),
                    value=float(feature_info['default']),
                    step=(feature_info['max'] - feature_info['min']) / 100,
                    help=feature_info['description']
                )

                # Show impact information
                impact_class = f"impact-{feature_info['impact']}"
                direction = "increases" if feature_info['direction'] == 'positive' else "decreases" if feature_info[
                    'direction'] == 'negative' else "influences"
                st.markdown(
                    f"<div class='feature-info'><span class='{impact_class}'>High impact:</span> Higher values {direction} emission classification</div>", unsafe_allow_html=True)

    with col2:
        # Display medium impact features in the second column
        # Only show top 2 medium impact features here
        for feature in medium_impact_features[:2]:
            if feature in feature_ranges:
                feature_info = feature_ranges[feature]

                # Create slider with appropriate range
                input_values[feature] = st.slider(
                    f"{feature} ({feature_info['unit']})",
                    min_value=float(feature_info['min']),
                    max_value=float(feature_info['max']),
                    value=float(feature_info['default']),
                    step=(feature_info['max'] - feature_info['min']) / 100,
                    help=feature_info['description']
                )

                # Show impact information
                impact_class = f"impact-{feature_info['impact']}"
                direction = "increases" if feature_info['direction'] == 'positive' else "decreases" if feature_info[
                    'direction'] == 'negative' else "influences"
                st.markdown(
                    f"<div class='feature-info'><span class='{impact_class}'>Medium impact:</span> Higher values {direction} emission classification</div>", unsafe_allow_html=True)

with tab2:
    st.header("Additional Factors")
    st.markdown(
        "These factors also influence a country's emission profile but have a lesser impact:")

    # Create three columns for a cleaner layout
    col1, col2, col3 = st.columns(3)

    # Distribute the remaining medium impact features and low impact features across columns
    remaining_features = medium_impact_features[2:] + \
        low_impact_features + other_features
    features_per_column = len(remaining_features) // 3 + 1

    for i, column in enumerate([col1, col2, col3]):
        with column:
            column_features = remaining_features[i *
                                                 features_per_column:(i+1)*features_per_column]
            for feature in column_features:
                if feature in feature_ranges:
                    feature_info = feature_ranges[feature]

                    # Create appropriate input based on feature type and range
                    input_values[feature] = st.slider(
                        f"{feature} ({feature_info['unit']})",
                        min_value=float(feature_info['min']),
                        max_value=float(feature_info['max']),
                        value=float(feature_info['default']),
                        step=(feature_info['max'] - feature_info['min']) / 100,
                        help=feature_info['description']
                    )

                    # Show impact information in small text
                    impact_class = f"impact-{feature_info['impact']}"
                    st.markdown(
                        f"<div class='feature-info'><span class='{impact_class}'>{feature_info['impact'].capitalize()} impact</span></div>", unsafe_allow_html=True)
                else:
                    # Generic slider for other features
                    input_values[feature] = st.slider(
                        feature,
                        min_value=-100.0, max_value=100.0, value=0.0, step=1.0
                    )
                    st.markdown(
                        "<div class='feature-info'>Impact unknown</div>", unsafe_allow_html=True)

with tab3:
    st.header("About This Tool")
    st.markdown("""
    This interactive dashboard uses a machine learning model to classify countries as high or low CO2 emitters based on various economic, demographic, and energy-related features.
    
    ### Key Features Affecting Classification:
    - **CO2 per capita**: Direct measure of emission levels per person
    - **Energy efficiency**: How much energy is produced per unit of CO2 emissions
    - **Renewable energy ratio**: Proportion of energy from renewable sources
    - **GDP per capita**: Economic output per person
    
    ### How to Use:
    1. Adjust the sliders to set different values for each feature
    2. See how changes affect the emission classification
    3. Explore which combinations of factors lead to high vs. low emission classifications
    
    ### About the Classification Approach:
    The dashboard uses feature normalization to ensure your inputs are properly scaled before being processed by the model. This approach doesn't require an external scaler file and provides consistent results regardless of your setup.
    
    ### Model Information:
    The underlying classification model was trained on global emissions data. It uses features related to energy consumption, economic indicators, and demographic factors to determine whether a country would be classified as a high or low emitter.
    """)

# Add a divider before the prediction section
st.markdown("---")
st.header("Make a Prediction")

# Confirm for user which features will be used
st.markdown(
    f"**Using {len(input_values)}/{len(feature_names)} features for prediction**")

# Create a button to trigger the prediction
predict_button = st.button("Classify Emission Level", type="primary")

if predict_button:
    # Display the values being used for prediction
    st.subheader("Input Values:")
    display_df = pd.DataFrame([input_values])
    st.dataframe(display_df.T.rename(columns={0: "Value"}))

    # Ensure all features are present in the input
    for feature in feature_names:
        if feature not in input_values:
            input_values[feature] = 0.0
            st.warning(
                f"Feature '{feature}' not provided, using default value of 0.")

    # Prepare the input data, ensuring order matches what the model expects
    input_df = pd.DataFrame(
        [{feature: input_values.get(feature, 0.0) for feature in feature_names}])

    # Process the input data for prediction
    try:
        if has_scaler:
            # Use the scaler extracted from the model
            input_scaled = scaler.transform(input_df)
            st.success(
                "✅ Using the scaler extracted from the model for accurate predictions.")
        else:
            # Use feature normalization based on domain knowledge
            input_scaled = normalize_features(input_df, feature_ranges)
            st.info("ℹ️ Using feature normalization for prediction. This provides reliable results without requiring an external scaler file.")
    except Exception as e:
        st.error(f"Error processing input data: {e}")
        st.stop()

    # Make the prediction
    try:
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.stop()

    # Display the result with a gauge or meter visualization
    st.subheader("Classification Result")

    # Display as a probability gauge
    if prediction == 1:
        emission_level = "High Emission Country 🔥"
        color = "red"
    else:
        emission_level = "Low Emission Country 🌿"
        color = "green"

    # Create columns for a cleaner layout
    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.markdown(f"### {emission_level}")
        st.markdown(f"""
        **Confidence Scores:**
        - Low Emission: {prediction_proba[0]:.2%}
        - High Emission: {prediction_proba[1]:.2%}
        """)

    with result_col2:
        # Create a gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_proba[1] * 100,
            title={'text': "High Emission Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))

        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Additional context about the prediction
    st.markdown("### Interpretation")

    if prediction == 1:
        st.markdown("""
        This combination of factors indicates a **high emission** profile, typical of:
        - Industrialized economies with high fossil fuel usage
        - High energy consumption per person
        - Lower than average renewable energy adoption
        """)
    else:
        st.markdown("""
        This combination of factors indicates a **low emission** profile, typical of:
        - Economies with significant renewable energy adoption
        - More efficient energy usage
        - Lower carbon intensity in their energy mix
        """)

    # Display key factors that influenced the prediction
    st.markdown("### Key Influencing Factors")

    # Identify factors with significant values
    emission_factors = []

    if 'CO2_per_capita' in input_values:
        if input_values['CO2_per_capita'] > 10:
            emission_factors.append("High CO2 emissions per capita")
        elif input_values['CO2_per_capita'] < 3:
            emission_factors.append("Low CO2 emissions per capita")

    if 'Renewable_ratio' in input_values:
        if input_values['Renewable_ratio'] > 0.5:
            emission_factors.append("High renewable energy usage")
        elif input_values['Renewable_ratio'] < 0.2:
            emission_factors.append("Low renewable energy usage")

    if 'Energy_per_CO2' in input_values:
        if input_values['Energy_per_CO2'] > 5:
            emission_factors.append("High energy efficiency")
        elif input_values['Energy_per_CO2'] < 2:
            emission_factors.append("Low energy efficiency")

    # Display the factors
    for factor in emission_factors:
        st.markdown(f"- {factor}")

    if not emission_factors:
        st.markdown("- Mixed factors with no dominant indicators")

    # Allow saving the prediction
    st.markdown("### Save this prediction")
    csv = pd.DataFrame([{**input_values, 'prediction': emission_level,
                       'high_emission_probability': prediction_proba[1]}]).to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download prediction as CSV",
        data=csv,
        file_name="emission_prediction.csv",
        mime="text/csv",
    )

# Footer with documentation link
st.markdown("---")
st.markdown("Interactive Emission Classifier | Created with Streamlit and Plotly")
