import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import os

# Set page configuration
st.set_page_config(
    page_title="GDP Growth Prediction Dashboard",
    page_icon="📊",
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
    .feature-slider {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main application title
st.title("🌍 GDP Growth Prediction Dashboard")
st.markdown("This dashboard shows predictions from a machine learning model for GDP growth based on environmental and economic factors.")

# Sidebar for controls
st.sidebar.header("📋 Controls")

# Function to load data and model


@st.cache_data
def load_data_and_model():
    try:
        # Check if file exists
        data_path = '/Users/chaotzuchieh/Documents/GitHub/cap5771sp25-project/Data/Final_merged_data.csv'
        if not os.path.exists(data_path):
            st.sidebar.error(f"❌ Data file not found: {data_path}")
            # Create example data
            df = create_demo_data()
            st.sidebar.warning("⚠️ Using demo data")
            return df, None, []

        # Load the data
        df = pd.read_csv(data_path)
        st.sidebar.success("✅ Data loaded successfully")

        # Load pre-trained model
        try:
            model_path = '/Users/chaotzuchieh/Documents/GitHub/cap5771sp25-project/Script/best_model_gdp_growth.pkl'
            if not os.path.exists(model_path):
                st.sidebar.error(f"❌ Model file not found: {model_path}")
                model = create_simple_model(df)
                st.sidebar.warning("⚠️ Using simple linear regression model")
            else:
                try:
                    model = joblib.load(model_path)
                    st.sidebar.success("✅ Model loaded successfully")
                except Exception as e:
                    st.sidebar.error(f"❌ Error loading model: {e}")
                    model = create_simple_model(df)
                    st.sidebar.warning(
                        "⚠️ Version incompatibility. Using simple linear regression model")

            # Define features required by the model
            required_features = [
                'CO2_growth_rate',          # Missing feature
                'Population(2022)',         # Available in the dataset
                'Access to clean fuels for cooking',  # Available in the dataset
                'Energy_per_CO2',           # Available in the dataset
                'GDP_per_energy',           # Missing feature
                'CO2_per_capita',           # Available in the dataset
                'Real_Purchasing_Power_GDP',  # Missing feature
                'Longitude',                # Available in the dataset
                # Available in the dataset
                'Electricity from fossil fuels (TWh)',
                'CO2_per_area'              # Missing feature
            ]

            # Create missing features
            df = create_missing_features(df)

            # Verify all required features exist
            missing_features = [
                f for f in required_features if f not in df.columns]
            if missing_features:
                st.sidebar.error(
                    f"❌ Missing required features: {missing_features}")
                if isinstance(model, LinearRegression):
                    pass  # Simple model already created
                else:
                    model = create_simple_model(df)
                    st.sidebar.warning(
                        "⚠️ Using simple linear regression model due to missing features")
            else:
                st.sidebar.success("✅ All required features are available")

            # Prepare data for prediction
            X = df[required_features].copy()

            # Handle missing values
            if X.isnull().any().any():
                st.sidebar.warning(
                    "⚠️ Missing values found in features, filling with zeros")
                X = X.fillna(0)

            # Make predictions
            if isinstance(model, LinearRegression):
                # Use simple model
                simple_features = [f for f in ['CO2_per_capita', 'Energy_per_CO2', 'gdp_per_capita']
                                   if f in df.columns]
                X_simple = df[simple_features].fillna(0)
                df['predicted_gdp_growth'] = model.predict(X_simple)
                st.sidebar.success(
                    f"✅ Predictions made using simple linear regression for {len(df)} rows")
                return df, model, simple_features
            else:
                # Use pre-trained model
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                df['predicted_gdp_growth'] = model.predict(X_scaled)
                st.sidebar.success(
                    f"✅ Predictions made using ML model for {len(df)} rows")
                return df, model, required_features

        except Exception as e:
            st.sidebar.error(f"❌ Model processing error: {e}")
            model = create_simple_model(df)
            simple_features = [f for f in ['CO2_per_capita', 'Energy_per_CO2', 'gdp_per_capita']
                               if f in df.columns]
            df['predicted_gdp_growth'] = model.predict(
                df[simple_features].fillna(0))
            st.sidebar.warning("⚠️ Using simple linear regression model")
            return df, model, simple_features

    except Exception as e:
        st.sidebar.error(f"❌ Data processing error: {str(e)}")
        # Create minimal demo dataset
        df = create_demo_data()
        st.sidebar.warning("⚠️ Using demo data due to errors")
        model = create_simple_model(df)
        simple_features = ['CO2_per_capita',
                           'Energy_per_CO2', 'gdp_per_capita']
        return df, model, simple_features


def create_demo_data():
    """Create a demo dataset"""
    df = pd.DataFrame({
        'Country': ['United States', 'China', 'Japan', 'Germany', 'India'],
        'Year': [2022, 2022, 2022, 2022, 2022],
        'gdp_growth': [2.1, 3.0, 1.7, 1.8, 7.2],
        'gdp_per_capita': [65000, 12000, 40000, 48000, 2500],
        'CO2_per_capita': [15.0, 8.0, 9.0, 9.5, 2.0],
        'Energy_per_CO2': [2.0, 1.5, 2.5, 3.0, 1.0],
        'Population(2022)': [330, 1400, 125, 83, 1390],
        'Access to clean fuels for cooking': [99.9, 65.0, 100.0, 100.0, 50.0],
        'Longitude': [-95.7, 104.2, 138.2, 10.4, 78.9],
        'Electricity from fossil fuels (TWh)': [2800, 5200, 750, 500, 1100],
        'predicted_gdp_growth': [2.2, 3.1, 1.6, 1.9, 7.0]
    })
    return df


def create_simple_model(df):
    """Create a simple linear regression model"""
    features = [f for f in ['CO2_per_capita', 'Energy_per_CO2', 'gdp_per_capita']
                if f in df.columns]

    if not features or 'gdp_growth' not in df.columns:
        # If unable to create model, return a dummy model
        model = LinearRegression()
        model.coef_ = np.array([0.0001, -0.05, 0.0])
        model.intercept_ = 2.0
        return model

    X = df[features].fillna(0)
    y = df['gdp_growth'].fillna(df['gdp_growth'].mean())

    model = LinearRegression()
    model.fit(X, y)
    return model


def create_missing_features(df):
    """Create missing features"""
    # 1. CO2_growth_rate
    if 'CO2_growth_rate' not in df.columns:
        df['CO2_growth_rate'] = 0

    # 2. GDP_per_energy
    if 'GDP_per_energy' not in df.columns:
        if 'gdp_per_capita' in df.columns and 'Primary energy consumption per capita (kWh/person)' in df.columns:
            df['GDP_per_energy'] = df['gdp_per_capita'] / \
                df['Primary energy consumption per capita (kWh/person)']
            df['GDP_per_energy'] = df['GDP_per_energy'].fillna(
                0).replace([np.inf, -np.inf], 0)
        else:
            df['GDP_per_energy'] = 0

    # 3. Real_Purchasing_Power_GDP
    if 'Real_Purchasing_Power_GDP' not in df.columns:
        if 'gdp_per_capita' in df.columns and 'Exchange_Rate' in df.columns:
            df['Real_Purchasing_Power_GDP'] = df['gdp_per_capita'] / \
                df['Exchange_Rate']
            df['Real_Purchasing_Power_GDP'] = df['Real_Purchasing_Power_GDP'].fillna(
                0)
        else:
            df['Real_Purchasing_Power_GDP'] = 0

    # 4. CO2_per_area
    if 'CO2_per_area' not in df.columns:
        if 'CO2 emission (Tons)' in df.columns and 'Area(Square kilometre)' in df.columns:
            df['CO2_per_area'] = df['CO2 emission (Tons)'] / \
                df['Area(Square kilometre)']
            df['CO2_per_area'] = df['CO2_per_area'].fillna(
                0).replace([np.inf, -np.inf], 0)
        else:
            df['CO2_per_area'] = 0

    return df


def forecast_future_gdp(country_data, model, features, forecast_years, feature_changes):
    """Forecast future GDP growth"""
    # Get the latest year data as baseline
    latest_data = country_data.sort_values(
        'Year', ascending=False).iloc[0].copy()

    # Create prediction dataframe for future years
    future_data = []

    for year in forecast_years:
        # Copy baseline data
        future_row = latest_data.copy()
        future_row['Year'] = year

        # Apply user-specified feature changes
        for feature, change_per_year in feature_changes.items():
            if feature in future_row:
                # Calculate years since latest data
                years_diff = year - latest_data['Year']
                # Apply change rate
                future_row[feature] = future_row[feature] * \
                    (1 + change_per_year/100) ** years_diff

        future_data.append(future_row)

    # Convert future data to DataFrame
    future_df = pd.DataFrame(future_data)

    # Use model to make predictions
    if isinstance(model, LinearRegression):
        # Simple linear model
        simple_features = [f for f in features if f in future_df.columns]
        X_future = future_df[simple_features].values.reshape(
            len(future_df), len(simple_features))
        future_df['predicted_gdp_growth'] = model.predict(X_future)
    else:
        # Pre-trained model
        X_future = future_df[features].values.reshape(
            len(future_df), len(features))
        # Standardize features
        scaler = StandardScaler()
        X_future_scaled = scaler.fit_transform(X_future)
        future_df['predicted_gdp_growth'] = model.predict(X_future_scaled)

    return future_df


# Load data and model
df, model, features = load_data_and_model()

# Check if predictions were made successfully
if 'predicted_gdp_growth' not in df.columns:
    st.error(
        "❌ Failed to generate predictions. Please check the error messages in the sidebar.")
    st.stop()

# Main content area
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Overview", "🔍 Country Analysis", "🔮 Future Forecast", "🧩 Feature Importance", "📊 Data Explorer"])

with tab1:
    st.header("Model Performance Overview")

    # Calculate error metrics
    try:
        mae = mean_absolute_error(df['gdp_growth'], df['predicted_gdp_growth'])
        rmse = np.sqrt(mean_squared_error(
            df['gdp_growth'], df['predicted_gdp_growth']))
        r2 = r2_score(df['gdp_growth'], df['predicted_gdp_growth'])

        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Absolute Error", f"{mae:.4f}", "Lower is better")
        col2.metric("Root Mean Squared Error",
                    f"{rmse:.4f}", "Lower is better")
        col3.metric("R² Score", f"{r2:.4f}", "Higher is better")
    except Exception as e:
        st.error(f"❌ Error calculating metrics: {str(e)}")

    # Create actual vs predicted values plot
    st.subheader("Actual vs Predicted GDP Growth")

    try:
        fig = px.scatter(df, x='gdp_growth', y='predicted_gdp_growth',
                         hover_data=['Country', 'Year'],
                         labels={'gdp_growth': 'Actual GDP Growth (%)',
                                 'predicted_gdp_growth': 'Predicted GDP Growth (%)'})

        fig.update_layout(
            height=600,
            width=800
        )

        # Add a diagonal reference line (perfect predictions)
        fig.add_shape(
            type="line", line=dict(dash="dash"),
            x0=min(df['gdp_growth'].min(), df['predicted_gdp_growth'].min()),
            y0=min(df['gdp_growth'].min(), df['predicted_gdp_growth'].min()),
            x1=max(df['gdp_growth'].max(), df['predicted_gdp_growth'].max()),
            y1=max(df['gdp_growth'].max(), df['predicted_gdp_growth'].max())
        )

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error creating scatter plot: {str(e)}")

    # Global statistics
    st.subheader("Prediction Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Actual GDP Growth")
        st.dataframe(df['gdp_growth'].describe())

    with col2:
        st.markdown("### Predicted GDP Growth")
        st.dataframe(df['predicted_gdp_growth'].describe())

with tab2:
    st.header("Country-Specific Analysis")

    # Country selection
    countries = sorted(df['Country'].unique())
    selected_country = st.selectbox("Select a country", countries)

    # Filter data for selected country
    country_data = df[df['Country'] == selected_country].sort_values('Year')

    if not country_data.empty:
        # Country info
        st.subheader(f"{selected_country} Overview")

        # Latest year data
        latest_year_data = country_data.sort_values(
            'Year', ascending=False).iloc[0]
        latest_year = int(latest_year_data['Year'])

        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest GDP Growth",
                    f"{latest_year_data['gdp_growth']:.2f}%", f"Year: {latest_year}")
        col2.metric("Latest Prediction", f"{latest_year_data['predicted_gdp_growth']:.2f}%",
                    f"{latest_year_data['predicted_gdp_growth'] - latest_year_data['gdp_growth']:.2f}%")
        col3.metric("GDP per Capita",
                    f"${latest_year_data['gdp_per_capita']:,.0f}")
        col4.metric("CO₂ per Capita",
                    f"{latest_year_data['CO2_per_capita']:.2f} tons")

        # Time series plot
        st.subheader(f"GDP Growth Over Time: {selected_country}")

        try:
            fig = px.line(country_data, x='Year', y=['gdp_growth', 'predicted_gdp_growth'],
                          labels={
                              'value': 'GDP Growth (%)', 'variable': 'Type'},
                          color_discrete_map={
                'gdp_growth': '#0068c9',
                'predicted_gdp_growth': '#ff5252'
            })

            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="GDP Growth (%)",
                legend_title="",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error creating time series plot: {str(e)}")

        # Data table
        st.subheader("Historical Data")

        display_cols = ['Year', 'gdp_growth', 'predicted_gdp_growth',
                        'gdp_per_capita', 'CO2_per_capita', 'Energy_per_CO2']

        # Make sure all columns exist
        available_cols = [
            col for col in display_cols if col in country_data.columns]

        st.dataframe(country_data[available_cols].sort_values(
            'Year', ascending=False), use_container_width=True)
    else:
        st.warning(f"No data available for {selected_country}")

with tab3:
    st.header("Future GDP Growth Forecast")

    # Country selection
    future_country = st.selectbox(
        "Select country to forecast", countries, key="future_country")

    # Filter data for selected country
    country_data = df[df['Country'] == future_country].sort_values('Year')

    if not country_data.empty:
        # Get the latest year
        latest_year = int(country_data['Year'].max())

        # Forecast year range
        st.subheader("Forecast Settings")

        col1, col2 = st.columns(2)

        with col1:
            # Fixed: ensure forecast_start is always > latest_year
            forecast_start = st.number_input("Forecast start year",
                                             min_value=latest_year + 1,
                                             max_value=latest_year + 10,
                                             value=latest_year + 1)

        with col2:
            # Fixed: ensure forecast_end can never be less than forecast_start
            forecast_end = st.number_input("Forecast end year",
                                           min_value=forecast_start,
                                           max_value=latest_year + 20,
                                           # Default to 5 years (start + 4)
                                           value=forecast_start + 4)

        forecast_years = list(range(forecast_start, forecast_end + 1))

        # Feature change assumptions
        st.subheader("Feature Change Assumptions")
        st.markdown(
            "Set the annual percentage change. For example: 2% means increasing by 2% each year")

        feature_col1, feature_col2 = st.columns(2)

        feature_changes = {}

        with feature_col1:
            st.markdown("### Economic Indicators")
            if 'gdp_per_capita' in df.columns:
                gdp_per_capita_change = st.slider("GDP per Capita Annual Growth (%)",
                                                  min_value=-10.0, max_value=15.0, value=2.0,
                                                  step=0.5, key="gdp_per_capita_change")
                feature_changes['gdp_per_capita'] = gdp_per_capita_change

            if 'GDP_per_energy' in df.columns:
                gdp_per_energy_change = st.slider("GDP per Energy Annual Change (%)",
                                                  min_value=-5.0, max_value=10.0, value=1.0,
                                                  step=0.5, key="gdp_per_energy_change")
                feature_changes['GDP_per_energy'] = gdp_per_energy_change

        with feature_col2:
            st.markdown("### Environmental Indicators")
            if 'CO2_per_capita' in df.columns:
                co2_per_capita_change = st.slider("CO2 per Capita Annual Change (%)",
                                                  min_value=-10.0, max_value=5.0, value=-1.0,
                                                  step=0.5, key="co2_per_capita_change")
                feature_changes['CO2_per_capita'] = co2_per_capita_change

            if 'Energy_per_CO2' in df.columns:
                energy_per_co2_change = st.slider("Energy Efficiency Annual Change (%)",
                                                  min_value=-5.0, max_value=10.0, value=2.0,
                                                  step=0.5, key="energy_per_co2_change")
                feature_changes['Energy_per_CO2'] = energy_per_co2_change

        # Population change
        if 'Population(2022)' in df.columns:
            pop_change = st.slider("Population Annual Growth (%)",
                                   min_value=-2.0, max_value=5.0, value=0.5,
                                   step=0.1, key="pop_change")
            feature_changes['Population(2022)'] = pop_change

        # Forecast button
        forecast_button = st.button("Generate Forecast")

        if forecast_button:
            with st.spinner("Forecasting future GDP growth..."):
                # Forecast future GDP
                future_df = forecast_future_gdp(
                    country_data,
                    model,
                    features,
                    forecast_years,
                    feature_changes
                )

                # Combine historical data with forecast for display
                historical_data = country_data[[
                    'Year', 'gdp_growth', 'predicted_gdp_growth']].copy()
                historical_data['data_type'] = 'Historical'

                future_display = future_df[[
                    'Year', 'predicted_gdp_growth']].copy()
                future_display['gdp_growth'] = np.nan  # No actual GDP growth
                future_display['data_type'] = 'Forecast'

                combined_data = pd.concat([historical_data, future_display])

                # Display forecast summary
                st.subheader(
                    f"GDP Growth Forecast for {future_country} ({forecast_start}-{forecast_end})")

                avg_predicted = future_display['predicted_gdp_growth'].mean()

                # Key metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Average GDP Growth (Forecast Period)",
                            f"{avg_predicted:.2f}%")
                col2.metric("Historical Average GDP Growth",
                            f"{historical_data['gdp_growth'].mean():.2f}%")
                col3.metric(
                    "Change", f"{avg_predicted - historical_data['gdp_growth'].mean():.2f}%")

                # Create forecast chart
                fig = go.Figure()

                # Add historical actual GDP growth
                fig.add_trace(go.Scatter(
                    x=historical_data['Year'],
                    y=historical_data['gdp_growth'],
                    mode='lines+markers',
                    name='Historical GDP Growth',
                    line=dict(color='#0068c9')
                ))

                # Add historical predicted GDP growth
                fig.add_trace(go.Scatter(
                    x=historical_data['Year'],
                    y=historical_data['predicted_gdp_growth'],
                    mode='lines+markers',
                    name='Historical Predictions',
                    line=dict(color='#ff5252', dash='dot')
                ))

                # Add future predicted GDP growth
                fig.add_trace(go.Scatter(
                    x=future_display['Year'],
                    y=future_display['predicted_gdp_growth'],
                    mode='lines+markers',
                    name='Future Forecast',
                    line=dict(color='#00bd9d')
                ))

                # Calculate approximate y-axis range for annotation
                y_min = min(
                    historical_data['gdp_growth'].min(
                    ) if not historical_data['gdp_growth'].empty else 0,
                    historical_data['predicted_gdp_growth'].min(
                    ) if not historical_data['predicted_gdp_growth'].empty else 0,
                    future_display['predicted_gdp_growth'].min(
                    ) if not future_display['predicted_gdp_growth'].empty else 0
                )

                y_max = max(
                    historical_data['gdp_growth'].max(
                    ) if not historical_data['gdp_growth'].empty else 10,
                    historical_data['predicted_gdp_growth'].max(
                    ) if not historical_data['predicted_gdp_growth'].empty else 10,
                    future_display['predicted_gdp_growth'].max(
                    ) if not future_display['predicted_gdp_growth'].empty else 10
                )

                y_range = y_max - y_min
                annotation_y_position = y_max - 0.05 * y_range

                fig.update_layout(
                    title=f"GDP Growth Forecast for {future_country}",
                    xaxis_title="Year",
                    yaxis_title="GDP Growth (%)",
                    height=500,
                    hovermode="x unified"
                )

                # Add vertical line separating historical and forecast
                fig.add_vline(x=latest_year + 0.5, line_width=1,
                              line_dash="dash", line_color="gray")

                # Fixed: use calculated y-position instead of accessing yaxis.range
                fig.add_annotation(
                    x=latest_year + 0.5,
                    y=annotation_y_position,
                    text="Forecast Start",
                    showarrow=False
                )

                st.plotly_chart(fig, use_container_width=True)

                # Display forecast data table
                st.subheader("Forecast Details")
                st.dataframe(future_display[['Year', 'predicted_gdp_growth']].sort_values(
                    'Year'), use_container_width=True)

                # Warning disclaimer
                st.info(
                    "Note: This forecast is based on your specified assumptions. Actual future values may be affected by many other factors.")
    else:
        st.warning(f"No data available for {future_country}")

with tab4:
    st.header("Feature Importance Analysis")

    # Try to extract feature importance
    try:
        if model is not None and not isinstance(model, str):
            # Try different ways to get feature importance based on model type
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                has_importance = True
            elif hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'feature_importances_'):
                importance = model.best_estimator_.feature_importances_
                has_importance = True
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
                has_importance = True
            elif hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'coef_'):
                importance = np.abs(model.best_estimator_.coef_)
                has_importance = True
            else:
                has_importance = False

            if has_importance:
                # Create a dataframe for feature importance
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)

                # Create a bar chart
                fig = px.bar(importance_df, y='Feature', x='Importance',
                             orientation='h',
                             title='Feature Importance',
                             color='Importance',
                             color_continuous_scale='Viridis')

                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

                # Display importance values
                st.subheader("Feature Importance Values")
                st.dataframe(importance_df, use_container_width=True)

                # Feature descriptions
                st.subheader("Feature Descriptions")

                feature_descriptions = {
                    'CO2_growth_rate': 'Annual growth rate of CO2 emissions',
                    'Population(2022)': 'Total population in 2022',
                    'Access to clean fuels for cooking': 'Percentage of population with access to clean fuels for cooking',
                    'Energy_per_CO2': 'Energy consumption per unit of CO2 emissions',
                    'GDP_per_energy': 'GDP per unit of energy consumption',
                    'CO2_per_capita': 'CO2 emissions per person',
                    'Real_Purchasing_Power_GDP': 'GDP adjusted for purchasing power parity',
                    'Longitude': 'Geographic longitude coordinate',
                    'Electricity from fossil fuels (TWh)': 'Electricity generated from fossil fuels in terawatt-hours',
                    'CO2_per_area': 'CO2 emissions per square kilometer'
                }

                for feature, importance in zip(importance_df['Feature'], importance_df['Importance']):
                    with st.expander(f"{feature} ({importance:.4f})"):
                        st.write(feature_descriptions.get(
                            feature, "No description available"))
            else:
                st.info(
                    "Feature importance information is not available for this model type")
        else:
            # If using demo model, show mock feature importance
            st.info("Using demo feature importance values (not from a real ML model)")

            # Create mock feature importance
            mock_features = [
                'GDP_per_energy',
                'CO2_per_capita',
                'Energy_per_CO2',
                'Access to clean fuels for cooking',
                'CO2_per_area',
                'Population(2022)',
                'Electricity from fossil fuels (TWh)',
                'Longitude',
                'CO2_growth_rate'
            ]

            mock_importance = [0.25, 0.18, 0.15,
                               0.12, 0.10, 0.08, 0.06, 0.04, 0.02]

            # Create a dataframe for feature importance
            importance_df = pd.DataFrame({
                'Feature': mock_features,
                'Importance': mock_importance
            }).sort_values('Importance', ascending=False)

            # Create a bar chart
            fig = px.bar(importance_df, y='Feature', x='Importance',
                         orientation='h',
                         title='Feature Importance (Demo Values)',
                         color='Importance',
                         color_continuous_scale='Viridis')

            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Display importance values
            st.subheader("Feature Importance Values")
            st.dataframe(importance_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error extracting feature importance: {str(e)}")

with tab5:
    st.header("Data Explorer")

    # Filter options
    st.subheader("Filter Data")

    col1, col2, col3 = st.columns(3)

    # Country filter
    with col1:
        country_filter = st.multiselect(
            "Select countries", countries, default=None)

    # Year range filter - Modified to handle single year case
    with col2:
        min_year = int(df['Year'].min())
        max_year = int(df['Year'].max())

        # Add a year offset if min_year equals max_year to avoid slider error
        if min_year == max_year:
            st.write(f"Data only available for year: {min_year}")
            year_min = min_year
            year_max = min_year
        else:
            year_range = st.slider(
                "Year range", min_year, max_year, (min_year, max_year))
            year_min = year_range[0]
            year_max = year_range[1]

    # GDP range filter
    with col3:
        min_gdp = float(df['gdp_per_capita'].min())
        max_gdp = float(df['gdp_per_capita'].max())

        # Add a small offset if min_gdp equals max_gdp to avoid slider error
        if abs(min_gdp - max_gdp) < 0.01:
            st.write(f"GDP per capita value: ${min_gdp:,.0f}")
            gdp_min = min_gdp
            gdp_max = min_gdp
        else:
            gdp_range = st.slider("GDP per capita range ($)",
                                  min_value=int(min_gdp),
                                  max_value=int(max_gdp),
                                  value=(int(min_gdp), int(max_gdp)))
            gdp_min = gdp_range[0]
            gdp_max = gdp_range[1]

    # Apply filters
    filtered_df = df.copy()

    if country_filter:
        filtered_df = filtered_df[filtered_df['Country'].isin(country_filter)]

    filtered_df = filtered_df[(filtered_df['Year'] >= year_min) &
                              (filtered_df['Year'] <= year_max)]

    filtered_df = filtered_df[(filtered_df['gdp_per_capita'] >= gdp_min) &
                              (filtered_df['gdp_per_capita'] <= gdp_max)]

    # Display filtered data
    st.subheader(f"Filtered Data ({len(filtered_df)} rows)")

    # Column selection
    all_columns = df.columns.tolist()
    default_columns = ['Country', 'Year', 'gdp_growth', 'predicted_gdp_growth',
                       'gdp_per_capita', 'CO2_per_capita', 'Energy_per_CO2']

    # Make sure all default columns exist
    default_columns = [col for col in default_columns if col in all_columns]

    selected_columns = st.multiselect(
        "Select columns to display", all_columns, default=default_columns)

    if selected_columns:
        st.dataframe(filtered_df[selected_columns], use_container_width=True)
    else:
        st.dataframe(filtered_df, use_container_width=True)

    # Download option
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name="filtered_gdp_data.csv",
        mime="text/csv",
    )

# Footer
st.markdown("---")
st.markdown("GDP Growth Prediction Dashboard | Created with Streamlit and Plotly")
st.info("Using demo feature importance values (not from a real ML model)")
