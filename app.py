# ============================================
#        STREAMLIT APPLICATION SCRIPT (V2 Features + Plots)
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os # To check file existence

# --- Configuration ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Model/Data Paths
MODEL_PATH = os.path.join(APP_DIR, 'salary_prediction_model_enhanced.joblib')
AUX_DATA_PATH = os.path.join(APP_DIR, 'auxiliary_data_enhanced.joblib')
# Plot Filenames (MUST match filenames saved in train_model.py)
HIST_TREND_PLOT = os.path.join(APP_DIR, 'salary_trends_historical_enhanced.png')
FEAT_IMPORTANCE_PLOT = os.path.join(APP_DIR, 'feature_importances_enhanced.png')
CORR_HEATMAP_PLOT = os.path.join(APP_DIR, 'correlation_heatmap_enhanced.png')
SALARY_EXP_PLOT = os.path.join(APP_DIR, 'salary_vs_experience_enhanced.png')

# --- Load Model and Data Functions ---
# [ Keep load_model and load_aux_data functions as they were ]
@st.cache_resource # Cache the loaded model object
def load_model(path):
    """Loads the saved model pipeline."""
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}. Run train_model.py.")
        return None
    try:
        model = joblib.load(path)
        if not hasattr(model, 'predict') or not hasattr(model, 'original_feature_names_'):
             st.error(f"Loaded object from {path} is not a valid model pipeline.")
             return None
        print("Model loaded successfully.")
        return model
    except Exception as e: st.error(f"Error loading model: {e}"); return None

@st.cache_data # Cache the loaded auxiliary data dictionary
def load_aux_data(path):
    """Loads the saved auxiliary data needed for the app."""
    if not os.path.exists(path):
        st.error(f"Aux data file not found: {path}. Run train_model.py.")
        return None
    try:
        aux_data = joblib.load(path)
        required_keys = [ # V2 keys
            'unique_roles', 'unique_industries', 'unique_locations', 'unique_perk_levels', 'unique_lifecycles', 'unique_company_sizes',
            'unique_company_tiers', 'unique_performance_levels', 'unique_negotiation_levels',
            'last_hist_year', 'avg_hist_experience', 'avg_hist_bonus_pct',
            'training_features', 'forecasted_econ_plus_industry_data']
        missing = [key for key in required_keys if key not in aux_data];
        if missing: st.error(f"Aux data missing keys: {missing}"); return None
        print("Auxiliary data loaded successfully.")
        unique_list_keys = ['unique_roles', 'unique_industries', 'unique_locations', 'unique_perk_levels', 'unique_lifecycles', 'unique_company_sizes', 'unique_company_tiers', 'unique_performance_levels', 'unique_negotiation_levels']
        for key in unique_list_keys:
             if key in aux_data and isinstance(aux_data.get(key), np.ndarray): aux_data[key] = aux_data[key].tolist()
        if 'forecasted_econ_plus_industry_data' in aux_data: # Ensure forecast Year is int
            forecast_df = aux_data['forecasted_econ_plus_industry_data'];
            if forecast_df is not None and not forecast_df.empty and 'Year' in forecast_df.columns:
                forecast_df.dropna(subset=['Year'], inplace=True);
                if not forecast_df['Year'].empty:
                    try: forecast_df['Year'] = forecast_df['Year'].astype(int); aux_data['forecasted_econ_plus_industry_data'] = forecast_df; print("Converted forecast 'Year' to int.")
                    except Exception as e: st.error(f"Could not convert forecast 'Year' to int: {e}"); return None
                else: print("Warning: Forecast 'Year' empty after NaN drop.")
            elif forecast_df is not None and not forecast_df.empty: st.error("Forecast data missing 'Year' column!"); return None
        return aux_data
    except Exception as e: st.error(f"Error loading aux data: {e}"); return None


# --- Prediction Function ---
# [ Keep predict_scenario_trend_enhanced function as it was ]
import streamlit as st
import pandas as pd
import numpy as np
import joblib
# Note: No matplotlib needed directly in this function if plots are separate

# --- Prediction Function (Handles V2 features, uses derived years, fixed .clip()) ---
def predict_scenario_trend_enhanced(model, scenario_inputs, aux_data):
    """
    Generates salary predictions for a user-defined scenario (including V2 features)
    using years derived from the loaded forecast data.

    Args:
        model: The trained pipeline object (e.g., loaded from joblib).
        scenario_inputs (dict): A dictionary containing user selections for all features
                                 (role, industry, location, experience, V2 features).
        aux_data (dict): A dictionary containing auxiliary data loaded from joblib,
                         including 'training_features' list and
                         'forecasted_econ_plus_industry_data' DataFrame.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with predictions and calculated growth rates.
            - pd.DataFrame: DataFrame with relevant economic context for display.
            Returns (pd.DataFrame(), pd.DataFrame()) on failure.
    """
    if model is None or aux_data is None:
        st.warning("Model or auxiliary data not loaded. Cannot predict.")
        print("ERROR: predict_scenario_trend_enhanced called with None model or aux_data.")
        return pd.DataFrame(), pd.DataFrame()

    print("\n--- Starting Prediction Generation ---")

    # --- 1. Extract required data from aux_data ---
    try:
        training_features = aux_data['training_features']
        forecasted_data = aux_data['forecasted_econ_plus_industry_data'].copy()
        last_hist_year = aux_data['last_hist_year']
    except KeyError as e:
        st.error(f"Auxiliary data is missing a required key: {e}")
        print(f"ERROR: Auxiliary data missing key: {e}")
        return pd.DataFrame(), pd.DataFrame()
    except AttributeError: # If forecast data isn't a DataFrame
         st.error("Loaded forecast data is not a valid DataFrame.")
         print("ERROR: aux_data['forecasted_econ_plus_industry_data'] is not a DataFrame.")
         return pd.DataFrame(), pd.DataFrame()

    # --- 2. Derive future years DIRECTLY from the loaded forecast data ---
    if 'Year' not in forecasted_data.columns or forecasted_data['Year'].isnull().all():
         st.error("Cannot predict: Loaded forecast data has no valid 'Year' column.")
         print("ERROR: No valid 'Year' column in forecast data.")
         return pd.DataFrame(), pd.DataFrame()
    try:
        # Ensure 'Year' is integer type again (safety check)
        forecasted_data['Year'] = forecasted_data['Year'].astype(int)
    except Exception as e:
         st.error(f"Failed to ensure forecast 'Year' is integer: {e}")
         print(f"ERROR: Failed to convert forecast 'Year' to int: {e}")
         return pd.DataFrame(), pd.DataFrame()

    available_future_years = sorted(forecasted_data['Year'].unique())
    if not available_future_years:
        st.warning("Could not find any valid future years in the loaded forecast data.")
        print("ERROR: No unique years found in forecast data 'Year' column.")
        return pd.DataFrame(), pd.DataFrame()

    future_years_list = available_future_years
    print(f"DEBUG: Using future years derived from forecast data: {future_years_list}")
    first_prediction_year = future_years_list[0]
    # current_real_year = pd.Timestamp.now().year # Use this if basing exp calc on current time

    # --- 3. Generate Scenario DataFrame for Future Years ---
    scenarios_list = []
    for year in future_years_list:
        # Calculate experience relative to the first prediction year
        # Ensures experience increases linearly from the start of the prediction window
        exp_at_year = scenario_inputs['current_experience'] + (year - first_prediction_year)

        scenario = {
            # Base features
            'Job Role': scenario_inputs['role'],
            'Industry': scenario_inputs['industry'],
            'Location': scenario_inputs['location'],
            'Years of Experience': max(0, round(exp_at_year)), # Ensure non-negative experience
            'Year': year, # Use derived year (already int)
            # Simulated V1 features
            'Perk_Level': scenario_inputs['perk_level'],
            'Target_Bonus_Pct': scenario_inputs['bonus_pct'],
            'Job_Role_Lifecycle': scenario_inputs['lifecycle'],
            # Simulated V2 features from user input
            'Company_Size': scenario_inputs['company_size'],
            'Company_Tier': scenario_inputs['company_tier'],
            'Individual_Performance': scenario_inputs['performance'],
            'Negotiation_Skill': scenario_inputs['negotiation'],
            'Has_Hot_Skill': int(scenario_inputs['has_hot_skill']) # Convert boolean to 0/1
        }
        # Check if all expected keys from scenario_inputs were used
        scenarios_list.append(scenario)

    if not scenarios_list:
        st.warning("Could not generate future year scenarios list."); print("ERROR: scenarios_list is empty.")
        return pd.DataFrame(), pd.DataFrame()

    future_scenarios_df = pd.DataFrame(scenarios_list)

    # --- 4. Prepare data for prediction (Merge, Add Missing, Select/Order) ---
    try:
        print("--- Debugging Before Merge ---")
        print("Scenario Years:", future_scenarios_df['Year'].unique(), "Dtype:", future_scenarios_df['Year'].dtype)
        print("Forecast Years:", forecasted_data['Year'].unique(), "Dtype:", forecasted_data['Year'].dtype)
        print("--- Performing Merge ---")
        prediction_input_df = future_scenarios_df.merge(forecasted_data, on='Year', how='left', suffixes=('', '_forecast'))
        prediction_input_df = prediction_input_df.loc[:,~prediction_input_df.columns.duplicated()] # Remove potential duplicate columns from merge
        print("Merge complete. Shape after merge:", prediction_input_df.shape)

        # Check for merge issues (NaNs in key forecast columns)
        key_forecast_cols = ['US_CPI_YoY', 'India_GDP_YoY', 'Industry_Avg_Salary_LPA']
        key_forecast_cols_exist = [col for col in key_forecast_cols if col in prediction_input_df.columns]
        if key_forecast_cols_exist:
            print("NaN counts in key forecast cols after merge:\n", prediction_input_df[key_forecast_cols_exist].isnull().sum())
            if prediction_input_df[key_forecast_cols_exist].isnull().all().all():
                 st.warning("WARNING: Key forecasted data columns are all NaN after merging. Check forecast data generation in train_model.py.")
                 print("WARNING: Key forecast columns are all NaN after merge.")
                 st.dataframe(forecasted_data[['Year']+key_forecast_cols_exist].head()) # Show problematic data

        # Add missing columns (likely engineered _Lag1 features) expected by the model
        missing_cols = [col for col in training_features if col not in prediction_input_df.columns]
        if missing_cols:
            print(f"DEBUG: Adding {len(missing_cols)} missing columns expected by model: {missing_cols}")
            for col in missing_cols:
                prediction_input_df[col] = np.nan # Imputer in pipeline will handle these

        # Final check: Ensure all required features are present before prediction
        missing_required = [col for col in training_features if col not in prediction_input_df.columns]
        if missing_required:
            st.error(f"CRITICAL Error: Required training features are still missing before prediction: {missing_required}")
            print(f"ERROR: Final required features missing: {missing_required}. Available: {prediction_input_df.columns.tolist()}")
            return pd.DataFrame(), pd.DataFrame()

        print(f"DEBUG: Final columns being sent to model ({len(training_features)}): {training_features[:10]}...")
        # Select and order columns exactly as the model was trained
        X_pred = prediction_input_df[training_features]

        # --- 5. Predict ---
        st.info(f"Predicting salaries for {len(X_pred)} future years...")
        predicted_salaries = model.predict(X_pred)
        print("Prediction complete.")

        # --- 6. Format Results ---
        result_df = future_scenarios_df.copy()
        result_df['Predicted_Salary_LPA'] = predicted_salaries
        # Ensure predictions are non-negative using correct clip syntax
        result_df['Predicted_Salary_LPA'] = result_df['Predicted_Salary_LPA'].clip(0, None)
        result_df = result_df.sort_values(by='Year')
        # Calculate growth rate based on the potentially clipped salaries
        result_df['Predicted_Growth_Rate (%)'] = result_df['Predicted_Salary_LPA'].pct_change().fillna(0) * 100
        print("Result formatting complete.")

        # --- 7. Extract Economic Context for Display ---
        cols_to_show_context = ['Year', 'US_CPI_YoY', 'India_GDP_YoY', 'Industry_Avg_Salary_LPA', 'Industry_Growth_Rate_Pct']
        # Find which of the desired context columns actually exist in the forecast data
        cols_exist = [col for col in cols_to_show_context if col in forecasted_data.columns]
        # Filter original forecast data by the derived years and existing columns
        econ_context_df = forecasted_data[forecasted_data['Year'].isin(future_years_list)][cols_exist]
        # Set Year as index for display if it exists
        if 'Year' in econ_context_df.columns:
            econ_context_df = econ_context_df.set_index('Year')

        print("--- Prediction Generation Successful ---")
        return result_df, econ_context_df

    # --- Error Handling for Prediction Preparation ---
    except Exception as e:
        st.error(f"An error occurred during prediction preparation or execution: {e}")
        print(f"ERROR: Exception during prediction preparation/execution: {e}")
        # Optionally print traceback for detailed debugging
        # import traceback
        # print(traceback.format_exc())
        return pd.DataFrame(), pd.DataFrame()


# --- Streamlit UI Application ---
def run_app():
    """Defines and runs the Streamlit user interface."""
    st.set_page_config(layout="wide")
    st.title("Future Salary Trend Predictor")
    st.markdown(""" Enter your profile details""")

    model = load_model(MODEL_PATH)
    aux_data = load_aux_data(AUX_DATA_PATH)

    if model is not None and aux_data is not None:
        col1, col2 = st.columns([1, 3]) # Input column, Output column

        with col1:
            # --- Input Widgets (remain the same) ---
            st.subheader("👤 Base Details"); selected_role = st.selectbox("Job Role", sorted(aux_data.get('unique_roles', ['N/A']))); selected_industry = st.selectbox("Industry", sorted(aux_data.get('unique_industries', ['N/A']))); selected_location = st.selectbox("Location", sorted(aux_data.get('unique_locations', ['N/A']))); default_exp = int(round(aux_data.get('avg_hist_experience', 5))); current_experience = st.number_input("Current Years of Experience", 0, 40, default_exp, 1)
            st.markdown("---"); st.subheader("🏢 Company & Role Context"); selected_comp_size = st.selectbox("Company Size", sorted(aux_data.get('unique_company_sizes', ['N/A']))); selected_comp_tier = st.selectbox("Company Tier", sorted(aux_data.get('unique_company_tiers', ['N/A']))); selected_perk = st.select_slider("Company Perk Level", sorted(aux_data.get('unique_perk_levels', ['Medium'])), 'Medium'); default_bonus = float(round(aux_data.get('avg_hist_bonus_pct', 10.0), 1)); selected_bonus = st.number_input("Target Bonus (%)", 0.0, 100.0, default_bonus, 0.5); unique_lifecycles_list = aux_data.get('unique_lifecycles', []); default_lifecycle_index = unique_lifecycles_list.index('Mature') if 'Mature' in unique_lifecycles_list else 0; selected_lifecycle = st.selectbox("Job Role Lifecycle", sorted(unique_lifecycles_list), index=default_lifecycle_index); selected_hot_skill = st.checkbox("Possess In-Demand Niche Skill?")
            st.markdown("---"); st.subheader("✨ Individual Factors"); selected_performance = st.select_slider("Individual Performance", sorted(aux_data.get('unique_performance_levels', ['Medium'])), 'Medium'); selected_negotiation = st.select_slider("Negotiation Skill", sorted(aux_data.get('unique_negotiation_levels', ['Medium'])), 'Medium')
            predict_button = st.button("Predict Salary Trend", type="primary", key="predict_btn")

        with col2:
            # --- Prediction Display Logic (remains the same) ---
            if predict_button:
                scenario_inputs = {'role': selected_role, 'industry': selected_industry, 'location': selected_location, 'current_experience': current_experience, 'perk_level': selected_perk, 'bonus_pct': selected_bonus, 'lifecycle': selected_lifecycle, 'company_size': selected_comp_size, 'company_tier': selected_comp_tier, 'performance': selected_performance, 'negotiation': selected_negotiation, 'has_hot_skill': selected_hot_skill }
                with st.spinner("Calculating predictions..."):
                    predictions_df, econ_context_df = predict_scenario_trend_enhanced(model, scenario_inputs, aux_data)
                if not predictions_df.empty:
                    st.subheader(f"Predicted Trend for '{selected_role}' in {selected_location}")
                    st.markdown(f"*(Tier: {selected_comp_tier}, Size: {selected_comp_size}, Perf: {selected_performance}, Nego: {selected_negotiation}, HotSkill: {selected_hot_skill})*")
                    # [Plotting code remains the same]
                    fig, ax = plt.subplots(figsize=(10, 5)); line1, = ax.plot(predictions_df['Year'], predictions_df['Predicted_Salary_LPA'], marker='o', linestyle='-', color='dodgerblue', label='Predicted Salary (LPA)'); ax.set_xlabel("Year"); ax.set_ylabel("Predicted Salary (LPA)", color='dodgerblue'); ax.tick_params(axis='y', labelcolor='dodgerblue'); ax.set_title("Predicted Salary Trend & Growth Rate"); ax.grid(True, linestyle='--', alpha=0.6); ax2 = ax.twinx(); line2, = ax2.plot(predictions_df['Year'], predictions_df['Predicted_Growth_Rate (%)'], marker='x', linestyle=':', color='darkorange', label='YoY Growth (%)'); ax2.set_ylabel("Predicted YoY Growth (%)", color='darkorange'); ax2.tick_params(axis='y', labelcolor='darkorange'); lines = [line1, line2]; ax.legend(lines, [l.get_label() for l in lines], loc='upper left'); fig.tight_layout(); st.pyplot(fig)
                    # [Table display code remains the same]
                    st.subheader("Prediction Details"); st.dataframe(predictions_df[['Year', 'Years of Experience', 'Predicted_Salary_LPA', 'Predicted_Growth_Rate (%)']].style.format({'Predicted_Salary_LPA': '{:.1f} LPA', 'Predicted_Growth_Rate (%)': '{:.1f}%', 'Years of Experience': '{:.0f}'}))
                    # [Economic context display code remains the same]
                    st.subheader("Economic & Industry Context (Forecast)");
                    if not econ_context_df.empty: st.dataframe(econ_context_df.style.format("{:.2f}")); st.caption("Note: Economic/Industry indicators based on basic linear trend forecasts.")
                    else: st.caption("Could not retrieve economic context data.")
                else: st.error("Failed to generate predictions for the selected scenario.")
            else: st.info(" Select scenario details and click 'Predict Salary Trend'.")

        # --- <<< NEW SECTION: Display Static Plots >>> ---
        st.markdown("---") # Add a separator
        with st.expander("View Training Data Insights & Model Plots"):
            st.subheader("Historical Salary vs Experience")
            if os.path.exists(SALARY_EXP_PLOT):
                st.image(SALARY_EXP_PLOT, caption="Salary vs Experience for Top 10 Roles (Historical Data)")
            else:
                st.caption(f"Plot not found: {os.path.basename(SALARY_EXP_PLOT)}")

            st.subheader("Feature Correlation (Historical)")
            if os.path.exists(CORR_HEATMAP_PLOT):
                st.image(CORR_HEATMAP_PLOT, caption="Correlation Matrix of Key Features (Historical Data)")
            else:
                st.caption(f"Plot not found: {os.path.basename(CORR_HEATMAP_PLOT)}")

            st.subheader("Model Feature Importance")
            if os.path.exists(FEAT_IMPORTANCE_PLOT):
                st.image(FEAT_IMPORTANCE_PLOT, caption="Top 30 Feature Importances for the Selected Model")
            else:
                st.caption(f"Plot not found: {os.path.basename(FEAT_IMPORTANCE_PLOT)}")

            st.subheader("Historical Salary Trends")
            if os.path.exists(HIST_TREND_PLOT):
                st.image(HIST_TREND_PLOT, caption="Median Historical Salary Trend for Top Roles")
            else:
                st.caption(f"Plot not found: {os.path.basename(HIST_TREND_PLOT)}")
        # --- <<< END NEW SECTION >>> ---

    else:
        # Error loading message remains the same
        st.error("🔴 Application cannot start. Model or auxiliary data failed to load.")
    

    # Sidebar remains the same
    st.sidebar.title("Disclaimer & Limitations:"); st.sidebar.info("""

Simulated Data: The underlying salary data used to train the model is simulated and designed to reflect general trends, but it may not capture the full complexity or specific nuances of the real job market.

Basic Forecasts: Economic and industry forecasts used for future predictions are based on simplified methods (e.g., linear trends) and serve as placeholders. Actual economic conditions can be highly volatile and unpredictable.

Unmodeled Factors: This model does not account for crucial real-world factors such as individual performance reviews, specific niche skills beyond the 'Hot Skill' flag, company funding/profitability, stock options/equity, specific benefits packages, or individual negotiation outcomes.

Illustrative Use Only: The predictions generated are estimates for illustrative purposes. They should not be used as the sole basis for career decisions, salary negotiations, or financial planning. Users should consult multiple sources and consider their unique circumstances."""); st.sidebar.markdown("---")
    last_year_info = aux_data['last_hist_year'] if aux_data else 'N/A'; st.sidebar.write(f"Model trained using data up to year: {last_year_info}")


if __name__ == "__main__":
    run_app()