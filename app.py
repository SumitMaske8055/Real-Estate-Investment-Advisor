import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
# You would use a library like shap for real SHAP values
# import shap 

# Set the title and overall app configuration
st.set_page_config(layout="wide", page_title="🏠 Property Investment Advisor with AI Explainability")

# --- Model and Feature Loading (User Must Customize) ---
# ==============================================================================
# 🛑 ACTION REQUIRED: REPLACE PLACEHOLDERS WITH YOUR ACTUAL MODEL AND FEATURES
# ==============================================================================

# 1. MOCK MODEL CLASS: Replace this entire block with your model loading logic.
class MockClassifier:
    """Simulates a trained model."""
    def predict(self, X):
        # We will randomly assign profitability for the demo
        return np.random.choice([0, 1], size=len(X))
    
    # NEW: Mock function to simulate SHAP prediction (important for realistic insights)
    def predict_proba(self, X):
        # Returns [prob_low_profit, prob_high_profit]
        # We check a column that is always present after feature engineering
        if X['age_of_property'].iloc[0] > 10 and np.random.rand() > 0.4:
            # Properties older than 10 years might be lower profit (mock logic)
            return np.array([[0.7, 0.3]]) 
        else:
            return np.array([[0.2, 0.8]])

try:
    # Attempt to load the real model if available, otherwise use the mock
    clf_model = MockClassifier() 
except: 
    st.error("Model file not found. Using Mock Classifier for demonstration.")
    clf_model = MockClassifier()

# 2. TRAINING FEATURE NAMES: Must match the features your model was trained on.
MOCK_TRAINING_FEATURE_NAMES = [
    'bhk', 'size_in_sqft', 'year_built', 'floor_no', 'total_floors',
    'age_of_property', 'nearby_schools', 'nearby_hospitals', 
    'price_per_sqft', 
    'city_chennai', 'city_jaipur', 'city_new_delhi', 'property_type_apartment', 
    'furnished_status_furnished', 'furnished_status_semi-furnished', 
    'owner_type_broker', 'availability_status_ready_to_move'
]
TRAINING_FEATURE_NAMES = MOCK_TRAINING_FEATURE_NAMES

# ==============================================================================
# -------------------- END OF ACTION REQUIRED SECTION --------------------------
# ==============================================================================

# --- Data Loading and Preprocessing (Unchanged) ---
@st.cache_data
def load_data():
    """Loads and preprocesses the property data."""
    # Assuming 'india_housing_prices.csv' is in the same directory
    df = pd.read_csv('india_housing_prices.csv')
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    df['availability_status'] = df['availability_status'].replace({
        'Under_Construction': 'Under Construction', 'Ready_to_Move': 'Ready to Move'
    })
    
    # Adding mock coordinates for map visualization
    np.random.seed(42) 
    city_coords = {
        'Chennai': (13.0827, 80.2707), 'Warangal': (17.9689, 79.5941),
        'New Delhi': (28.6139, 77.2090), 'Jaipur': (26.9124, 75.7873),
        'Bangalore': (12.9716, 77.5946), 'Kolkata': (22.5726, 88.3639),
        'Hyderabad': (17.3850, 78.4867), 'Mumbai': (19.0760, 72.8777)
    }
    df['latitude'] = df['city'].apply(lambda x: np.random.normal(city_coords.get(x, (20, 77))[0], 0.5))
    df['longitude'] = df['city'].apply(lambda x: np.random.normal(city_coords.get(x, (20, 77))[1], 0.5))
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("🚨 **FILE NOT FOUND ERROR** 🚨: Please make sure 'india_housing_prices.csv' is in the same directory as 'app.py'.")
    st.stop()


# --- Sidebar Filters (Unchanged) ---
st.sidebar.title("⚙️ Market Filters")
selected_city = st.sidebar.selectbox("Dashboard City Filter", ["All"] + sorted(df['city'].unique().tolist()))
if selected_city != "All":
    df_filtered = df[df['city'] == selected_city]
else:
    df_filtered = df

min_price, max_price = st.sidebar.slider(
    "Price Range (in Lakhs)",
    float(df_filtered['price_in_lakhs'].min()), 
    float(df_filtered['price_in_lakhs'].max()), 
    (float(df_filtered['price_in_lakhs'].min()), float(df_filtered['price_in_lakhs'].max()))
)
df_filtered = df_filtered[
    (df_filtered['price_in_lakhs'] >= min_price) & (df_filtered['price_in_lakhs'] <= max_price)
]
selected_bhk = st.sidebar.multiselect(
    "BHK Filter", sorted(df['bhk'].unique().tolist()), default=sorted(df['bhk'].unique().tolist())
)
df_filtered = df_filtered[df_filtered['bhk'].isin(selected_bhk)]
st.sidebar.markdown("---")
st.sidebar.info(f"Showing **{len(df_filtered)}** properties out of **{len(df)}** total properties.")


# --- PREDICTION INPUT SECTION ---
st.sidebar.title("🤖 Prediction Input")
st.sidebar.markdown("---")
st.sidebar.subheader("Property Features for Prediction")

# Input widgets for features required by the prediction model
pred_city = st.sidebar.selectbox("City", sorted(df['city'].unique().tolist()), key='pred_city')
pred_locality = st.sidebar.selectbox("Locality (Simulated)", sorted(df['locality'].unique().tolist()), key='pred_locality')
pred_bhk = st.sidebar.selectbox("BHK", sorted(df['bhk'].unique().tolist()), key='pred_bhk')
pred_size = st.sidebar.number_input("Size (SqFt)", min_value=100, max_value=20000, value=1200, step=100, key='pred_size')

# Inputs required by the MOCK_TRAINING_FEATURE_NAMES list
pred_price = st.sidebar.number_input("Expected Selling Price (Lakhs)", min_value=1.0, value=150.0, step=10.0, key='pred_price')
pred_year = st.sidebar.number_input("Year Built", min_value=1950, max_value=2025, value=2015, step=1, key='pred_year')
pred_floor = st.sidebar.number_input("Floor Number", min_value=1, max_value=50, value=5, step=1, key='pred_floor')
pred_total_floors = st.sidebar.number_input("Total Floors in Building", min_value=1, max_value=50, value=15, step=1, key='pred_total_floors')
pred_schools = st.sidebar.number_input("Nearby Schools Score (1-10)", min_value=1, max_value=10, value=7, step=1, key='pred_schools')
pred_hospitals = st.sidebar.number_input("Nearby Hospitals Score (1-10)", min_value=1, max_value=10, value=5, step=1, key='pred_hospitals')

pred_status = st.sidebar.selectbox("Furnished Status", sorted(df['furnished_status'].unique().tolist()), key='pred_status')
pred_owner = st.sidebar.selectbox("Owner Type", sorted(df['owner_type'].unique().tolist()), key='pred_owner')
pred_avail = st.sidebar.selectbox("Availability Status", sorted(df['availability_status'].unique().tolist()), key='pred_avail')

predict_button = st.sidebar.button("Predict Profitability", use_container_width=True)


# --- Main Dashboard Layout ---
st.title("🏡 Indian Property Market Analysis")

# Check if the filtered dataframe is empty
if df_filtered.empty:
    st.error("No properties match the selected filters. Please adjust your criteria.")
else:
    # --- 1. Key Metrics (KPIs) ---
    col1, col2, col3, col4 = st.columns(4)

    total_properties = len(df_filtered)
    avg_price = df_filtered['price_in_lakhs'].mean()
    avg_size = df_filtered['size_in_sqft'].mean()
    avg_price_per_sqft = df_filtered['price_per_sqft'].mean()
    
    col1.metric("Total Properties", f"{total_properties:,}")
    col2.metric("Avg. Price", f"₹{avg_price:,.2f} Lakhs")
    col3.metric("Avg. Size", f"{avg_size:,.0f} SqFt")
    col4.metric("Avg. Price/SqFt", f"₹{avg_price_per_sqft:.2f}")

    st.markdown("---")

    # --- 2. Charts (Two columns) ---
    col5, col6 = st.columns([1, 1.5])

    # 2a. Property Type Distribution (Pie Chart)
    with col5:
        st.subheader("Property Type Distribution")
        type_counts = df_filtered['property_type'].value_counts().reset_index()
        type_counts.columns = ['Property_Type', 'Count']
        
        fig_type = px.pie(
            type_counts, 
            values='Count', 
            names='Property_Type', 
            title='Percentage of Available Property Types',
            hole=0.4
        )
        st.plotly_chart(fig_type, use_container_width=True)

    # 2b. Price Distribution (Histogram)
    with col6:
        st.subheader("Price Distribution (Price in Lakhs)")
        
        max_price_filtered = df_filtered['price_in_lakhs'].max()
        bins = np.arange(0, max_price_filtered + 50, 50) 
        
        fig_price = px.histogram(
            df_filtered, 
            x='price_in_lakhs', 
            nbins=len(bins) - 1,
            color='availability_status', 
            title='Price Distribution by Availability Status',
            height=450
        )
        fig_price.update_layout(xaxis_title="Price in Lakhs", yaxis_title="Number of Properties")
        st.plotly_chart(fig_price, use_container_width=True)

    st.markdown("---")

    # --- 3. Property Geography and Valuation Analysis ---
    col7, col8 = st.columns([1.5, 1])

    # --- Column 1: Property Locations Map ---
    with col7:
        st.subheader("Property Locations & Price Heat Map")
        map_data = df_filtered[['latitude', 'longitude', 'city', 'price_in_lakhs', 'property_type']].dropna()
        
        if not map_data.empty:
            fig_map = px.scatter_mapbox(
                map_data,
                lat="latitude",
                lon="longitude",
                hover_name="city",
                hover_data=["property_type", "price_in_lakhs"],
                color="price_in_lakhs",
                color_continuous_scale=px.colors.sequential.Plasma,
                zoom=4,
                height=500,
                title="Geographical Distribution of Properties (Colored by Price)"
            )
            fig_map.update_layout(mapbox_style="carto-positron")
            fig_map.update_layout(margin={"r":0,"t":25,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Map data not available for the current selection. Adjust filters.")

    # --- Column 2: Top Valuation Localities ---
    with col8:
        st.subheader("Top Localities by Average Price")
        locality_price = df_filtered.groupby('locality')['price_in_lakhs'].agg(['mean', 'count']).reset_index()
        locality_price.columns = ['Locality', 'Avg_Price_Lakhs', 'Count']
        locality_price = locality_price.sort_values(by='Avg_Price_Lakhs', ascending=False)
        
        if not locality_price.empty:
            highest_avg_locality_name = locality_price.iloc[0]['Locality']
            highest_avg_locality_price = locality_price.iloc[0]['Avg_Price_Lakhs']
            st.metric(
                label="Most Expensive Locality",
                value=f"₹{highest_avg_locality_price:,.2f} Lakhs",
                delta=f"{highest_avg_locality_name}",
                delta_color="off"
            )
            
        locality_price_top10 = locality_price.head(10)

        fig_bar = px.bar(
            locality_price_top10, 
            x='Avg_Price_Lakhs', 
            y='Locality', 
            orientation='h',
            color='Count',
            color_continuous_scale=px.colors.sequential.Viridis,
            title='Top 10 Localities (Ranked by Avg. Price)',
            height=400
        )
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}) 
        fig_bar.update_xaxes(title_text="Average Price (Lakhs)")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("---")

    # --- 4. Tabular Data and Market Insights ---
    st.subheader("Detailed Property Data")
    st.dataframe(df_filtered.head(100))

    st.markdown("---")
    
    ## 💡 Key Market Insights
    st.header("💡 Key Market Insights")

    most_common_type = df_filtered['property_type'].mode()[0]
    st.info(f"The most common property type in the current selection is **{most_common_type}**.")

    if selected_city == "All" and not df_filtered.empty:
        highest_avg_price_city = df_filtered.groupby('city')['price_in_lakhs'].mean().idxmax()
        highest_avg_price_value = df_filtered.groupby('city')['price_in_lakhs'].mean().max()
        st.success(f"**{highest_avg_price_city}** has the highest average property price at **₹{highest_avg_price_value:,.2f} Lakhs**, suggesting it is a premium market.")
    elif not df_filtered.empty:
        highest_avg_locality = locality_price.iloc[0]['Locality']
        highest_avg_price_value_loc = locality_price.iloc[0]['Avg_Price_Lakhs']
        st.success(f"Within **{selected_city}**, the locality **{highest_avg_locality}** shows the highest average price at **₹{highest_avg_price_value_loc:,.2f} Lakhs**.")

    furnishing_status = df_filtered['furnished_status'].value_counts(normalize=True).mul(100).round(1)
    if 'Unfurnished' in furnishing_status.index:
         st.warning(f"Approximately **{furnishing_status['Unfurnished']}%** of the properties are **Unfurnished**. This may indicate a market preference for buyers/renters to customize their homes or a high supply of new/under-construction projects.")

    st.markdown("---")

    # --- 5. Investment Prediction Section ---
    st.header("🔮 Investment Profitability Prediction")
    
    if predict_button:
        try:
            # 1. Prepare the input data for the model
            current_year = pd.Timestamp('now').year
            input_data = pd.DataFrame({
                # RAW INPUT FEATURES
                'city': [pred_city], 'locality': [pred_locality], 'bhk': [pred_bhk], 
                'size_in_sqft': [pred_size], 'year_built': [pred_year],
                'floor_no': [pred_floor], 'total_floors': [pred_total_floors],
                'nearby_schools': [pred_schools], 'nearby_hospitals': [pred_hospitals],
                'furnished_status': [pred_status], 'owner_type': [pred_owner],
                'availability_status': [pred_avail],
                # CALCULATED FEATURES (REQUIRED BY MOCK MODEL)
                'price_per_sqft': [pred_price * 100000 / pred_size],
            })

            # Calculate derived Age of Property
            input_data['age_of_property'] = current_year - input_data['year_built']

            # Prepare categorical columns for OHE (matching preprocessing in training)
            for col in ['city', 'furnished_status', 'owner_type', 'availability_status']:
                input_data[col] = input_data[col].str.lower().str.replace(' ', '_').str.replace('-', '_')
            
            # 2. Replicate all preprocessing (One-Hot Encoding)
            categorical_cols = input_data.select_dtypes(include=['object']).columns.tolist()
            X_processed = pd.get_dummies(input_data, columns=categorical_cols, prefix=categorical_cols)

            # 3. CRITICAL STEP: Align Features
            input_row = X_processed.reindex(columns=TRAINING_FEATURE_NAMES, fill_value=0)

            # 4. Prediction
            pred_prob = clf_model.predict_proba(input_row)[0]
            pred_class = np.argmax(pred_prob) # 1 for high profit, 0 for low profit
            
            # 5. Display Prediction
            if pred_class == 1:
                st.success(f"✅ **High Profitability Predicted** with {pred_prob[1]*100:.1f}% confidence for {pred_bhk} BHK in {pred_city}")
                st.markdown("This property shows characteristics that historically yielded **strong returns**.")
                st.balloons()
            else:
                st.warning(f"⚠️ **Lower Profitability Predicted** with {pred_prob[0]*100:.1f}% confidence.")
                st.markdown("This property aligns with **average or lower-performing** assets. Review the detailed explanation below.")
            
            # --- 6. SHAP Explainability (NEW SECTION - FIXED) ---
            st.markdown("---")
            st.header("🧠 Model Explainability (SHAP Insights)")
            
            # --- CORRECTION: Safer Mock SHAP Calculation ---
            
            # 1. Isolate the numerical/derived features we want to analyze (the first 9 features)
            numerical_features_for_shap = TRAINING_FEATURE_NAMES[:9] 
            
            # Get the input values for these features as a Series
            # Using .T.iloc[:, 0] accesses the single row of input_row
            input_values_series = input_row[numerical_features_for_shap].T.iloc[:, 0]
            
            # 2. Calculate MOCK SHAP values based on input values
            np.random.seed(42)
            mock_contributions = input_values_series.apply(
                lambda x: np.random.uniform(-0.5, 0.5) * x if x != 0 else np.random.uniform(-0.1, 0.1)
            ).fillna(0)
            
            # 3. Apply the strong negative influence to price_per_sqft for realism
            if 'price_per_sqft' in mock_contributions.index:
                mock_contributions['price_per_sqft'] = mock_contributions['price_per_sqft'] * -2
            
            # 4. Create the final DataFrame for plotting
            shap_data = pd.DataFrame({
                'Feature': mock_contributions.index.str.replace('_', ' ').str.title(),
                'Contribution': mock_contributions.values,
                'Value': input_values_series.values
            })
            shap_data['Value'] = shap_data['Value'].round(2)
            
            # --- End of CORRECTION ---

            # Filter and sort to show only the top 8 most impactful features
            shap_data = shap_data.sort_values(by='Contribution', ascending=False).head(8)
            
            # Map SHAP contribution to a more user-friendly format
            shap_data['Impact'] = shap_data['Contribution'].apply(lambda x: 'Positive (Increases Profit)' if x > 0.05 else ('Negative (Decreases Profit)' if x < -0.05 else 'Neutral'))
            
            st.markdown("#### Top Factors Driving This Prediction:")
            
            # Plot the SHAP-style waterfall/bar chart
            fig_shap = px.bar(
                shap_data.sort_values(by='Contribution', ascending=True),
                x='Contribution',
                y='Feature',
                color='Impact',
                color_discrete_map={
                    'Positive (Increases Profit)': 'green', 
                    'Negative (Decreases Profit)': 'red', 
                    'Neutral': 'lightgray'
                },
                orientation='h',
                title="Feature Contribution to Profitability Score"
            )
            fig_shap.update_layout(xaxis_title="SHAP Contribution (Model Log-Odds)", yaxis_title="")
            st.plotly_chart(fig_shap, use_container_width=True)
            
            # Realistic Interpretation
            st.markdown("#### 🗣️ Realistic Investment Insights")
            
            col_pos, col_neg = st.columns(2)
            
            with col_pos:
                positive_factors = shap_data[shap_data['Impact'] == 'Positive (Increases Profit)']
                if not positive_factors.empty:
                    st.success("**Strongest Positive Drivers:**")
                    for index, row in positive_factors.head(3).iterrows():
                        st.markdown(f"* **{row['Feature']} ({row['Value']}):** This feature is a significant strength, pushing the profitability prediction higher. It indicates strong market demand for properties with this specific attribute (e.g., high **Nearby Schools** score or ideal **BHK** size).")
                else:
                    st.info("No strong positive drivers identified for this property configuration.")
            
            with col_neg:
                negative_factors = shap_data[shap_data['Impact'] == 'Negative (Decreases Profit)']
                if not negative_factors.empty:
                    st.error("**Strongest Negative Drivers (Investment Risk):**")
                    for index, row in negative_factors.head(3).iterrows():
                        st.markdown(f"* **{row['Feature']} ({row['Value']}):** This feature poses a risk. It is significantly dragging down the potential for high profitability. For example, a high **Price Per Sqft** often negatively correlates with future profit margins.")
                else:
                    st.info("No strong negative drivers identified for this property configuration.")
            
        except Exception as e:
            st.error(f"Prediction or SHAP Error: {e}")
            st.caption("Review the **Model Loading Section** and ensure the provided features align exactly with your trained model's requirements.")