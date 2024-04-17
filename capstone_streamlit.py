# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Set page title and layout
st.set_page_config(page_title="Saksham Gupta - Customer Segmentation Analysis", layout="wide")

# Header
st.header("Customer Segmentation Analysis for Retail")

# Load data
@st.cache_resource()
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/sgx-saksham/Predictive-Analysis-streamlit/main/Capstone_data_cleaned.csv')  
    return df.copy()  

df = load_data()

# Label encode categorical columns
le = LabelEncoder()
cat_cols = ['dim_gender', 'dim_location', 'dim_preferred_payment_method', 'dim_product_category', 'dim_season', 'dim_preferred_product_category']
df[cat_cols] = df[cat_cols].apply(le.fit_transform)

# Standardize numerical columns
scaler = StandardScaler()
num_cols = ['dim_gender', 'dim_age', 'dim_preferred_product_category', 'dim_season', 'meas_store_visits_per_month', 'meas_total_spending', 'meas_annual_income', 'meas_discount_usage', 'meas_days_since_last_purchase', 'meas_loyalty_points', 'meas_average_basket_size', 'meas_satisfaction_score', 'meas_purchase_frequency']
df[num_cols] = scaler.fit_transform(df[num_cols])

# Initial data insights
st.write("### Initial Data Insights:")
st.write("Explore some initial insights from the dataset here, such as summary statistics or a few sample rows:")
st.write(df.describe())  # Display summary statistics


st.sidebar.title("Options")
segmentation_type = st.sidebar.selectbox('Select segmentation type', ['Demographic', 'Behavioral', 'Purchase History', 'Preferences', 'Seasonal', 'Engagement'])

if segmentation_type == 'Demographic':
    selected_features = ['dim_age', 'dim_gender', 'meas_total_spending']
elif segmentation_type == 'Behavioral':
    selected_features = ['meas_store_visits_per_month', 'meas_total_spending', 'meas_discount_usage', 'meas_purchase_frequency']
elif segmentation_type == 'Purchase History':
    selected_features = ['meas_store_visits_per_month', 'meas_days_since_last_purchase', 'meas_average_basket_size']
elif segmentation_type == 'Preferences':
    selected_features = ['dim_preferred_product_category', 'dim_preferred_payment_method', 'meas_loyalty_points', 'meas_satisfaction_score']
elif segmentation_type == 'Seasonal':
    selected_features = ['dim_season', 'meas_total_spending', 'meas_satisfaction_score']
elif segmentation_type == 'Engagement':
    selected_features = ['meas_discount_usage','meas_store_visits_per_month', 'meas_days_since_last_purchase']

# Display selected variables
st.sidebar.markdown(f"**Selected Features for {segmentation_type} Segmentation:**")
for feature in selected_features:
    st.sidebar.write("- "+feature)
# Button for training the dataset
if st.sidebar.button('Train the dataset'):
    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[selected_features])
    st.session_state['trained'] = True  # Set the 'trained' state to True

# Display the visualization using Plotly
if st.sidebar.button('View visualization'):
    if 'trained' in st.session_state and st.session_state['trained']: # Check if the dataset is trained
        # PCA
        pca = PCA(n_components=3)  # Use 3 components for 3D plot
        df[['PC1', 'PC2', 'PC3']] = pca.fit_transform(df[selected_features])

        # Assuming pca is already fitted

        # 3D Scatter plot with Plotly
        fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='Cluster', title="Customer Segments",
                            labels={'PC1': 'Component_1', 'PC2': 'Component_2', 'PC3': 'Component_3'},
                            opacity=0.8, size_max=10, color_continuous_scale='viridis')
        st.plotly_chart(fig)

        # Calculate average feature values for each cluster
        cluster_means = df.groupby('Cluster')[selected_features].mean()

        # Display the average feature values for each cluster
        st.write("Segment Profiling: Understanding Customer Behavior through **Cluster Analysis**:")
        st.write(cluster_means)

        cluster_counts = df['Cluster'].value_counts()
        cluster_counts_df = pd.DataFrame({'Cluster': cluster_counts.index, 'Count': cluster_counts.values})
        cluster_counts_df = cluster_counts_df.sort_values(by='Cluster')

        # Create bar chart using Plotly Express with custom colors and data labels
        fig = px.bar(cluster_counts_df, x='Cluster', y='Count', title='Number of Data Points in Each Cluster',
                    labels={'Cluster': 'Cluster Label', 'Count': 'Number of Data Points'},
                    color='Cluster', color_continuous_scale='viridis', text='Count')

        fig.update_traces(textposition='outside', textfont=dict(color='black', size=12))
        st.plotly_chart(fig)
        
        # loadings
        loadings = pca.components_

        # Create a DataFrame to display the loadings
        loadings_df = pd.DataFrame(loadings, columns=df[selected_features].columns, index=['PC1', 'PC2', 'PC3'])

        # Display the loadings
        st.write("Unveiling Data Patterns: Exploring Key Drivers through **Loadings Analysis**:")
        st.write(loadings_df)

        # Calculate the overall effect of each feature
        overall_effect = loadings_df.abs().sum()

        # Find the feature with the highest overall effect
        most_effective_feature = overall_effect.idxmax()
        highest_effect_value = overall_effect.max()

        # Display the result
        st.write("Overall Effect of Features:")
        st.write(overall_effect)
        st.write(f"The feature with the highest overall effect is **{most_effective_feature}** with a total effect of **{highest_effect_value}**.")
        # Inference
        st.markdown("**Inference:**")
        st.markdown("The 3D visualization displays customer segments based on the selected features for clustering.")
        st.markdown("Each cluster represents a group of customers with similar characteristics.")
        st.markdown("This information can be used to tailor marketing strategies, improve sales, and understand customer behavior.")
        st.markdown("To get to know about each cluster and component [*click here*](https://customer-segmentation-team-1.netlify.app/clusters)")
        st.markdown("To get every detail about each cluster and component and its analysis [*click here*](https://colab.research.google.com/drive/1HESEctJ4dT_Gi7o6f0k2kdMI8z99L57J?usp=sharing)")
    else:
        st.sidebar.error('Please train the dataset first.')

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Team 1 ")
st.sidebar.markdown("- Saksham Gupta")
st.sidebar.markdown("- Prasanna Venkadesh")
st.sidebar.markdown("- Abirami B")
