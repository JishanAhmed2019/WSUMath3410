import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import ks_2samp

# Function to load data
def load_data(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        return df
    return None

st.set_page_config(page_title="Dataset Distribution Comparison", layout="wide")
st.title('Dataset Distribution Comparison App')
st.markdown("This app compares the distributions of two datasets using the Kolmogorov-Smirnov test.")

# User input for datasets
uploaded_file_1 = st.sidebar.file_uploader("Choose Dataset 1 (CSV or Excel)", type=['csv', 'xlsx'], key='file1')
uploaded_file_2 = st.sidebar.file_uploader("Choose Dataset 2 (CSV or Excel)", type=['csv', 'xlsx'], key='file2')

# Load the datasets
dataset1 = load_data(uploaded_file_1)
dataset2 = load_data(uploaded_file_2)

# Select columns to compare
if dataset1 is not None and dataset2 is not None:
    column1 = st.sidebar.selectbox('Select the column from Dataset 1 to compare', dataset1.columns, key='col1')
    column2 = st.sidebar.selectbox('Select the column from Dataset 2 to compare', dataset2.columns, key='col2')
    
    # Plotting the distribution of the selected columns
    fig1 = px.histogram(dataset1, x=column1, nbins=50, color_discrete_sequence=['blue'], title=f'Distribution of {column1} in Dataset 1')
    fig2 = px.histogram(dataset2, x=column2, nbins=50, color_discrete_sequence=['magenta'], title=f'Distribution of {column2} in Dataset 2')
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
    
    # Compare distributions for the selected column
    if st.sidebar.button('Perform Kolmogorov-Smirnov Test'):
        ks_statistic, ks_pvalue = ks_2samp(dataset1[column1], dataset2[column2])
        
        st.write(f"Kolmogorov-Smirnov Test Results for {column1} vs. {column2}:")
        st.write(f"Statistic: {ks_statistic}, P-value: {ks_pvalue}")
        
        if ks_pvalue < 0.05:
            st.success(f"The distributions of {column1} and {column2} are significantly different (p < 0.05).")
        else:
            st.info(f"No significant difference found in the distributions of {column1} and {column2} (p >= 0.05).")

# Signature
st.sidebar.markdown("---")
st.sidebar.markdown("Created by Dr. Jishan Ahmed")

# In case no file is uploaded
if not uploaded_file_1 or not uploaded_file_2:
    st.markdown("Please upload datasets to compare their distributions.")
