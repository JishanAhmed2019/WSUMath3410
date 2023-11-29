import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

# Function to perform T-test and plot
def perform_t_test(sample_mean, sample_size, alpha):
    population_mean = 100  # Hypothetical population mean
    population_std = 15    # Hypothetical population standard deviation
    
    standard_error = population_std / np.sqrt(sample_size)
    t_statistic = (sample_mean - population_mean) / standard_error
    df = sample_size - 1
    p_value = t.sf(np.abs(t_statistic), df) * 2  # two-tailed test

    fig, ax = plt.subplots()
    x = np.linspace(t.ppf(0.001, df), t.ppf(0.999, df), 100)
    y = t.pdf(x, df)
    ax.plot(x, y, label='T-distribution')
    
    crit_t = t.ppf(1 - alpha/2, df)
    ax.fill_between(x, y, where=(x >= crit_t) | (x <= -crit_t), color='red', alpha=0.5, label='Rejection region')
    ax.axvline(t_statistic, color='green', linestyle='dashed', label=f'T-statistic = {t_statistic:.2f}')
    
    # Annotating null and alternative hypothesis
    ax.text(0, max(y)/1.2, f'Null Hypothesis $H_0$: $\mu = {population_mean}$', ha='center', va='center', fontsize=10)
    ax.text(t_statistic, max(y)/1.1, f'Alternative Hypothesis $H_a$: $\mu \\neq {population_mean}$', ha='center', va='center', fontsize=10, color='blue')
    
    ax.legend()
    
    return fig, t_statistic, p_value

# Streamlit app layout
st.title('T-Test Demonstration')

sample_mean = st.sidebar.slider('Sample Mean', 80.0, 120.0, 100.0)
sample_size = st.sidebar.slider('Sample Size', 5, 100, 30)
alpha = st.sidebar.slider('Significance Level (Alpha)', 0.01, 0.10, 0.05)

if st.sidebar.button('Perform T-Test'):
    fig, t_statistic, p_value = perform_t_test(sample_mean, sample_size, alpha)
    st.pyplot(fig)
    
    st.write(f'T-statistic: {t_statistic:.2f}')
    st.write(f'P-value: {p_value:.4f}')
    if p_value < alpha:
        st.error("Reject the null hypothesis.")
    else:
        st.success("Fail to reject the null hypothesis.")
