import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, laplace, norm, gumbel_r, kurtosis
import streamlit as st

# Use a basic and widely available style
plt.style.use('classic')  # Ensures compatibility

# Set the number of samples
num_samples = 10000

# Function to plot the sum, mean, or max of uniform or Laplace random variables
def plot_statistic(num_vars, distribution, statistic):
    if distribution == 'Uniform':
        data = uniform.rvs(size=(num_samples, num_vars))
    else:  # Laplace
        data = laplace.rvs(size=(num_samples, num_vars))
    
    if statistic == 'Sum':
        stat_data = np.sum(data, axis=1)
    elif statistic == 'Mean':
        stat_data = np.mean(data, axis=1)
    elif statistic == 'Max':
        stat_data = np.max(data, axis=1)
    
    # Calculate statistics
    data_mean = np.mean(stat_data)
    data_var = np.var(stat_data)
    data_kurtosis = kurtosis(stat_data)

    plt.figure(figsize=(10, 6))
    count, bins, ignored = plt.hist(stat_data, bins=30, density=True, alpha=0.6, color='orangered', edgecolor='black')
    
    if statistic != 'Max':
        mu, std = norm.fit(stat_data)
        x = np.linspace(mu - 4*std, mu + 4*std, 100)
        p = norm.pdf(x, mu, std)
    else:
        param = gumbel_r.fit(stat_data)
        x = np.linspace(min(stat_data), max(stat_data), 100)
        p = gumbel_r.pdf(x, *param)
    plt.plot(x, p, 'blue', linewidth=2)
    
    # Annotate statistics on the plot
    plt.annotate(f'Mean: {data_mean:.2f}\nVariance: {data_var:.2f}\nKurtosis: {data_kurtosis:.2f}',
                 xy=(0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round,pad=0.5", fc="greenyellow", ec="black", lw=2), fontsize=10)

    plt.title(f'{statistic} of {num_vars} {distribution} Variables')
    plt.xlabel(f'{statistic} Value')
    plt.ylabel('Density')
    st.pyplot(plt)  # Use st.pyplot() to render matplotlib plot

# Streamlit UI components
st.image("math_horiz.png", use_column_width=True)
st.title("Central Limit Theorem Visualization")
num_vars = st.sidebar.number_input("Number of Variables:", min_value=1, value=1, step=1)
distribution = st.sidebar.selectbox("Distribution:", ['Uniform', 'Laplace'])
statistic = st.sidebar.radio("Statistic:", ['Sum', 'Mean', 'Max'])

# Trigger re-plotting when the user changes any input
plot_statistic(num_vars, distribution, statistic)

st.markdown('---')
st.markdown('*Created by Dr. Jishan Ahmed*')
