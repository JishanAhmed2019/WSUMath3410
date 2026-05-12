import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Function for the interactive PDF and CDF plots
def interactive_pdf_cdf(mu=0, sigma=1, x_point=0):
    x = np.linspace(norm.ppf(0.001, mu, sigma), norm.ppf(0.999, mu, sigma), 1000)
    pdf = norm.pdf(x, mu, sigma)
    cdf = norm.cdf(x, mu, sigma)

    pdf_val = norm.pdf(x_point, mu, sigma)
    cdf_val = norm.cdf(x_point, mu, sigma)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # PDF plot
    ax1.plot(x, pdf, 'b-', lw=2)
    ax1.fill_between(x, pdf, where=(x <= x_point), color="lime", alpha=0.4)
    ax1.axvline(x=x_point, color='green', linestyle='--')
    ax1.text(x_point, pdf_val, f'PDF({x_point:.2f}) = {pdf_val:.4f}', verticalalignment='bottom', horizontalalignment='right')
    ax1.set_title('Probability Density Function')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Probability Density')

    # CDF plot
    ax2.plot(x, cdf, 'r-', lw=2)
    ax2.axvline(x=x_point, color='darkorange', linestyle='--')
    ax2.text(x_point, cdf_val, f'CDF({x_point:.2f}) = {cdf_val:.4f}', verticalalignment='bottom', horizontalalignment='right')
    ax2.set_title('Cumulative Distribution Function')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Cumulative Probability')

    plt.tight_layout()
    return fig

# Streamlit app layout
st.markdown("<h1 style='text-align: center; color: blue;'>PDF and CDF Visualization</h1>", unsafe_allow_html=True)

# Sidebar sliders
mu = st.sidebar.slider('Mean (μ)', -3.0, 3.0, 0.0, 0.1)
sigma = st.sidebar.slider('Standard Deviation (σ)', 0.1, 3.0, 1.0, 0.1)
x_point = st.sidebar.slider('X Value', -3.0, 3.0, 0.0, 0.1)

# Display the plots
st.pyplot(interactive_pdf_cdf(mu, sigma, x_point))

# Footer (signature)
st.markdown("<h4 style='text-align: center; color: purple;'>Created by Dr. Jishan Ahmed</h4>", unsafe_allow_html=True)
