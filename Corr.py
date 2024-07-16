import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import cm
from scipy.interpolate import make_interp_spline

# Set up the Streamlit app layout and widgets
st.title('Correlation Coefficient Visualization')
st.markdown('Adapted from Michael Pyrcz, Professor, The University of Texas at Austin')

# Signature
st.markdown("*Created by Dr. Jishan Ahmed*")

# Add sliders and checkboxes for user input
ndata = st.slider('Number of Samples', min_value=0, max_value=10000, value=5000, step=1000)
corr = st.slider('Correlation Coefficient (ρ)', min_value=-1.0, max_value=1.0, value=0.0, step=0.1)
cond = st.checkbox('Show Conditionals')
raster = st.checkbox('Show Joint')

def add_grid(sub_plot):
    sub_plot.grid(True, which='major', linewidth=1.0)
    sub_plot.grid(True, which='minor', linewidth=0.2)
    sub_plot.tick_params(which='major', length=7)
    sub_plot.tick_params(which='minor', length=4)
    sub_plot.xaxis.set_minor_locator(AutoMinorLocator())
    sub_plot.yaxis.set_minor_locator(AutoMinorLocator())

def f_make(ndata, corr, cond, raster):
    cmap = cm.inferno
    np.random.seed(seed=73072)
    mean = np.array([0, 0])
    correl = np.array([[1.0, corr], [corr, 1.0]], dtype=float)
    sample = np.random.multivariate_normal(mean, correl, size=ndata)
    df = pd.DataFrame(sample, columns=['X1', 'X2'])
    
    fig, axs = plt.subplots(3, 3, figsize=(12, 12), gridspec_kw={'width_ratios': [3, 0.1, 1], 'height_ratios': [1, 0.1, 3], 'wspace': 0.1, 'hspace': 0.1})
    plt_scatter = axs[2, 0]
    plt_x1 = axs[0, 0]
    plt_x2 = axs[2, 2]
    axs[0, 1].axis('off')
    axs[1, 1].axis('off')
    axs[1, 0].axis('off')
    axs[0, 2].axis('off')
    axs[1, 2].axis('off')
    
    if raster:
        plt_scatter.hist2d(df['X1'], df['X2'], bins=30, range=[[-3, 3], [-3, 3]], density=False, cmap=plt.cm.Reds, alpha=1.0, zorder=1)
    else:
        plt_scatter.scatter(sample[:, 0], sample[:, 1], color='red', alpha=0.2, edgecolors='black', label='Samples', zorder=100)
        plt_scatter.legend(loc='upper left')
    
    plt_scatter.set_xlabel(r'$x_1$')
    plt_scatter.set_ylabel(r'$x_2$')
    plt_scatter.set_xlim([-3.0, 3.0])
    plt_scatter.set_ylim([-3.0, 3.0])
    add_grid(plt_scatter)
    
    for x in np.linspace(-3.0, 3.0, 31):
        plt_scatter.plot([x, x], [-3, 3], color='grey', lw=0.2)
        plt_scatter.plot([-3, 3], [x, x], color='grey', lw=0.2)
    
    for x in np.linspace(-2.0, 2.0, 5):
        plt_scatter.plot([x, x], [-3, 3], color='grey', lw=0.5)
        plt_scatter.plot([-3, 3], [x, x], color='grey', lw=0.5)
    
    if cond:
        nbins = 6
        X1_new = np.linspace(-2.0, 2.0, 300)
        X1_bins = np.linspace(-2.5, 2.5, nbins)
        X1_centroids = np.linspace((X1_bins[0]+X1_bins[1])*0.5, (X1_bins[-2]+X1_bins[-1])*0.5, nbins-1)
        df['X1_bins'] = pd.cut(df['X1'], X1_bins, labels=X1_centroids)
        
        cond_exp = df.groupby('X1_bins')['X2'].mean()
        cond_P90 = df.groupby('X1_bins')['X2'].quantile(.9)
        cond_P10 = df.groupby('X1_bins')['X2'].quantile(.1)
        
        spl_exp = make_interp_spline(X1_centroids, cond_exp, k=3)
        spl_P90 = make_interp_spline(X1_centroids, cond_P90, k=3)
        spl_P10 = make_interp_spline(X1_centroids, cond_P10, k=3)
        cond_exp_spl = spl_exp(X1_new)
        cond_P90_spl = spl_P90(X1_new)
        cond_P10_spl = spl_P10(X1_new)
        
        plt_scatter.plot(X1_new, cond_exp_spl, color='white', lw=4, zorder=100)
        plt_scatter.plot(X1_new, cond_exp_spl, color='black', lw=2, zorder=200)
        plt_scatter.plot(X1_new, cond_P90_spl, color='white', lw=4, zorder=100)
        plt_scatter.plot(X1_new, cond_P90_spl, 'r--', color='black', lw=2, zorder=200)
        plt_scatter.plot(X1_new, cond_P10_spl, color='white', lw=4, zorder=100)
        plt_scatter.plot(X1_new, cond_P10_spl, 'r--', color='black', lw=2, zorder=200)
        
        plt_scatter.annotate('Exp', [X1_new[0]-0.3, cond_exp_spl[0]])
        plt_scatter.annotate('Exp', [X1_new[-1]+0.05, cond_exp_spl[-1]])
        plt_scatter.annotate('P10', [X1_new[0]-0.3, cond_P10_spl[0]])
        plt_scatter.annotate('P10', [X1_new[-1]+0.05, cond_P10_spl[-1]])
        plt_scatter.annotate('P90', [X1_new[0]-0.3, cond_P90_spl[0]])
        plt_scatter.annotate('P90', [X1_new[-1]+0.05, cond_P90_spl[-1]])
    
    counts = plt_x1.hist(sample[:, 0], density=True, alpha=0.0, bins=np.linspace(-3.0, 3.0, 31))[0]
    N, bins, patches = plt_x1.hist(sample[:, 0], density=True, alpha=1.0, edgecolor='black', bins=np.linspace(-3.0, 3.0, 31))
    for i in range(0, 30):
        patches[i].set_facecolor(plt.cm.Reds(counts[i]-np.min(counts)/(np.max(counts)-np.min(counts))))
    
    plt_x1.set_ylim([0.0, 0.8])
    add_grid(plt_x1)
    plt_x1.set_xlabel(r'$x_1$')
    plt_x1.set_ylabel(r'Density')
    plt_x1.set_title(r'Bivariate Standard Gaussian Distributed Data with $\rho =$' + str(np.round(corr, 2)) + '.')
    
    counts = plt_x2.hist(sample[:, 1], density=True, alpha=0.0, bins=np.linspace(-3.0, 3.0, 31))[0]
    N, bins, patches = plt_x2.hist(sample[:, 1], orientation='horizontal', density=True, alpha=1.0, edgecolor='black', bins=np.linspace(-3.0, 3.0, 31))
    for i in range(0, 30):
        patches[i].set_facecolor(plt.cm.Reds(counts[i]-np.min(counts)/(np.max(counts)-np.min(counts))))
    
    plt_x2.set_xlim([0.0, 0.8])
    add_grid(plt_x2)
    plt_x2.set_ylabel(r'$x_2$')
    plt_x2.set_xlabel(r'Density')
    plt_scatter.set_ylabel(r'$x_2$')
    
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.1)
    st.pyplot(fig)

# Call the function to make the samples and plot
f_make(ndata, corr, cond, raster)
