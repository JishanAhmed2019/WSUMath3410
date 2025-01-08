import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Set page title and icon
st.set_page_config(page_title="Coin Flip Simulation", page_icon="🎲")

# Title and description
st.title("🎲 Coin Flip Probability Simulation")
st.markdown("""
This app simulates flipping a fair coin multiple times and visualizes how the probability of **Heads** or **Tails** converges to **0.5** over time.
""")

# Sidebar for user input
st.sidebar.header("Settings")
num_flips = st.sidebar.slider("Number of Coin Flips", min_value=10, max_value=10000, value=1000, step=10)
seed = st.sidebar.number_input("Random Seed", value=42, help="Set a seed for reproducibility.")

# Simulate coin flips
np.random.seed(seed)
flips = np.random.randint(0, 2, num_flips)  # 1 = Heads, 0 = Tails

# Cumulative counts of heads and tails
cumulative_heads = np.cumsum(flips)
cumulative_tails = np.arange(1, num_flips + 1) - cumulative_heads

# Probabilities over time
prob_heads = cumulative_heads / np.arange(1, num_flips + 1)
prob_tails = cumulative_tails / np.arange(1, num_flips + 1)

# Create interactive plot
fig = go.Figure()

# Add traces for heads and tails
fig.add_trace(go.Scatter(
    x=np.arange(1, num_flips + 1),
    y=prob_heads,
    mode="lines",
    name="Probability of Heads",
    line=dict(color="blue")
))

fig.add_trace(go.Scatter(
    x=np.arange(1, num_flips + 1),
    y=prob_tails,
    mode="lines",
    name="Probability of Tails",
    line=dict(color="orange")
))

# Add theoretical probability line
fig.add_trace(go.Scatter(
    x=np.arange(1, num_flips + 1),
    y=[0.5] * num_flips,
    mode="lines",
    name="Theoretical Probability (0.5)",
    line=dict(color="red", dash="dash")
))

# Customize layout
fig.update_layout(
    title="Probability of Heads or Tails Over Time",
    xaxis_title="Number of Flips",
    yaxis_title="Probability",
    hovermode="x unified",
    showlegend=True,
    template="plotly_white"
)

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# Add signature
st.markdown("---")
st.markdown("Developed by **Dr. Jishan Ahmed**")