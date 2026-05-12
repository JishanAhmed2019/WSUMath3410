import streamlit as st
import random
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Monty Hall Simulator", page_icon="🚪", layout="centered")

# Title and description
st.title("Monty Hall Problem Simulator 🚗🐐")
st.markdown("""
**How it works:**
1. You pick one of three doors.
2. The host opens a door, revealing a goat.
3. Do you **switch** or **stay** with your choice?
4. See how your strategy performs over time!
""")

# Sidebar for user input
st.sidebar.header("Settings")
num_simulations = st.sidebar.slider("Number of Simulations", 100, 10000, 1000, 100)
strategy = st.sidebar.radio("Choose Strategy", ["Switch", "Stay"])

# Function to simulate the Monty Hall Problem
def monty_hall_simulation(num_simulations, switch):
    wins = 0
    win_rates = []

    for i in range(1, num_simulations + 1):
        doors = ['goat', 'goat', 'car']
        random.shuffle(doors)  # Randomly place the car
        contestant_choice = random.randint(0, 2)
        remaining_doors = [i for i in range(3) if i != contestant_choice and doors[i] == 'goat']
        monty_opens = random.choice(remaining_doors)
        final_choice = [i for i in range(3) if i != contestant_choice and i != monty_opens][0] if switch else contestant_choice
        wins += doors[final_choice] == 'car'
        win_rates.append(wins / i)

    return win_rates

# Run simulation
switch = strategy == "Switch"
win_rates = monty_hall_simulation(num_simulations, switch)

# Plot results
st.subheader(f"Results for **{strategy}** Strategy")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, num_simulations + 1), win_rates, label=f"{strategy} Strategy", color="#1f77b4" if switch else "#ff7f0e")
ax.axhline(y=2/3 if switch else 1/3, color="#1f77b4" if switch else "#ff7f0e", linestyle="--", alpha=0.5, label="Theoretical Win Rate")
ax.set_xlabel("Number of Simulations")
ax.set_ylabel("Win Rate")
ax.set_title(f"Win Rates Over Time ({strategy} Strategy)")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Display win rate
st.metric("Final Win Rate", f"{win_rates[-1]:.2%}")

# Signature
st.markdown("---")
st.markdown("Developed by **Dr. Jishan Ahmed**")
