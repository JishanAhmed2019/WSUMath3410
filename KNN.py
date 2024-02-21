import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# Initialize session state for 'k' value if it doesn't exist
if 'k_value' not in st.session_state:
    st.session_state['k_value'] = 0

# Function to create the data and figure
def create_figure(k_value):
    np.random.seed(11)
    df = pd.DataFrame({'X1': np.random.randint(1, 10, 9),
                       'X2': np.random.randint(1, 10, 9),
                        'Y': np.random.choice(['Class 2', 'Class 1'], size=9)})
    df.loc[len(df)] = [6, 3, 'Unknown']  # query point
    df['Distance'] = ((df[['X1', 'X2']] - df.iloc[-1, :2]) ** 2).sum(axis=1)  # distances from query point
    df = df.sort_values(by='Distance')
    df['Predicted Class'] = ['Unknown', 'Class 1', 'Class 1', 'Class 1', 'Class 1', 'Class 1', 'Class 1', 'Class 2', 'Class 2', 'Class 2']

    # Plot with plotly
    color_dict = {"Class 1": "#636EFA", "Class 2": "#EF553B", "Unknown": "#7F7F7F"}
    fig = px.scatter(df, x="X1", y="X2", color='Y', color_discrete_map=color_dict,
                     range_x=[0, 10], range_y=[0, 10],
                     width=650, height=520)
    fig.update_traces(marker=dict(size=20,
                                  line=dict(width=1)))

    shape_dict = {}
    for k in range(0, k_value + 1):
        shape_dict[k] = [dict(type="line", xref="x", yref="y", x0=x, y0=y, x1=6, y1=3, layer='below',
                              line=dict(color="Black", width=2)) for x, y in df.iloc[1:k+1, :2].to_numpy()]
        if k != 0:
            shape_dict[k].append(dict(type="circle", xref="x", yref="y", x0=5.75, y0=2.75, x1=6.25, y1=3.25,
                                      fillcolor=color_dict[df.iloc[k, 4]]))
    fig.update_layout(shapes=shape_dict[k_value])

    return fig

# Streamlit app
st.title('Interactive k-Nearest Neighbors Visualization')

# Improved layout for '+' and '-' buttons
st.write("Adjust 'k' value:")
col1, col2 = st.columns([1, 1])

with col1:
    decrease = st.button('Decrease K', key='decrease')
with col2:
    increase = st.button('Increase K', key='increase')

if decrease:
    if st.session_state.k_value > 0:
        st.session_state.k_value -= 1
if increase:
    if st.session_state.k_value < 9:
        st.session_state.k_value += 1

st.write(f"Current 'k' value: {st.session_state.k_value}")

# Display the figure with the current 'k' value
fig = create_figure(st.session_state.k_value)
st.plotly_chart(fig)

# Signature
st.markdown("*Dr. Jishan Ahmed*")
