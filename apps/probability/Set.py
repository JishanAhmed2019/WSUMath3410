import streamlit as st
from itertools import chain, combinations
import streamlit.components.v1 as components

def power_set(s):
    return set(frozenset(combo) for combo in chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

def format_set_output(s):
    if all(isinstance(el, frozenset) for el in s):
        formatted_output = ', '.join(['{' + ', '.join(repr(el) for el in subset) + '}' for subset in s])
    else:
        formatted_output = ', '.join(repr(el) for el in s)
    return '{' + formatted_output + '}'

st.title("Set Operations App")

# Use columns instead of beta_columns
col1, col2 = st.columns(2)

with col1:
    set_a_input = st.text_input("Set A (comma separated)", "Apple, Orange, Banana")

with col2:
    set_b_input = st.text_input("Set B (comma separated)", "Apple, Grape, Cherry")

operation = st.selectbox("Choose an Operation", 
                         ["Union", "Intersection", "Difference (A - B)", 
                          "Difference (B - A)", "Power Set of A", "Power Set of B", 
                          "Complement of A", "Complement of B"])

if st.button("Compute"):
    try:
        setA = {item.strip() for item in set_a_input.split(',') if item.strip()}
        setB = {item.strip() for item in set_b_input.split(',') if item.strip()}
        result = None

        if operation == "Union":
            result = setA | setB
        elif operation == "Intersection":
            result = setA & setB
        elif operation == "Difference (A - B)":
            result = setA - setB
        elif operation == "Difference (B - A)":
            result = setB - setA
        elif operation == "Power Set of A":
            result = power_set(setA)
        elif operation == "Power Set of B":
            result = power_set(setB)
        elif operation == "Complement of A":
            universal_set = setA | setB
            result = universal_set - setA
        elif operation == "Complement of B":
            universal_set = setA | setB
            result = universal_set - setB

        st.success(f"Result of {operation}: {format_set_output(result)}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Add a signature
components.html(
    """
    <style>
    .signature {
        position: fixed;
        bottom: 0;
        right: 0;
        font-size: 12px;
        background-color: white;
        padding: 5px;
    }
    </style>
    <div class='signature'>
        Created By Dr. Jishan Ahmed
    </div>
    """,
    height=50,
)