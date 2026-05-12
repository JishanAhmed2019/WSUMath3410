# bayes_app.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configure page
st.set_page_config(
    page_title="Bayesian Machine Analyzer",
    page_icon="⚙️",
    layout="centered"
)

# Set seaborn theme
sns.set_theme(style="whitegrid", palette="pastel")

# Custom CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    .header {
        background: linear-gradient(45deg, #2c3e50, #3498db);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .sidebar .slider {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .metric-box {
        background: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #3498db;
    }
    
    .footer {
        background: #2c3e50;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 2rem;
        text-align: center;
    }
    
    .plot-container {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header with gradient
    st.markdown("""
    <div class="header">
        <h1 style="margin:0;text-align:center">🏭 Bayesian Machine Failure Analyzer</h1>
        <p style="text-align:center;margin:0.5rem 0">Visualizing Conditional Probability in Industrial Systems</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.markdown("### ⚙️ Control Panel")
        failure_rate = st.slider(
            "🚨 Machine Failure Rate (%)",
            min_value=1.0, max_value=50.0, value=10.0,
            step=0.5, format="%.1f%%"
        ) / 100

        detection_rate = st.slider(
            "✅ Test Detection Rate (%)",
            min_value=50.0, max_value=100.0, value=90.0,
            step=0.5, format="%.1f%%"
        ) / 100

        false_alarm = st.slider(
            "⚠️ False Alarm Rate (%)",
            min_value=1.0, max_value=50.0, value=20.0,
            step=0.5, format="%.1f%%"
        ) / 100

    # Calculate probabilities
    total_machines = 1000
    failures = int(total_machines * failure_rate)
    working = total_machines - failures
    
    true_positives = int(failures * detection_rate)
    false_negatives = failures - true_positives
    false_positives = int(working * false_alarm)
    true_negatives = working - false_positives
    
    p_failure_given_positive = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    # Left: Machine population
    ax1.set_title(f"🏗️ Factory Status ({total_machines} Machines)", fontsize=14, pad=20, weight='bold')
    ax1.barh(['Machines'], [total_machines], color='#b3e0ff', height=0.5)
    ax1.barh(['Machines'], [failures], color='#ff6b6b', height=0.5)
    
    ax1.text(failures/2, 0, f"{failures} Failures\n({failure_rate:.1%})", 
            ha='center', va='center', color='white', fontsize=12, weight='bold')
    ax1.text(failures + working/2, 0, f"{working} Working\n({1-failure_rate:.1%})", 
            ha='center', va='center', color='#2c3e50', fontsize=12, weight='bold')
    ax1.axis('off')

    # Right: Test results
    ax2.set_title(f"🔍 Maintenance Test Analysis\nProbability of Failure Given Alarm: {p_failure_given_positive:.1%}", 
                fontsize=14, pad=20, weight='bold')
    results = {
        'True Alarms ✅': true_positives,
        'False Alarms ⚠️': false_positives,
        'Missed Failures ❌': false_negatives,
        'Confirmed OK ✔️': true_negatives
    }
    colors = ['#4cd137', '#fbc531', '#e84118', '#487eb0']
    bars = ax2.barh(list(results.keys()), list(results.values()), color=colors, height=0.6)
    
    for bar in bars:
        width = bar.get_width()
        ax2.text(width/2, bar.get_y() + bar.get_height()/2, 
                f"{width}\n({width/total_machines:.1%})", 
                ha='center', va='center', color='white', fontsize=11, weight='bold')

    # Display in plot container
    with st.container():
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    # Formula breakdown with pre-calculated values
    with st.expander("📝 Detailed Probability Breakdown", expanded=True):
        # Pre-calculate components
        numerator = detection_rate * failure_rate
        false_positive_component = false_alarm * (1 - failure_rate)
        denominator = numerator + false_positive_component
        
        st.markdown("""
        **Bayes' Theorem Formula:**
        """)
        st.latex(r'''
        P(F|A) = \frac{P(A|F)P(F)}{P(A|F)P(F) + P(A|\neg F)P(\neg F)}
        ''')
        
        st.markdown(f"""
        **Step-by-Step Calculation:**
        
        1. Calculate numerator:
        """)
        st.latex(fr'''
        P(A|F) \times P(F) = {detection_rate:.3f} \times {failure_rate:.3f} = {numerator:.3f}
        ''')
        
        st.markdown(f"""
        2. Calculate denominator:
        """)
        st.latex(fr'''
        \begin{{align*}}
        P(A|F)P(F) &+ P(A|\neg F)P(\neg F) \\
        &= ({detection_rate:.3f} \times {failure_rate:.3f}) + ({false_alarm:.3f} \times {1-failure_rate:.3f}) \\
        &= {numerator:.3f} + {false_positive_component:.3f} \\
        &= {denominator:.3f}
        \end{{align*}}
        ''')
        
        st.markdown(f"""
        3. Final probability:
        """)
        st.latex(fr'''
        P(F|A) = \frac{{{numerator:.3f}}}{{{denominator:.3f}}} = {p_failure_given_positive:.1%}
        ''')
        
        st.markdown(f"""
        **Key Terms:**
        - $P(F)$: Base failure rate ({failure_rate:.1%})
        - $P(A|F)$: Detection rate ({detection_rate:.1%})
        - $P(A|¬ F)$: False alarm rate ({false_alarm:.1%})
        - $P(¬F)$: Healthy machines ({1-failure_rate:.1%})
        """)

    # Developer signature
    st.markdown("""
    <div class="footer">
        <p style="margin:0">Developed by Dr. Jishan Ahmed</p>
        <p style="margin:0;font-size:0.8em">Professor of Industrial Analytics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()