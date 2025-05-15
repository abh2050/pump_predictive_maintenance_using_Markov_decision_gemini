import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables at the beginning
load_dotenv()

# --------- SIDEBAR EXPLANATION ---------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/599/599305.png", width=60)
    st.title("MDP Pump Maintenance")
    st.markdown("---")
    st.markdown("#### What does this app do?")
    st.write("""
    This application helps you analyze sensor data from an industrial pump using a mathematical technique called a **Markov Decision Process (MDP)**.

    - **Upload your pump's historical sensor data (CSV).**
    - The app classifies the pump's health at each row (**Healthy, Degraded, Faulty**).
    - Recommends the **best maintenance action** ("Do Nothing", "Preventive", or "Corrective Maintenance").
    - Generates a professional maintenance report using Google Gemini AI.
    """)
    st.markdown("---")
    st.markdown("#### What is an MDP, in simple terms?")
    st.write("""
    A **Markov Decision Process** is a framework used in decision-making where outcomes are partly random and partly under your control.

    For pump maintenance, it means:
    - The app learns how the pump's health changes over time.
    - It calculates the best action to take at each state (healthy, degraded, or faulty) to maximize long-term performance and minimize cost.
    - This helps avoid unnecessary maintenance and costly breakdowns, improving reliability and saving money.
    """)
    st.markdown("---")
    st.caption("Made with ❤️ by Abhishek Shah")

# --------- MAIN APP ---------
st.markdown(
    "<h1 style='text-align: center;'>MDP Predictive Maintenance for Pump</h1>",
    unsafe_allow_html=True
)

# Check if API key is available
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found in environment variables. API features will not work.")
    st.info("Please add your API key to a .env file in the same directory as the app.")

# --------- DATA LOADING OPTIONS ---------
st.write("Choose your data source:")

data_option = st.radio(
    "Select data source",
    ["Use example dataset", "Upload your own data"],
    horizontal=True
)

df = None

if data_option == "Use example dataset":
    example_path = "/Users/abhishekshah/Desktop/predictive_maintenance/synthetic_pump_data.csv"
    
    if os.path.exists(example_path):
        # Load example data automatically without requiring button click
        df = pd.read_csv(example_path, parse_dates=["timestamp"])
        st.success(f"Loaded example dataset with {len(df)} records")
        st.write("Sample of the example data:")
        st.dataframe(df.head())
    else:
        st.error(f"Example file not found at {example_path}. Please check the file path.")
else:
    uploaded = st.file_uploader(
        "Choose a CSV file (columns: timestamp, vibration, temperature, pressure, flow_rate)", 
        type=["csv"]
    )
    
    if uploaded:
        try:
            df = pd.read_csv(uploaded, parse_dates=["timestamp"])
            st.success(f"Loaded {uploaded.name}")
            st.write("Sample of your data:")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading the uploaded file: {str(e)}")

# --------- MDP PIPELINE ----------
def mdp_pipeline(df, gamma=0.95):
    """Process sensor data through MDP pipeline to determine optimal maintenance actions"""
    try:
        # Create a copy to avoid modifying the original dataframe
        df_copy = df.copy()
        
        def classify_state(row):
            if row["vibration"] > 0.6 or row["temperature"] > 65 or row["pressure"] < 25:
                return "Faulty"
            elif row["vibration"] > 0.3 or row["temperature"] > 62 or row["pressure"] < 28:
                return "Degraded"
            else:
                return "Healthy"
        df_copy["state"] = df_copy.apply(classify_state, axis=1)

        states = ["Healthy", "Degraded", "Faulty"]
        actions = ["Do Nothing", "Preventive Maintenance", "Corrective Maintenance"]
        R = {
            "Healthy": {"Do Nothing": 0, "Preventive Maintenance": -10, "Corrective Maintenance": -30},
            "Degraded": {"Do Nothing": -5, "Preventive Maintenance": -10, "Corrective Maintenance": -30},
            "Faulty": {"Do Nothing": -100, "Preventive Maintenance": -20, "Corrective Maintenance": -50},
        }
        df_copy["best_action"] = df_copy["state"].map(lambda x: "Do Nothing")  # Initialize

        # Count transitions
        transition_counts = {s: {a: {s1: 0 for s1 in states} for a in actions} for s in states}
        action_counts = {s: {a: 0 for a in actions} for s in states}
        for t in range(len(df_copy) - 1):
            s, a, s1 = df_copy.loc[t, "state"], df_copy.loc[t, "best_action"], df_copy.loc[t + 1, "state"]
            transition_counts[s][a][s1] += 1
            action_counts[s][a] += 1

        # Estimate transition probabilities
        estimated_P = {
            s: {
                a: {
                    s1: transition_counts[s][a][s1] / action_counts[s][a]
                    if action_counts[s][a] > 0 else 0
                    for s1 in states
                }
                for a in actions
            }
            for s in states
        }

        # Value iteration
        V = {s: 0 for s in states}
        policy = {s: None for s in states}
        for _ in range(100):
            delta = 0
            new_V = V.copy()
            for s in states:
                values = []
                for a in actions:
                    expected = R[s][a] + gamma * sum(
                        estimated_P[s][a][s1] * V[s1] for s1 in states
                    )
                    values.append(expected)
                best_val = max(values)
                delta = max(delta, abs(V[s] - best_val))
                new_V[s] = best_val
                policy[s] = actions[np.argmax(values)]
            V = new_V
            if delta < 1e-4:
                break

        df_copy["best_action"] = df_copy["state"].map(lambda x: policy[x])
        return df_copy, V, policy
    except Exception as e:
        st.error(f"Error in MDP pipeline: {str(e)}")
        raise e

# --------- GEMINI INTEGRATION ----------
def configure_gemini_api():
    """Configure the Gemini API with your key"""
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            st.error("Gemini API key not found. Please check your .env file.")
            return None
            
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('models/gemini-2.0-flash')  # Using more reliable model
    except Exception as e:
        st.error(f"Error configuring Gemini API: {str(e)}")
        return None

def gemini_analysis(df, V, policy):
    """Generate a maintenance report using Gemini AI based on the MDP results"""
    try:
        state_counts = df["state"].value_counts().to_dict()
        total_states = sum(state_counts.values())
        state_percentages = {s: (count/total_states)*100 for s, count in state_counts.items()}
        
        # Find anomalies
        faulty_indices = df[df["state"] == "Faulty"].index.tolist()
        degraded_indices = df[df["state"] == "Degraded"].index.tolist()
        
        # Safety checks before accessing indices
        faulty_timestamps = []
        if faulty_indices:
            faulty_timestamps = [str(ts) for ts in df.loc[faulty_indices[:5], "timestamp"].tolist()]
            
        degraded_timestamps = []
        if degraded_indices:
            degraded_timestamps = [str(ts) for ts in df.loc[degraded_indices[:5], "timestamp"].tolist()]
        
        # Find transitions with error handling
        state_transitions = []
        for i in range(1, len(df)):
            try:
                if df.loc[i-1, "state"] == "Healthy" and df.loc[i, "state"] == "Faulty":
                    state_transitions.append({
                        "from": "Healthy",
                        "to": "Faulty",
                        "timestamp": str(df.loc[i, "timestamp"]),
                        "metrics": {
                            "vibration": float(df.loc[i, "vibration"]),
                            "temperature": float(df.loc[i, "temperature"]),
                            "pressure": float(df.loc[i, "pressure"]),
                            "flow_rate": float(df.loc[i, "flow_rate"]),
                        }
                    })
            except Exception as e:
                st.warning(f"Error processing transition at index {i}: {str(e)}")
                continue
        
        # Prepare context for Gemini
        analysis_context = {
            "state_percentages": state_percentages,
            "state_transitions": state_transitions[:5],
            "faulty_timestamps": faulty_timestamps,
            "degraded_timestamps": degraded_timestamps,
            "policy": {k: v for k, v in policy.items()},  # Ensure serializable
            "value_function": {k: float(v) for k, v in V.items()}  # Convert to float for JSON
        }
        
        model = configure_gemini_api()
        if not model:
            return "**Error:** Unable to initialize Gemini API. Please check your API key."
        
        prompt = f"""
You are an experienced industrial maintenance engineer analyzing pump sensor data.

The data has been processed through a Markov Decision Process model and classified into three states:
- Healthy: Normal operating conditions
- Degraded: Early warning signs, pump still operational
- Faulty: Critical issues requiring immediate attention

Here is the current state of the pump system:
{json.dumps(analysis_context, indent=2)}

Please provide your maintenance report in **markdown format** (not HTML or code), with section headings, bullet points, tables (in markdown), and bold important terms. The report must include:
1. A brief summary of the pump's current condition
2. Specific times when anomalies were detected and what likely caused them
3. Recommendations for immediate actions
4. Potential maintenance schedule based on the detected patterns
5. Cost-benefit analysis of performing maintenance now vs. waiting
"""
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Gemini API error: {str(e)}")
            return f"**Error generating analysis:** {str(e)}\n\nPlease ensure your API key is valid and you have proper permissions."
    except Exception as e:
        st.error(f"Error in analysis generation: {str(e)}")
        return f"**Error in analysis:** {str(e)}"

# --------- MAIN LOGIC ----------
if df is not None:
    run_analysis = st.button("Run Predictive Maintenance Analysis", key="run_mdp_analysis")
    
    if run_analysis:
        try:
            with st.spinner("Running MDP pipeline..."):
                df_mdp, value_map, optimal_policy = mdp_pipeline(df)
                
            # Display MDP results
            st.markdown("---")
            st.subheader("MDP State Classification and Best Action")
            st.dataframe(df_mdp[["timestamp", "vibration", "temperature", "pressure", "flow_rate", "state", "best_action"]].head(20))
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**State Distribution:**")
                st.write(df_mdp["state"].value_counts())
            
            with col2:
                st.markdown("**Action Distribution:**")
                st.write(df_mdp["best_action"].value_counts())
            
            st.markdown("**Optimal Policy:**")
            st.json(optimal_policy)
            
            st.markdown("**State Value Function:**")
            st.json({k: round(float(v), 2) for k, v in value_map.items()})
            
            # Check if API key exists before attempting Gemini analysis
            if api_key:
                with st.spinner("Generating Gemini maintenance report..."):
                    report = gemini_analysis(df_mdp, value_map, optimal_policy)
                st.markdown("---")
                st.subheader("Gemini-Powered Maintenance Report")
                st.markdown(report)
            else:
                st.warning("Gemini API key not found. Skipping maintenance report generation.")
                
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
    else:
        st.info("Click the button above to run the full MDP + Gemini analysis pipeline.")
else:
    st.warning("Please select a data source to continue.")

st.markdown("---")
st.caption("Powered by Streamlit, Markov Decision Process, and Gemini AI.")