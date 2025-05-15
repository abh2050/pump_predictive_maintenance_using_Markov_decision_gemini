
# MDP Pump Maintenance – Predictive Maintenance App

![](https://www.dxpe.com/wp-content/uploads/2018/03/industrial-pump-maintenance-for-centrifugal-pumps-1.jpg)

## Overview

**MDP Pump Maintenance** is an interactive Streamlit web app that helps you analyze industrial pump sensor data and make optimal maintenance decisions using a **Markov Decision Process (MDP)**.
The app automatically classifies the pump’s health at every time step, recommends the best action (Do Nothing, Preventive, or Corrective Maintenance), and generates a professional maintenance report using **Google Gemini AI**.

---
## App
https://pumppredictivemaintenancemdp.streamlit.app/

## Features

* **Upload your pump’s historical sensor data (CSV)**
* **Classify pump health:** Healthy, Degraded, or Faulty
* **Recommend best maintenance action** at each time point via MDP
* **Auto-generate a professional maintenance report** using Gemini AI
* **Interactive visualizations:** See state & action distributions
* **Example dataset included** for quick testing

---

## What is an MDP?

A **Markov Decision Process** is a mathematical framework for decision-making where outcomes are partly random and partly controlled.
In pump maintenance, this means:

* Learning how the pump's health changes over time
* Calculating the best action at each state (Healthy/Degraded/Faulty) to maximize reliability and minimize cost
* Avoiding unnecessary maintenance and expensive breakdowns

---

## Demo

<img src="https://user-images.githubusercontent.com/abh2050/mdp-demo.gif" width="600"/>

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/abh2050/pump_predictive_maintenance_using_Markov_decision_gemini.git
cd pump_predictive_maintenance_using_Markov_decision_gemini
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Key Python libraries used:**

* `streamlit`
* `pandas`, `numpy`
* `google-generativeai`
* `python-dotenv`

### 3. Add Your Gemini API Key

* Create a `.env` file in the root directory:

  ```
  GEMINI_API_KEY=your_google_gemini_api_key_here
  ```
* If you do not add an API key, Gemini-powered features will not work, but the MDP logic will.

### 4. Run the App

```bash
streamlit run app.py
```

Or replace `app.py` with the actual script name.

---

## Usage

1. **Choose your data source:**

   * Use the provided example dataset (synthetic sensor data)
   * OR upload your own CSV (`timestamp, vibration, temperature, pressure, flow_rate`)

2. **Run Predictive Maintenance Analysis**

   * The app will classify the pump’s state and recommend best actions

3. **View Results:**

   * State and action distributions
   * MDP policy and value function

4. **Generate Maintenance Report (Gemini):**

   * If Gemini API key is configured, click to generate a markdown report summarizing the analysis, detected anomalies, and recommended actions

---

## Example CSV Format

```csv
timestamp,vibration,temperature,pressure,flow_rate
2024-01-01 00:00:00,0.21,61,30,50
2024-01-01 01:00:00,0.32,62,28,49
...
```

* **Required columns:** `timestamp`, `vibration`, `temperature`, `pressure`, `flow_rate`
* Timestamps can be in any standard format.

---

## Key App Logic

* **State Classification:**
  Uses sensor thresholds to label rows as Healthy, Degraded, or Faulty.

* **MDP Value Iteration:**
  Estimates the optimal action for each state using a reward table and learned transition probabilities.

* **AI-Powered Maintenance Report:**
  Summarizes findings, anomalies, and gives recommendations in markdown using Gemini LLM.

---

## Troubleshooting

* **GEMINI\_API\_KEY not found:**
  Add your Gemini API key to a `.env` file in the project directory.
* **Data loading errors:**
  Ensure your CSV follows the required format and includes all columns.

---

## Credits

**Author:** Abhishek Shah
Made with ❤️ using Streamlit, Markov Decision Process, and Gemini AI.

---

## License

This project is for educational and demonstration purposes.
For commercial or production use, review license requirements for Gemini API and Streamlit.

---

*Questions or feedback?*
Open an issue or contact [Abhishek Shah](mailto:your.email@example.com)

---

Let me know if you want a shorter, more technical, or business-focused version!

