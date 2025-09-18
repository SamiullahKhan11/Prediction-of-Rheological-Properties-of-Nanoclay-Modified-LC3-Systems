import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# Load Models
# =========================
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Could not load model at {path}: {e}")
        return None

model_SYS = load_model("GUI_SYS.joblib")
model_DYS = load_model("GUI_DYS.joblib")
model_PV = load_model("GUI_PV.joblib")

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Prediction of Rheological properties", layout="wide")

# =========================
# Custom Styling (Fonts & Layout)
# =========================
st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            font-size: 18px !important;
        }
        .title-box {
            background-color: DodgerBlue;
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 34px !important;
            font-weight: bold;
        }
        .step-title {
            font-size: 24px !important;
            font-weight: bold;
            color: DodgerBlue;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .stNumberInput input {
            font-size: 20px !important;
            height: 55px !important;
        }
        div[data-testid="stNumberInput"] label {
            font-size: 18px !important;
            font-weight: bold;
        }
        .footer {
            margin-top: 30px;
            text-align: center;
            font-size: 16px !important;
            color: gray;
        }
        div.stButton > button:first-child {
            background-color: red;
            color: white;
            font-size: 20px;
            font-weight: bold;
            border-radius: 8px;
            width: 100%;
            height: 50px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Title
# =========================
st.markdown('<div class="title-box">Prediction of Rheological Properties of Nanoclay-Modified LC³ Systems</div>', unsafe_allow_html=True)

st.write(
    
    """

**This application is developed to predict key rheological properties, such as static yield stress (SYS), dynamic yield stress (DYS), and plastic viscosity (PV), at 10 and 30 minutes, based on user-defined LC³ mix design inputs.**

The predictions are driven by trained machine learning based **Random Forest** models, developed from an extensive experimental dataset involving various binder compositions along with nanoclay dosages. This tool is designed to support sustainable mix design and process optimization by predicting the rheological performance of advanced nanoclay-modified LC³-based cementitious systems.

**⚠️ Input Validation Notes:**

- **Binder Composition Constraint:**

The sum of the binder components, ordinary portland cement (OPC), calcined clay (CC), limestone powder (LP), and gypsum (GYP), **must equal 100%**. If the total is either more or less than 100%, the app will halt execution to prevent invalid predictions.

- **Recommended Input Ranges:**

All input parameters are checked against suggested value ranges. If any parameter falls outside its recommended bounds, the app will still proceed, but a cautionary message will be displayed. **Users are advised to apply thier own judgment** when interpreting results based on **out-of-range inputs**, as prediction reliability may be reduced.

**This intuitive interface is suitable for both academic research and industrial implementation, particularly in developing low-carbon, high-performance binders for modern construction.**   
""")

# =========================
# Step 1 – User Inputs
# =========================
st.markdown('<div class="step-title">Step 1: Enter Mix Parameters</div>', unsafe_allow_html=True)

# Row 1
col1, col2, col3 = st.columns(3)
with col1:
    opc = st.number_input("OPC (%) [40–100]", min_value=0.0, value=50.0)
with col2:
    cc = st.number_input("CC (%) [0–43]", min_value=0.0, value=20.0)
with col3:
    lp = st.number_input("LP (%) [0–23]", min_value=0.0, value=10.0)

# Row 2
col4, col5, col6 = st.columns(3)
with col4:
    nc = st.number_input("NC (%) [0–5]", min_value=0.0, value=2.0)
with col5:
    gyp = st.number_input("GYP (%) [0–3]", min_value=0.0, value=2.0)
with col6:
    sb = st.number_input("S/B [0.5–2.5]", min_value=0.0, value=1.0)

# Row 3
col7, col8, _ = st.columns([1, 1, 1])
with col7:
    wb = st.number_input("W/B [0.4–0.55]", min_value=0.0, value=0.45)
with col8:
    sp = st.number_input("SP (%) [0.5–2.5]", min_value=0.0, value=1.0)

# =========================
# Step 2 – Validation
# =========================
st.markdown('<div class="step-title">Step 2: Validate Inputs</div>', unsafe_allow_html=True)

binder_sum = opc + cc + lp + gyp
if binder_sum != 100:
    st.error(f"❌ Binder content must equal 100%. Current sum = {binder_sum:.2f}%")
    st.stop()
else:
    st.success("✅ Binder content check passed (sum = 100%).")

ranges = {
    "OPC (%)": (40, 100, opc),
    "CC (%)": (0, 43, cc),
    "LP (%)": (0, 23, lp),
    "NC (%)": (0, 5, nc),
    "GYP (%)": (0, 3, gyp),
    "S/B": (0.5, 2.5, sb),
    "W/B": (0.4, 0.55, wb),
    "SP (%)": (0.5, 2.5, sp),
}
out_of_range = []
for param, (low, high, val) in ranges.items():
    if not (low <= val <= high):
        out_of_range.append(f"{param} = {val} (allowed: {low}–{high})")

if out_of_range:
    st.warning("⚠️ Some parameters are outside the recommended ranges:\n" + "\n".join(out_of_range))
else:
    st.success("✅ All inputs are within recommended ranges.")

# =========================
# Step 3 – Predictions
# =========================
st.markdown('<div class="step-title">Step 3: Predictions</div>', unsafe_allow_html=True)

if model_SYS and model_DYS and model_PV:
    if st.button("Predict rheological parameters", key="predict_btn"):
        results = []
        feature_names = ["OPC (%)", "CC (%)", "LP (%)", "NC (%)", "GYP (%)",
                         "S/B", "W/B", "SP (%)", "Time"]

        for time_val, label in zip([0, 1], ["10 min", "30 min"]):
            input_values = [opc, cc, lp, nc, gyp, sb, wb, sp, time_val]
            input_data = pd.DataFrame([input_values], columns=feature_names)

            try:
                sys_pred = model_SYS.predict(input_data)[0]
                dys_pred = model_DYS.predict(input_data)[0]
                pv_pred = model_PV.predict(input_data)[0]
                results.append([label, sys_pred, dys_pred, pv_pred])
            except Exception as e:
                st.error(f"Prediction failed for time={label}: {e}")

        if results:
            df_results = pd.DataFrame(results, columns=["Time", "SYS (Pa)", "DYS (Pa)", "PV (Pa.s)"])

            st.subheader("Prediction Results")

            # --- Build full HTML table as one string ---
            table_html = """
            <style>
                .rheo-table {
                    width: 80%;
                    margin: 0 auto;
                    border-collapse: collapse;
                    font-family: Arial, sans-serif;
                    font-size: 18px;
                }
                .rheo-table th {
                    background-color: DodgerBlue;
                    color: white;
                    padding: 12px;
                    text-align: center;
                    border: 1px solid #ccc;
                }
                .rheo-table td {
                    padding: 10px;
                    text-align: center;
                    border: 1px solid #ccc;
                }
                .rheo-table tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
            </style>
            <table class="rheo-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>SYS (Pa)</th>
                        <th>DYS (Pa)</th>
                        <th>PV (Pa.s)</th>
                    </tr>
                </thead>
                <tbody>
            """

            for _, row in df_results.iterrows():
                table_html += f"<tr><td>{row['Time']}</td><td>{row['SYS (Pa)']:.2f}</td><td>{row['DYS (Pa)']:.2f}</td><td>{row['PV (Pa.s)']:.2f}</td></tr>"

            table_html += "</tbody></table>"

            st.markdown(table_html, unsafe_allow_html=True)


            # --- Graphs
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Yield Stress vs Time")
                plt.figure(figsize=(5, 4))
                sns.lineplot(x="Time", y="SYS (Pa)", data=df_results, marker="o", color="red", label="SYS")
                sns.lineplot(x="Time", y="DYS (Pa)", data=df_results, marker="s", color="dodgerblue", label="DYS")
                plt.ylabel("Yield Stress (Pa)")
                plt.xlabel("Time")
                plt.legend()
                st.pyplot(plt)
                plt.clf()  # Clear plot to avoid overlap

            with col2:
                st.subheader("Plastic Viscosity vs Time")
                plt.figure(figsize=(5, 4))
                sns.lineplot(x="Time", y="PV (Pa.s)", data=df_results, marker="o", color="green", label="PV")
                plt.ylabel("Plastic Viscosity (Pa.s)")
                plt.xlabel("Time")
                plt.legend()
                st.pyplot(plt)
                plt.clf()  # Clear plot to avoid overlap
else:
    st.error("Models could not be loaded. Please check the .joblib files.")


# =========================
# Footer
# =========================
st.markdown('<div class="footer">-------- End of App --------</div>', unsafe_allow_html=True)






