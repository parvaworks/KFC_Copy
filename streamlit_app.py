# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# IMPORTANT: set_page_config must be the first Streamlit command
st.set_page_config(page_title="Push Notification Analysis — Fixed Dataset", layout="wide")

# --- App title and description (only after set_page_config) ---
st.title("📊 Push Notification Performance Analysis — Fixed Dataset")

st.markdown("""
This dashboard analyses open rate performance for **PR (VAR1)** vs **Social (VAR2)**  
using the file: `message_comparison_report-2025-11-11 (1).csv`

We compute:
- **Direct Open Rate (DOR)** = Direct Opens / Sends  
- **Total Open Rate (TOR)** = Total Opens / Sends  
- **Winner Variant** = higher DOR performer  
- **Margin of Victory (%)** = (PR DOR – Social DOR) / Social DOR × 100  
and run **t-tests** to test significance between PR and Social.
""")

# --- Load fixed CSV file ---
DATA_FILE = "message_comparison_report-2025-11-11 (1).csv"

try:
    # read with python engine to allow flexible separators if any
    df = pd.read_csv(DATA_FILE, sep=None, engine="python")
except FileNotFoundError:
    st.error(f"❌ Could not find `{DATA_FILE}`. Please place it in the same folder as this app.")
    st.stop()
except Exception as e:
    st.error(f"Error reading `{DATA_FILE}`: {e}")
    st.stop()

# --- Preprocessing ---
df.columns = df.columns.str.strip()
df['Variant'] = df['Variant'].replace({'VAR1': 'PR', 'VAR2': 'Social'})

# --- Compute Open Rates (guard divide-by-zero just in case) ---
df['Android_Direct_Open_Rate'] = df['Direct Opens (Android Push)'] / df['Sends (Android Push)'].replace(0, np.nan)
df['Android_Total_Open_Rate'] = df['Total Opens (Android Push)'] / df['Sends (Android Push)'].replace(0, np.nan)
df['iOS_Direct_Open_Rate'] = df['Direct Opens (iOS Push)'] / df['Sends (iOS Push)'].replace(0, np.nan)
df['iOS_Total_Open_Rate'] = df['Total Opens (iOS Push)'] / df['Sends (iOS Push)'].replace(0, np.nan)

# --- Sidebar Filters ---
st.sidebar.header("🔎 Filter Data")
day_filter = st.sidebar.multiselect("Select Day(s)", df['Day'].unique(), default=list(df['Day'].unique()))
entity_filter = st.sidebar.multiselect("Select Entity (Cohort)", df['Entity'].unique(), default=list(df['Entity'].unique()))
slot_filter = st.sidebar.multiselect("Select Slot(s)", df['Slot'].unique(), default=list(df['Slot'].unique()))
platform_filter = st.sidebar.multiselect("Select Platform(s)", ['Android', 'iOS'], default=['Android', 'iOS'])

filtered_df = df[df['Day'].isin(day_filter) & df['Entity'].isin(entity_filter) & df['Slot'].isin(slot_filter)]

# --- Overview Metrics ---
st.subheader("📈 Overall Performance Summary")
def summarize(platform):
    if platform == 'Android':
        dor = filtered_df['Android_Direct_Open_Rate'].mean(skipna=True)
        tor = filtered_df['Android_Total_Open_Rate'].mean(skipna=True)
    else:
        dor = filtered_df['iOS_Direct_Open_Rate'].mean(skipna=True)
        tor = filtered_df['iOS_Total_Open_Rate'].mean(skipna=True)
    return dor, tor

for platform in platform_filter:
    dor, tor = summarize(platform)
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"{platform} Direct Open Rate", f"{(dor or 0)*100:.2f}%")
    with col2:
        st.metric(f"{platform} Total Open Rate", f"{(tor or 0)*100:.2f}%")

# --- Grouping + Statistical Tests ---
st.subheader("🧮 Variant Comparison & Significance Testing")

group_cols = st.multiselect(
    "Select Grouping Columns (e.g. Day, Entity, Slot)",
    ['Day', 'Entity', 'Slot'],
    default=['Day', 'Entity']
)

if group_cols:
    results = []
    # Make sure groupby keys are consistent tuples for zip
    for group_vals, group_df in filtered_df.groupby(group_cols):
        # ensure group_vals is a tuple
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)

        for platform in platform_filter:
            if platform == 'Android':
                dor_col, tor_col = 'Android_Direct_Open_Rate', 'Android_Total_Open_Rate'
            else:
                dor_col, tor_col = 'iOS_Direct_Open_Rate', 'iOS_Total_Open_Rate'

            pr = group_df[group_df['Variant'] == 'PR']
            social = group_df[group_df['Variant'] == 'Social']

            if len(pr) > 0 and len(social) > 0:
                dor_ttest = ttest_ind(pr[dor_col].dropna(), social[dor_col].dropna(), equal_var=False, nan_policy='omit')
                tor_ttest = ttest_ind(pr[tor_col].dropna(), social[tor_col].dropna(), equal_var=False, nan_policy='omit')

                pr_dor = pr[dor_col].mean()
                social_dor = social[dor_col].mean()

                # Winner and margin
                if pd.isna(pr_dor) or pd.isna(social_dor):
                    winner = 'N/A'
                    margin = np.nan
                else:
                    if pr_dor > social_dor:
                        winner = 'PR'
                    elif pr_dor < social_dor:
                        winner = 'Social'
                    else:
                        winner = 'Tie'
                    margin = ((pr_dor - social_dor) / social_dor) * 100 if social_dor != 0 else np.nan

                entry = {col: val for col, val in zip(group_cols, group_vals)}
                entry.update({
                    'Platform': platform,
                    'PR_DOR': pr_dor,
                    'Social_DOR': social_dor,
                    'PR_TOR': pr[tor_col].mean(),
                    'Social_TOR': social[tor_col].mean(),
                    'DOR_pvalue': dor_ttest.pvalue,
                    'TOR_pvalue': tor_ttest.pvalue,
                    'DOR_Significant': '✅' if dor_ttest.pvalue < 0.05 else '❌',
                    'TOR_Significant': '✅' if tor_ttest.pvalue < 0.05 else '❌',
                    'Winner_Variant': winner,
                    'Margin_of_Victory (%)': margin
                })
                results.append(entry)

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        st.dataframe(
            result_df.style.format({
                'PR_DOR': '{:.2%}', 'Social_DOR': '{:.2%}',
                'PR_TOR': '{:.2%}', 'Social_TOR': '{:.2%}',
                'DOR_pvalue': '{:.4f}', 'TOR_pvalue'
