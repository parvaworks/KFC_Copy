import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# --- App Config ---
st.set_page_config(page_title="Push Notification Analysis", layout="wide")
st.title("📊 Push Notification Performance Analysis with Statistical Significance")

st.markdown("""
Upload your CSV to analyse open rate performance for **PR (VAR1)** vs **Social (VAR2)**.
You can filter by **day**, **entity**, **slot**, and platform (Android/iOS).

We’ll compute:
- **Direct Open Rate (DOR)** = Direct Opens / Sends  
- **Total Open Rate (TOR)** = Total Opens / Sends  
and run **t-tests** between PR and Social to check if performance differences are statistically significant.
""")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=None, engine="python")
    df.columns = df.columns.str.strip()
    df['Variant'] = df['Variant'].replace({'VAR1': 'PR', 'VAR2': 'Social'})

    # --- Compute Open Rates ---
    df['Android_Direct_Open_Rate'] = df['Direct Opens (Android Push)'] / df['Sends (Android Push)']
    df['Android_Total_Open_Rate'] = df['Total Opens (Android Push)'] / df['Sends (Android Push)']
    df['iOS_Direct_Open_Rate'] = df['Direct Opens (iOS Push)'] / df['Sends (iOS Push)']
    df['iOS_Total_Open_Rate'] = df['Total Opens (iOS Push)'] / df['Sends (iOS Push)']

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
            dor = filtered_df['Android_Direct_Open_Rate'].mean()
            tor = filtered_df['Android_Total_Open_Rate'].mean()
        else:
            dor = filtered_df['iOS_Direct_Open_Rate'].mean()
            tor = filtered_df['iOS_Total_Open_Rate'].mean()
        return dor, tor

    for platform in platform_filter:
        dor, tor = summarize(platform)
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"{platform} Direct Open Rate", f"{dor*100:.2f}%")
        with col2:
            st.metric(f"{platform} Total Open Rate", f"{tor*100:.2f}%")

    # --- Grouping + Statistical Tests ---
    st.subheader("🧮 Variant Comparison & Significance Testing")

    group_cols = st.multiselect(
        "Select Grouping Columns (e.g. Day, Entity, Slot)",
        ['Day', 'Entity', 'Slot'],
        default=['Day', 'Entity']
    )

    if group_cols:
        results = []
        for group_vals, group_df in filtered_df.groupby(group_cols):
            group_label = ', '.join(map(str, group_vals)) if isinstance(group_vals, tuple) else str(group_vals)

            for platform in platform_filter:
                if platform == 'Android':
                    dor_col, tor_col = 'Android_Direct_Open_Rate', 'Android_Total_Open_Rate'
                else:
                    dor_col, tor_col = 'iOS_Direct_Open_Rate', 'iOS_Total_Open_Rate'

                pr = group_df[group_df['Variant'] == 'PR']
                social = group_df[group_df['Variant'] == 'Social']

                if len(pr) > 0 and len(social) > 0:
                    dor_ttest = ttest_ind(pr[dor_col], social[dor_col], equal_var=False, nan_policy='omit')
                    tor_ttest = ttest_ind(pr[tor_col], social[tor_col], equal_var=False, nan_policy='omit')

                    results.append({
                        **{col: val for col, val in zip(group_cols, group_vals if isinstance(group_vals, tuple) else [group_vals])},
                        'Platform': platform,
                        'PR_DOR': pr[dor_col].mean(),
                        'Social_DOR': social[dor_col].mean(),
                        'PR_TOR': pr[tor_col].mean(),
                        'Social_TOR': social[tor_col].mean(),
                        'DOR_pvalue': dor_ttest.pvalue,
                        'TOR_pvalue': tor_ttest.pvalue,
                        'DOR_Significant': '✅' if dor_ttest.pvalue < 0.05 else '❌',
                        'TOR_Significant': '✅' if tor_ttest.pvalue < 0.05 else '❌'
                    })

        result_df = pd.DataFrame(results)

        if not result_df.empty:
            # --- Display results table ---
            st.dataframe(
                result_df.style.format({
                    'PR_DOR': '{:.2%}', 'Social_DOR': '{:.2%}',
                    'PR_TOR': '{:.2%}', 'Social_TOR': '{:.2%}',
                    'DOR_pvalue': '{:.4f}', 'TOR_pvalue': '{:.4f}'
                }),
                use_container_width=True
            )

            # --- Download ---
            csv = result_df.to_csv(index=False)
            st.download_button("📥 Download Results CSV", csv, "variant_significance_results.csv")

            # --- Optional Plot for Significant Differences ---
            st.subheader("📊 Significant Differences Visualization")
            sig_df = result_df[result_df['DOR_Significant'] == '✅']

            if not sig_df.empty:
                plt.figure(figsize=(8,4))
                plt.bar(sig_df['Platform'] + " " + sig_df[group_cols[0]], 
                        sig_df['PR_DOR'] - sig_df['Social_DOR'])
                plt.title("PR - Social DOR Difference (Significant Only)")
                plt.ylabel("Difference in Direct Open Rate")
                plt.xticks(rotation=45)
                st.pyplot(plt)
            else:
                st.info("No statistically significant DOR differences found.")
        else:
            st.warning("No valid PR vs Social comparisons available for the selected filters.")
else:
    st.info("Please upload your CSV to begin analysis.")
