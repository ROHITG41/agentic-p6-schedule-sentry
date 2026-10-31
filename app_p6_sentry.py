# =====================================================================
# APPLICATION: app_p6_sentry.py
# PURPOSE: Shipped Web Application for Autonomous P6 Schedule & Delay Audit
# RUNTIME: Python 3.12 / Streamlit / LangChain / Pydantic / Google Gemini
# =====================================================================

import os
import streamlit as st
import pandas as pd
from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------
# UI CONFIGURATION & BRANDING
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Autonomous P6 Schedule & Delay Sentry",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ Autonomous Primavera P6 Schedule & Forensic Delay Sentry")
st.caption("Engineered by Rohit Kumar, PMP, CEng | Applied Agentic AI for Capital Infrastructure")
st.markdown("---")

# ---------------------------------------------------------------------
# SIDEBAR CONTROLS & API CONFIGURATION
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    # Check Streamlit Secrets first; fallback to text input
    api_key_from_secrets = st.secrets.get("GEMINI_API_KEY", "")
    
    if api_key_from_secrets:
        os.environ["GEMINI_API_KEY"] = api_key_from_secrets
        st.success("API Key loaded securely from Secrets.")
    else:
        user_key = st.text_input("Enter Google Gemini API Key:", type="password")
        if user_key:
            os.environ["GEMINI_API_KEY"] = user_key
    
    st.markdown("### 📋 Audit Governance Standards")
    st.info(
        "**DCMA-14 Integrity Thresholds:**\n"
        "- Open Ends (Logic): <= 5.0%\n"
        "- Leads (Negative Lags): 0.0%\n"
        "- Excessive Lags (> 5d): <= 5.0%\n"
        "- High Float (> 44d): <= 5.0%\n"
        "- CPLI Index: >= 0.95\n"
        "- BEI Velocity Index: >= 0.95"
    )

# ---------------------------------------------------------------------
# STAGE 1: DETERMINISTIC AUDIT LOGIC (CPU registers, 0.00 ms)
# ---------------------------------------------------------------------
def run_dcma14_audit(df_acts: pd.DataFrame, df_rels: pd.DataFrame, data_date: int, crit_length: int):
    total_acts = len(df_acts)
    total_rels = len(df_rels)
    
    # 1. Open ends logic check
    has_pred = set(df_rels['succ_id'].unique())
    has_succ = set(df_rels['pred_id'].unique())
    
    open_ends = df_acts[
        (~df_acts['activity_id'].isin(has_pred)) | 
        (~df_acts['activity_id'].isin(has_succ))
    ]['activity_id'].tolist()
    open_ends_pct = (len(open_ends) / total_acts) * 100.0 if total_acts > 0 else 0.0

    # 2. Leads & Excessive Lags
    leads = df_rels[df_rels['lag_days'] < 0]
    excess_lags = df_rels[df_rels['lag_days'] > 5]
    excess_lags_pct = (len(excess_lags) / total_rels) * 100.0 if total_rels > 0 else 0.0

    # 3. High Float (> 44 days)
    high_float = df_acts[df_acts['total_float_days'] > 44]
    high_float_pct = (len(high_float) / total_acts) * 100.0 if total_acts > 0 else 0.0

    # 4. Critical Path Length Index (CPLI)
    min_tf = df_acts['total_float_days'].min() if not df_acts.empty else 0
    cpli = (crit_length + min_tf) / crit_length if crit_length > 0 else 1.0

    # 5. Baseline Execution Index (BEI)
    planned_due = df_acts[df_acts['planned_finish_day'] <= data_date]
    actually_done = planned_due[planned_due['actual_finish_day'].notna() & (planned_due['actual_finish_day'] <= data_date)]
    bei = len(actually_done) / len(planned_due) if len(planned_due) > 0 else 1.0

    reasons = []
    if open_ends_pct > 5.0:
        reasons.append(f"Open ends logic exceeds 5.0% threshold (Current: {open_ends_pct:.1f}%)")
    if len(leads) > 0:
        reasons.append(f"Negative lags (leads) detected: {len(leads)} instances (Strictly 0.0% permitted)")
    if excess_lags_pct > 5.0:
        reasons.append(f"Excessive lags (> 5 days) exceed 5.0% threshold (Current: {excess_lags_pct:.1f}%)")
    if high_float_pct > 5.0:
        reasons.append(f"High float (> 44 days) exceeds 5.0% threshold (Current: {high_float_pct:.1f}%)")
    if cpli < 0.95:
        reasons.append(f"CPLI is {cpli:.3f}; below minimum project feasibility threshold of 0.95")
    if bei < 0.95:
        reasons.append(f"BEI is {bei:.3f}; execution momentum lags commitment (Target: >= 0.95)")

    return {
        "total_acts": total_acts,
        "total_rels": total_rels,
        "open_ends_count": len(open_ends),
        "open_ends_pct": open_ends_pct,
        "leads_count": len(leads),
        "excess_lags_count": len(excess_lags),
        "excess_lags_pct": excess_lags_pct,
        "high_float_count": len(high_float),
        "high_float_pct": high_float_pct,
        "cpli": cpli,
        "bei": bei,
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "critical_activities": df_acts[df_acts['total_float_days'] <= 0]
    }

# ---------------------------------------------------------------------
# SAMPLE DATASET GENERATOR
# ---------------------------------------------------------------------
def load_sample_schedule():
    acts = pd.DataFrame([
        {"activity_id": "ACT-1010", "name": "CUP Primary Chilled Header Fab", "duration_days": 21, "total_float_days": -21, "planned_finish_day": 45, "actual_finish_day": None},
        {"activity_id": "ACT-1020", "name": "Condenser Water Pump Alignment", "duration_days": 14, "total_float_days": -21, "planned_finish_day": 59, "actual_finish_day": None},
        {"activity_id": "ACT-1030", "name": "Cooling Tower VFD Cabling", "duration_days": 10, "total_float_days": 65, "planned_finish_day": 70, "actual_finish_day": None},
        {"activity_id": "ACT-1040", "name": "AHU Air Balancing (Floor 1-3)", "duration_days": 15, "total_float_days": 16, "planned_finish_day": 60, "actual_finish_day": None},
        {"activity_id": "ACT-1050", "name": "Integrated Systems Testing (IST)", "duration_days": 25, "total_float_days": -21, "planned_finish_day": 90, "actual_finish_day": None},
    ])
    rels = pd.DataFrame([
        {"pred_id": "ACT-1010", "succ_id": "ACT-1020", "type": "FS", "lag_days": 25}, # Artificial lag (+25d)
        {"pred_id": "ACT-1020", "succ_id": "ACT-1050", "type": "FS", "lag_days": -5}, # Prohibited lead (-5d)
        {"pred_id": "ACT-1040", "succ_id": "ACT-1050", "type": "FS", "lag_days": 0},
        # ACT-1030 is dangling / open end
    ])
    return acts, rels

# ---------------------------------------------------------------------
# MAIN DASHBOARD WORKFLOW
# ---------------------------------------------------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1. Ingest Primavera P6 Snapshot")
    use_sample = st.checkbox("Load Sample Metro Station Central Plant Schedule", value=True)
    
    if use_sample:
        df_activities, df_relationships = load_sample_schedule()
    else:
        file_acts = st.file_uploader("Upload Activities CSV", type=["csv"])
        file_rels = st.file_uploader("Upload Relationships CSV", type=["csv"])
        if file_acts and file_rels:
            df_activities = pd.read_csv(file_acts)
            df_relationships = pd.read_csv(file_rels)
        else:
            st.stop()

    st.write("### Schedule Activity Register Preview", df_activities.head())
    st.write("### Logic Relationships Preview", df_relationships.head())

with col_right:
    st.subheader("2. Deterministic DCMA-14 Pre-Flight Audit")
    if st.button("🚀 Execute Forensic Integrity Check", type="primary"):
        audit_results = run_dcma14_audit(df_activities, df_relationships, data_date=50, crit_length=90)
        
        # Display Metrics Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Open Ends", f"{audit_results['open_ends_pct']:.1f}%", "Max 5%", delta_color="inverse")
        m2.metric("Negative Leads", f"{audit_results['leads_count']}", "Must be 0", delta_color="inverse")
        m3.metric("Critical Float", f"{df_activities['total_float_days'].min()} d", "Slipping", delta_color="inverse")
        m4.metric("CPLI Index", f"{audit_results['cpli']:.2f}", "Min 0.95", delta_color="normal" if audit_results['cpli'] >= 0.95 else "inverse")

        if audit_results['passed']:
            st.success("✅ DCMA-14 INTEGRITY AUDIT PASSED: Schedule network is statistically sound.")
        else:
            st.error("❌ DCMA-14 INTEGRITY AUDIT FAILED: Network logic manipulation detected.")
            for r in audit_results['reasons']:
                st.warning(f"⚠️ {r}")

        st.markdown("---")
        st.subheader("3. Multi-Agent Contractual Adjudication (FIDIC Cl. 8.4)")
        
        if not os.environ.get("GEMINI_API_KEY"):
            st.info("💡 To run the AI Contractual Adjudicator, please enter your Gemini API key in the left sidebar or configure Secrets.")
        else:
            with st.spinner("Agentic Loop Analyzing Schedule Graph & Evaluating Entitlement..."):
                llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.0)
                
                adjudication_prompt = ChatPromptTemplate.from_messages([
                    ("system", 
                     "You are the Lead Claims Adjudicator for a major infrastructure capital program. "
                     "Analyze the provided DCMA-14 audit results and Critical Path slip data. "
                     "Evaluate whether the contractor's claim for a 21-day delay on ACT-1010 due to "
                     "site access denial is contractually valid under FIDIC Clause 8.4, and highlight "
                     "any artificial float hoarding tactics identified in the schedule network."),
                    ("human", 
                     "AUDIT METRICS:\n{metrics}\n\nCRITICAL ACTIVITIES:\n{crit_acts}\n\n"
                     "Output a structured Executive Adjudication Memo with: Findings, Schedule Manipulation Defects, and Legal Ruling.")
                ])
                
                chain = adjudication_prompt | llm
                response = chain.invoke({
                    "metrics": str(audit_results),
                    "crit_acts": audit_results['critical_activities'].to_string()
                })
                
                # Format response content safely as a string
                if isinstance(response.content, list):
                    report_text = "\n".join([str(p.get("text", p)) if isinstance(p, dict) else str(p) for p in response.content])
                else:
                    report_text = str(response.content)

                st.markdown(report_text)
                st.download_button(
                    label="📥 Download Adjudication Report",
                    data=report_text,
                    file_name="FIDIC_Delay_Adjudication.doc",
                    mime="text/markdown"
                )
