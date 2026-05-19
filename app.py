"""Customer Churn Predictor — Advanced Streamlit Dashboard.

Tabs:
  1. Predict          — customer risk score + churn probability gauge
  2. SHAP Explanation — why the model made this prediction
  3. Business Impact  — cost calculator (FP vs FN tradeoff)
  4. Model Performance — confusion matrix, ROC curve, model comparison
"""

import json
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from config import (
    MODEL_PATH, BACKGROUND_DATA_PATH, METRICS_PATH,
    COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, DECISION_THRESHOLD,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
)


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(str(MODEL_PATH))


@st.cache_resource
def load_background():
    if BACKGROUND_DATA_PATH.exists():
        with open(BACKGROUND_DATA_PATH, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


model = load_model()
background = load_background()
metrics = load_metrics()


# ── Input form helper ─────────────────────────────────────────────────────────
def build_input_form():
    """Render the customer input form and return a one-row DataFrame."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        st.markdown("**Services**")
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multi_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_bkp = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        dev_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_sup = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with col3:
        st.markdown("**Billing**")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=0.5)
        total = st.number_input("Total Charges ($)", 0.0, 10000.0, round(monthly * tenure, 2), step=1.0)

    return pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multi_lines, "InternetService": internet,
        "OnlineSecurity": online_sec, "OnlineBackup": online_bkp,
        "DeviceProtection": dev_prot, "TechSupport": tech_sup,
        "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
    }])


# ── Risk gauge ────────────────────────────────────────────────────────────────
def risk_gauge(prob: float):
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"projection": "polar"})
    theta = np.linspace(np.pi, 0, 300)
    # colour zones: green → amber → red
    for start, end, colour in [(np.pi, np.pi * 2 / 3, "#4CAF50"),
                                (np.pi * 2 / 3, np.pi / 3, "#FFC107"),
                                (np.pi / 3, 0, "#F44336")]:
        ax.barh(1, end - start, left=start, height=0.5, color=colour, alpha=0.85)
    needle_angle = np.pi * (1 - prob)
    ax.annotate("", xy=(needle_angle, 1), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=2))
    ax.set_ylim(0, 1.5)
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    ax.set_axis_off()
    ax.set_title(f"Churn Risk: {prob * 100:.1f}%", fontsize=13, fontweight="bold", pad=10)
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────
st.title("📡 Customer Churn Predictor")
st.caption("Voting Classifier Ensemble  ·  ROC-AUC 0.8496  ·  SHAP Explainability")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Predict", "🔍 SHAP Explanation", "💰 Business Impact", "📊 Model Performance"
])


# ── Tab 1: Predict ────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Enter customer details")
    input_df = build_input_form()

    if st.button("Predict Churn Risk", type="primary"):
        prob = float(model.predict_proba(input_df)[0][1])
        pred = int(prob >= DECISION_THRESHOLD)

        st.session_state["input_df"] = input_df
        st.session_state["prob"] = prob
        st.session_state["pred"] = pred

        st.divider()
        r1, r2, r3 = st.columns([1, 1, 1])

        with r1:
            fig = risk_gauge(prob)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with r2:
            st.metric("Churn Probability", f"{prob * 100:.1f}%")
            st.metric("Decision Threshold", f"{DECISION_THRESHOLD * 100:.0f}%")
            if pred == 1:
                st.error("⚠️  HIGH RISK — Customer likely to CHURN")
            else:
                st.success("✅  LOW RISK — Customer likely to STAY")

        with r3:
            risk_level = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
            st.markdown(f"**Risk Level:** {risk_level}")
            if pred == 1:
                st.markdown("**Recommended action:**")
                if input_df["Contract"].iloc[0] == "Month-to-month":
                    st.markdown("- Offer annual contract discount")
                if input_df["TechSupport"].iloc[0] == "No":
                    st.markdown("- Bundle free tech support trial")
                if input_df["tenure"].iloc[0] < 12:
                    st.markdown("- Assign dedicated onboarding rep")
                st.markdown("- Proactive outreach within 48 hours")


# ── Tab 2: SHAP Explanation ───────────────────────────────────────────────────
with tab2:
    st.subheader("Why did the model make this prediction?")

    if "input_df" not in st.session_state:
        st.info("Run a prediction in the **Predict** tab first.")
    else:
        input_df = st.session_state["input_df"]
        prob = st.session_state["prob"]

        if background is not None:
            try:
                import shap
                st.caption("Computing SHAP values — this may take a few seconds on first run…")

                @st.cache_resource
                def get_explainer(_model, _background):
                    return shap.Explainer(_model.predict_proba, _background)

                explainer = get_explainer(model, background)
                shap_values = explainer(input_df)

                # Waterfall plot for the Churn class (index 1)
                col_wf, col_bar = st.columns(2)

                with col_wf:
                    st.markdown("**Waterfall — individual prediction breakdown**")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    shap.plots.waterfall(shap_values[0, :, 1], show=False)
                    st.pyplot(plt.gcf(), use_container_width=True)
                    plt.close()

                with col_bar:
                    st.markdown("**Feature contributions (magnitude)**")
                    vals = shap_values[0, :, 1].values
                    names = shap_values[0, :, 1].feature_names
                    idx = np.argsort(np.abs(vals))[::-1][:10]
                    colors = ["#F44336" if v > 0 else "#4CAF50" for v in vals[idx]]

                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.barh(range(len(idx)), vals[idx], color=colors)
                    ax.set_yticks(range(len(idx)))
                    ax.set_yticklabels([names[i] for i in idx])
                    ax.set_xlabel("SHAP value (impact on churn probability)")
                    ax.axvline(0, color="black", lw=0.8)
                    ax.invert_yaxis()
                    ax.set_title("Top 10 Features — Red = increases churn risk")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                st.caption(
                    "Red bars push the prediction toward **Churn**. "
                    "Green bars push toward **Stay**. "
                    "Bar length = strength of influence."
                )

            except Exception as e:
                st.warning(f"SHAP computation failed: {e}")
        else:
            st.warning(
                "Background data not found (`background_data.pkl`).  \n"
                "Run the **SHAP section** in the training notebook to generate it, "
                "then restart the app."
            )


# ── Tab 3: Business Impact ────────────────────────────────────────────────────
with tab3:
    st.subheader("Cost-Benefit Analysis of Churn Predictions")
    st.markdown(
        "Adjust the business parameters below. The calculator shows the expected cost "
        "of your model at different operating thresholds."
    )

    c1, c2 = st.columns(2)
    with c1:
        fn_cost = st.number_input(
            "Cost per missed churner (False Negative, $)",
            value=COST_FALSE_NEGATIVE, min_value=0, step=10,
            help="Revenue lost when the model fails to identify a churner",
        )
        fp_cost = st.number_input(
            "Cost per false alarm (False Positive, $)",
            value=COST_FALSE_POSITIVE, min_value=0, step=5,
            help="Cost of sending a retention offer to a customer who wouldn't have churned",
        )

    with c2:
        n_customers = st.number_input("Monthly customer base", value=10000, step=500)
        churn_rate = st.slider("Expected churn rate (%)", 1, 50, 27) / 100

    if metrics:
        cm = metrics["confusion_matrix"]
        tn, fp_m, fn_m, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        total = tn + fp_m + fn_m + tp

        # Scale confusion matrix to the user-specified customer base
        scale = n_customers / total
        TP = int(tp * scale)
        FP = int(fp_m * scale)
        FN = int(fn_m * scale)
        TN = int(tn * scale)

        model_cost = FP * fp_cost + FN * fn_cost
        no_model_cost = int(n_customers * churn_rate) * fn_cost
        savings = no_model_cost - model_cost

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Churners caught (TP)", f"{TP:,}")
        m2.metric("False alarms (FP)", f"{FP:,}", help="Unnecessary retention offers sent")
        m3.metric("Missed churners (FN)", f"{FN:,}", help="Churners not identified")
        m4.metric("Net savings vs no model", f"${savings:,.0f}", delta=f"${savings:,.0f}")

        st.divider()
        st.markdown("**Threshold Sensitivity**")
        st.markdown(
            "Lowering the threshold catches more churners (↑ Recall) but increases false alarms (↑ FP cost). "
            "Raising it reduces false alarms but misses more churners."
        )

        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        rows = []
        for t in thresholds:
            # Simplified linear interpolation for illustration
            adj_recall = min(1.0, metrics["recall"] + (0.5 - t) * 0.5)
            adj_precision = min(1.0, metrics["precision"] - (0.5 - t) * 0.3)
            est_tp = int(n_customers * churn_rate * adj_recall)
            est_fp = int(est_tp * (1 - adj_precision) / max(adj_precision, 0.01))
            est_fn = int(n_customers * churn_rate * (1 - adj_recall))
            cost = est_fp * fp_cost + est_fn * fn_cost
            rows.append({
                "Threshold": t,
                "Recall (%)": round(adj_recall * 100, 1),
                "Precision (%)": round(adj_precision * 100, 1),
                "Churners Caught": est_tp,
                "False Alarms": est_fp,
                "Total Cost ($)": f"{cost:,.0f}",
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.warning("Run `metrics/model_performance.json` generation first.")


# ── Tab 4: Model Performance ──────────────────────────────────────────────────
with tab4:
    st.subheader("Validation Set Performance  (1,409 customers)")

    if metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy",  f"{metrics['accuracy']:.2%}")
        c2.metric("Precision", f"{metrics['precision']:.2%}")
        c3.metric("Recall",    f"{metrics['recall']:.2%}")
        c4.metric("F1 Score",  f"{metrics['f1']:.2%}")
        c5.metric("ROC-AUC",   f"{metrics['auc_roc']:.4f}")

        col_cm, col_roc = st.columns(2)

        with col_cm:
            st.markdown("**Confusion Matrix**")
            cm = np.array(metrics["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(4, 3))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Stay", "Churn"])
            ax.set_yticklabels(["Stay", "Churn"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            for i in range(2):
                for j in range(2):
                    color = "white" if cm[i, j] > cm.max() / 2 else "black"
                    ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                            color=color, fontsize=13, fontweight="bold")
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_roc:
            st.markdown("**ROC Curve**")
            fpr = metrics["roc"]["fpr"]
            tpr = metrics["roc"]["tpr"]
            auc_val = metrics["roc"]["auc"]
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(fpr, tpr, color="#1565C0", lw=2, label=f"Voting Classifier  AUC = {auc_val:.4f}")
            ax.fill_between(fpr, tpr, alpha=0.08, color="#1565C0")
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
            ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.markdown("**Model Comparison (10-fold CV, ROC-AUC)**")
        st.dataframe(pd.DataFrame(metrics["model_comparison"]), use_container_width=True)

        st.markdown("**Class Imbalance — Before vs After SMOTE**")
        smote = metrics.get("smote_comparison")
        if smote:
            st.dataframe(pd.DataFrame(smote), use_container_width=True)
    else:
        st.warning("No metrics found. Run the evaluation section in the notebook.")
