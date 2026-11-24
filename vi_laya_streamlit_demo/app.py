from __future__ import annotations

import time

import streamlit as st

from predictor import LayaViClassifier

st.set_page_config(
    page_title="Vi Intent Classification Demo",
    page_icon="📶",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1120px; padding-top: 2.2rem;}
      [data-testid="stMetric"] {padding: 12px 4px;}
      .demo-note {border-left: 4px solid #f3a712; padding: 10px 14px;
                  background: #fff9e9; border-radius: 4px; color: #503b00;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Vi Customer Intent Classifier")
st.caption("Fine-tuned Laya model · two-stage family and intent classification")
st.markdown(
    '<div class="demo-note"><strong>Demo routing threshold.</strong> The default 0.80 '
    "threshold is for demonstration only and is not an approved production policy.</div>",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the trained Vi model…")
def load_classifier() -> LayaViClassifier:
    return LayaViClassifier()


with st.sidebar:
    st.header("Routing settings")
    threshold = st.slider(
        "Combined confidence threshold", min_value=0.0, max_value=1.0,
        value=0.80, step=0.01, format="%.2f",
    )
    st.caption("Auto-handle when combined confidence meets this demo threshold.")
    st.divider()
    st.write("Model: trained local Laya checkpoint")
    st.write("Pipeline: family → intent")
    st.caption("Hindi and Hinglish are accepted as text input.")

try:
    classifier = load_classifier()
except Exception as exc:
    st.error(f"Model could not be loaded: {exc}")
    st.info("Check the checkpoint path and install the pinned dependencies from requirements.txt.")
    st.stop()

with st.sidebar:
    st.caption(
        f"Family-specific catalog: {classifier.intent_count} intents across "
        f"{len(classifier.family_intents)} families."
    )

examples = {
    "English · recharge not credited": "I recharged yesterday but my balance has not been credited.",
    "English · network issue": "My mobile data is not working even though I have an active plan.",
    "Hinglish · balance check": "Mera data balance aur validity check karni hai.",
    "Hinglish · talk to support": "Mujhe customer care executive se baat karni hai.",
    "Hindi · roaming": "मुझे अंतरराष्ट्रीय रोमिंग चालू करनी है।",
    "Out of scope · example": "Can you recommend a good recipe for dinner?",
}

example_label = st.selectbox("Try an example", ["Choose an example", *examples])
default_text = examples.get(example_label, "")
with st.form("classify_form"):
    utterance = st.text_area(
        "Customer message",
        value=default_text,
        placeholder="Type an English, Hindi or Hinglish customer message…",
        height=130,
    )
    submitted = st.form_submit_button("Classify message", type="primary")

if submitted:
    if not utterance.strip():
        st.warning("Enter a customer message before classifying.")
    else:
        try:
            started = time.perf_counter()
            result = classifier.predict(utterance.strip(), threshold=threshold)
            latency_ms = (time.perf_counter() - started) * 1000

            st.subheader("Classification")
            if result["intent_mapping_available"]:
                st.caption(
                    "Confidence is the model probability for each selected choice; "
                    "combined confidence multiplies the family and intent probabilities."
                )
            else:
                st.info(result["mapping_message"])
            family_col, intent_col, overall_col = st.columns(3)
            family_col.metric("Intent family", result["family"])
            intent_col.metric("Predicted intent", result["intent"] or "Not configured")
            overall_col.metric(
                "Combined confidence",
                f'{result["confidence"]:.1%}' if result["confidence"] is not None else "—",
            )

            confidence_cols = st.columns(3)
            confidence_cols[0].metric("Family confidence", f'{result["family_confidence"]:.1%}')
            confidence_cols[1].metric(
                "Intent confidence",
                f'{result["intent_confidence"]:.1%}' if result["intent_confidence"] is not None else "—",
            )
            confidence_cols[2].metric("Inference latency", f"{latency_ms:.1f} ms")

            if result["decision"] == "AUTO_HANDLE":
                st.success(f"Decision: Auto-handle · threshold {threshold:.0%}")
            else:
                st.warning(f"Decision: Human review · threshold {threshold:.0%}")
            with st.expander("Raw prediction details"):
                st.json(result["raw"])
        except Exception as exc:
            st.error(f"Inference failed: {exc}")

st.divider()
st.caption("Demo only. Validate the taxonomy, calibration and routing threshold before production use.")
