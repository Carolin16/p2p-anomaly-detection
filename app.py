import streamlit as st
import requests

st.set_page_config(page_title="Invoice Fraud Detection",  layout="wide")

API_BASE = "https://carolinjames-p2p-anomaly-api.hf.space"

st.markdown("""
<style>
/* Tighten column subheaders */
div[data-testid="column"] h3 {
    margin-top: -0.75rem !important;
    margin-bottom: 0.5rem !important;
}
/* Remove excess padding above/below st.write paragraphs */
div[data-testid="stMarkdownContainer"] p {
    margin-bottom: 0.25rem !important;
}
/* Equal-height explanation cards */
div[data-testid="stAlert"] {
    min-height: 140px;
}
/* Tighten metric spacing */
div[data-testid="stMetric"] {
    padding: 0 !important;
}
.risk-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

SCENARIOS = {
    "overbilling": {
        "invoice_id": "INV-15450", "vendor_id": "V-042",
        "vendor_name": "GlobalParts Co", "vendor_category": "Industrial",
        "po_reference": "PO-60249", "po_amount": 56228.92,
        "invoice_amount": 83305.81, "gr_amount": 56228.92,
        "deviation_pct": 48.15, "days_since_last_invoice": 45,
        "is_new_vendor": False, "three_way_match": False,
        "invoice_date": "2024-03-12", "anomaly_type": None, "label": 0
    },
    "duplicate": {
        "invoice_id": "INV-00892", "vendor_id": "V-017",
        "vendor_name": "TechSupply Ltd", "vendor_category": "Technology",
        "po_reference": "PO-30014", "po_amount": 22000,
        "invoice_amount": 21500, "gr_amount": 21500,
        "deviation_pct": -2.27, "days_since_last_invoice": 3,
        "is_new_vendor": False, "three_way_match": True,
        "invoice_date": "2024-02-05", "anomaly_type": None, "label": 0
    },
    "phantom": {
        "invoice_id": "INV-44102", "vendor_id": "V-088",
        "vendor_name": "FastFreight Inc", "vendor_category": "Logistics",
        "po_reference": "PO-77021", "po_amount": 31000,
        "invoice_amount": 30500, "gr_amount": 0,
        "deviation_pct": -1.6, "days_since_last_invoice": 60,
        "is_new_vendor": False, "three_way_match": False,
        "invoice_date": "2024-04-18", "anomaly_type": None, "label": 0
    },
    "newvendor": {
        "invoice_id": "INV-99001", "vendor_id": "V-991",
        "vendor_name": "Apex Solutions LLC", "vendor_category": "Consulting",
        "po_reference": "PO-99100", "po_amount": 15000,
        "invoice_amount": 14800, "gr_amount": 14800,
        "deviation_pct": -1.3, "days_since_last_invoice": 999,
        "is_new_vendor": True, "three_way_match": True,
        "invoice_date": "2024-05-01", "anomaly_type": None, "label": 0
    }
}

def render_risk_score(score: float):
    pct = int(score * 100)
    if pct >= 70:
        color, label = "#e53935", "HIGH RISK"
    elif pct >= 40:
        color, label = "#fb8c00", "MEDIUM RISK"
    else:
        color, label = "#43a047", "LOW RISK"
    st.markdown(
        f"""
        <div style="margin-bottom:12px;">
            <div class="risk-label" style="color:{color};">{label} — {pct}%</div>
            <div style="background:#e0e0e0;border-radius:6px;height:10px;width:100%;">
                <div style="background:{color};width:{pct}%;height:10px;border-radius:6px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ── Header ──
st.title("Invoice Fraud Detection")
st.write("Select a scenario below to see how the system detects and explains invoice fraud in real time.")
st.divider()

# ── Scenario buttons ──
st.subheader("Pick a Scenario")
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Overbilling", use_container_width=True, type="primary"):
        st.session_state.invoice = SCENARIOS["overbilling"]
        st.session_state.pop("result", None)
        st.session_state.pop("explanation", None)
with col2:
    if st.button("Duplicate Invoice", use_container_width=True, type="primary"):
        st.session_state.invoice = SCENARIOS["duplicate"]
        st.session_state.pop("result", None)
        st.session_state.pop("explanation", None)
with col3:
    if st.button("Phantom Delivery", use_container_width=True, type="primary"):
        st.session_state.invoice = SCENARIOS["phantom"]
        st.session_state.pop("result", None)
        st.session_state.pop("explanation", None)
with col4:
    if st.button("New Vendor Risk", use_container_width=True, type="primary"):
        st.session_state.invoice = SCENARIOS["newvendor"]
        st.session_state.pop("result", None)
        st.session_state.pop("explanation", None)

st.divider()

# ── Two column layout: Invoice | Result ──
if "invoice" in st.session_state:
    inv = st.session_state.invoice
    left, right = st.columns(2)

    # ── LEFT: ALL invoice details ──
    with left:
        st.subheader("Invoice Details")
        st.write(f"**Vendor:** {inv['vendor_name']}")
        st.write(f"**Category:** {inv['vendor_category']}")
        st.write(f"**PO Reference:** {inv['po_reference']}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Invoice Amount", f"${inv['invoice_amount']:,.0f}")
        m2.metric("PO Amount", f"${inv['po_amount']:,.0f}")
        m3.metric("Goods Received", f"${inv['gr_amount']:,.0f}")

        st.write(f"**Days Since Last Invoice:** {inv['days_since_last_invoice']}")
        st.write(f"**New Vendor:** {'Yes' if inv['is_new_vendor'] else 'No'}")
        st.write(f"**3-Way Match:** {'Pass' if inv['three_way_match'] else 'Fail'}")

        st.write("")
        if st.button("Run Detection", use_container_width=True, type="primary"):
            with st.spinner("Analyzing invoice..."):
                response = requests.post(f"{API_BASE}/invoice", json=inv)
                st.session_state.result = response.json()
                st.session_state.pop("explanation", None)

    # ── RIGHT: Detection result ──
    with right:
        st.subheader("Detection Result")

        if "result" not in st.session_state:
            st.info("Run detection to see results here.")
        else:
            result = st.session_state.result

            render_risk_score(result["ml_score"])

            if result["anomaly"]:
                st.error("This invoice has been flagged for review")
                for flag in result["flags"]:
                    if flag["is_anomaly"]:
                        if flag["anomaly_type"] == "overbilling":
                            st.markdown(
                                f"**Overbilling** — {inv['vendor_name']} invoiced "
                                f"\${inv['invoice_amount']:,.2f} but the PO only allows "
                                f"\${inv['po_amount']:,.2f}. "
                                f"Overage: \${inv['invoice_amount'] - inv['po_amount']:,.2f}."
                            )
                        elif flag["anomaly_type"] == "duplicate_invoice":
                            st.markdown(
                                f"**Duplicate Invoice** — {inv['vendor_name']} already submitted "
                                f"an invoice just {inv['days_since_last_invoice']} days ago for the same PO."
                            )
                        elif flag["anomaly_type"] == "phantom_delivery":
                            st.markdown(
                                f"**Phantom Delivery** — {inv['vendor_name']} is requesting "
                                f"\${inv['invoice_amount']:,.2f} but the warehouse recorded \$0 in goods received."
                            )
                        elif flag["anomaly_type"] == "new_vendor_risk":
                            st.markdown(
                                f"**New Vendor Risk** — {inv['vendor_name']} is an unknown vendor "
                                f"requesting \${inv['invoice_amount']:,.2f}."
                            )
                st.write("")
                if st.button("Get RAG Explanation", use_container_width=True, type="primary"):
                    with st.spinner("Generating explanation..."):
                        payload = {"invoice": inv, "flags": result["flags"]}
                        response = requests.post(f"{API_BASE}/explain", json=payload)
                        st.session_state.explanation = response.json()["explanation"]
            else:
                st.success(f"{inv['vendor_name']}'s invoice looks clean — approved for payment.")

@st.dialog("RAG Explanation", width="large")
def show_explanation_dialog(explanation: str):
    # No $ escaping needed — we render raw HTML, not Streamlit markdown
    sections = {"FINDING": "", "EVIDENCE": "", "RISK": "", "ACTION": ""}
    current = None

    for line in explanation.split("\n"):
        for key in sections:
            if line.startswith(key):
                current = key
                line = line.replace(f"{key}:", "").strip()
        if current:
            sections[current] += line + "\n"

    def format_content(text: str) -> str:
        """Convert dash-prefixed lines to HTML bullets; plain lines to paragraphs."""
        lines = [l for l in text.strip().split("\n") if l.strip()]
        bullet_lines = [l for l in lines if l.strip().startswith("-")]
        if bullet_lines:
            items = "".join(
                f"<li style='margin-bottom:6px;'>{l.strip().lstrip('- ').strip()}</li>"
                for l in lines if l.strip()
            )
            return f"<ul style='margin:0; padding-left:18px;'>{items}</ul>"
        return "<br>".join(lines)

    steps = [
        ("1", "What happened",       sections["FINDING"],  "#1976d2", "#e3f2fd"),
        ("2", "Why we're confident", sections["EVIDENCE"], "#f57c00", "#fff8e1"),
        ("3", "Business impact",     sections["RISK"],     "#c62828", "#ffebee"),
        ("4", "What to do next",     sections["ACTION"],   "#2e7d32", "#f1f8e9"),
    ]

    for num, title, content, color, bg in steps:
        st.markdown(
            f"""
            <div style="
                display: flex;
                gap: 16px;
                align-items: flex-start;
                background: {bg};
                border-left: 4px solid {color};
                border-radius: 8px;
                padding: 16px 20px;
                margin-bottom: 12px;
            ">
                <div style="
                    min-width: 28px;
                    height: 28px;
                    background: {color};
                    color: white;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: 700;
                    font-size: 0.85rem;
                    flex-shrink: 0;
                ">{num}</div>
                <div style="flex:1;">
                    <div style="font-weight: 700; color: {color}; margin-bottom: 6px; font-size: 0.95rem;">
                        {title}
                    </div>
                    <div style="color: #333; line-height: 1.6; font-size: 0.92rem;">
                        {format_content(content)}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

if "explanation" in st.session_state:
    show_explanation_dialog(st.session_state.explanation)