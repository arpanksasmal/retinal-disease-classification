"""
RetinaAI - Diabetic Retinopathy Detection
==========================================
Patient-friendly UI redesign:
- High contrast text throughout (WCAG AA minimum)
- Plain-language explanations replacing clinical jargon
- Clear visual hierarchy with readable section headers
- Meaningful stat row for non-technical users
- Visible footer disclaimer
- Accessible color palette on dark background

Run:
    streamlit run app.py
"""

import os
import io
import base64
import torch
import torch.nn.functional as F
import streamlit as st
import numpy as np
from PIL import Image
from torchvision import transforms
import plotly.graph_objects as go

from model import RetinalCNN

# ─── Constants ────────────────────────────────────────────────────────────────

NUM_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "model_training", "saved_models", "best_model.pt",
)

CLASS_META = {
    "No DR": {
        "grade": "Grade 0",
        "color": "#00D4AA",
        "bg": "rgba(0,212,170,0.10)",
        "border": "rgba(0,212,170,0.5)",
        "icon": "✦",
        "urgency": "✅ No Action Needed",
        "urgency_plain": "Your retina looks healthy!",
        "desc": "No signs of diabetic retinopathy were found in this image. Keep up with your regular check-ups to stay ahead of any changes.",
        "action": "👁 Schedule your next annual eye exam",
        "clinical": "No microaneurysms, haemorrhages, or neovascularisation detected.",
        "severity_label": "Healthy",
        "severity_level": 0,
    },
    "Mild DR": {
        "grade": "Grade 1",
        "color": "#F5C842",
        "bg": "rgba(245,200,66,0.10)",
        "border": "rgba(245,200,66,0.5)",
        "icon": "◈",
        "urgency": "📅 Follow-up in 6 Months",
        "urgency_plain": "Early changes detected — routine monitoring advised.",
        "desc": "Very early signs of diabetic retinopathy are present. Small balloon-like swellings (microaneurysms) have been detected in blood vessels. This is the earliest stage and is manageable with proper care.",
        "action": "📋 Book a follow-up with your eye doctor in 6 months",
        "clinical": "Few microaneurysms visible. No vision-threatening lesions currently detected.",
        "severity_label": "Mild",
        "severity_level": 1,
    },
    "Moderate DR": {
        "grade": "Grade 2",
        "color": "#FF8C42",
        "bg": "rgba(255,140,66,0.10)",
        "border": "rgba(255,140,66,0.5)",
        "icon": "◉",
        "urgency": "📅 Review Needed in 3 Months",
        "urgency_plain": "Moderate changes detected — please see a doctor soon.",
        "desc": "A moderate level of diabetic retinopathy has been detected. There are multiple areas of blood vessel damage including small bleeds and swelling. Medical review is recommended to prevent further progression.",
        "action": "🏥 See your eye doctor within 3 months",
        "clinical": "Dot and blot haemorrhages. Possible hard exudates. Medical review recommended.",
        "severity_label": "Moderate",
        "severity_level": 2,
    },
    "Severe DR": {
        "grade": "Grade 3",
        "color": "#FF4D6D",
        "bg": "rgba(255,77,109,0.10)",
        "border": "rgba(255,77,109,0.5)",
        "icon": "⬡",
        "urgency": "🚨 Urgent — See a Specialist Now",
        "urgency_plain": "Serious damage detected — do not delay treatment.",
        "desc": "Severe diabetic retinopathy has been detected. There is widespread bleeding and damage across the retina. Without prompt treatment, this can progress to vision loss. Please see an ophthalmologist (eye specialist) as soon as possible.",
        "action": "🚑 Seek urgent ophthalmologist referral — do not delay",
        "clinical": "Extensive haemorrhages in all quadrants. Venous beading. IRMA present.",
        "severity_label": "Severe",
        "severity_level": 3,
    },
    "Proliferative DR": {
        "grade": "Grade 4",
        "color": "#C840E9",
        "bg": "rgba(200,64,233,0.10)",
        "border": "rgba(200,64,233,0.5)",
        "icon": "★",
        "urgency": "🆘 Emergency — Immediate Treatment Required",
        "urgency_plain": "Most advanced stage — immediate action needed.",
        "desc": "The most advanced stage of diabetic retinopathy has been detected. Abnormal new blood vessels are growing in the eye, which can cause sudden, severe vision loss or blindness if untreated. Please seek emergency ophthalmology care immediately.",
        "action": "🏥 Go to an eye emergency service or hospital today",
        "clinical": "Neovascularisation of disc or retina. High risk of vitreous haemorrhage and blindness.",
        "severity_label": "Critical",
        "severity_level": 4,
    },
}

SEVERITY_COLORS = ["#00D4AA", "#F5C842", "#FF8C42", "#FF4D6D", "#C840E9"]

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RetinaAI — Eye Health Check",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

/* ── Base Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #0A0F1A;
}
section[data-testid="stSidebar"] { display: none; }
.block-container {
    padding: 2rem 3rem 5rem 3rem;
    max-width: 1200px;
}

/* ── Header ── */
.rtn-header {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    padding: 2rem 0 1.5rem 0;
    border-bottom: 1px solid rgba(0,212,170,0.2);
    margin-bottom: 2rem;
}
.rtn-logo {
    width: 56px; height: 56px;
    background: linear-gradient(135deg, #00D4AA, #0099FF);
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.7rem; flex-shrink: 0;
    box-shadow: 0 0 28px rgba(0,212,170,0.35);
}
.rtn-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem; font-weight: 800;
    color: #F0F4FF;
    letter-spacing: -0.5px; margin: 0; line-height: 1;
}
.rtn-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.92rem;
    color: #A0B0C8;        /* ✅ HIGH CONTRAST - was #00D4AA faint monospace */
    margin: 4px 0 0 0;
}
.rtn-badge {
    margin-left: auto;
    background: rgba(0,212,170,0.12);
    border: 1px solid rgba(0,212,170,0.35);
    color: #00D4AA;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem; letter-spacing: 1.5px;
    padding: 6px 14px; border-radius: 20px;
    text-transform: uppercase;
}

/* ── Section Headers — HIGH CONTRAST FIX ── */
.section-hdr {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;          /* ✅ LARGER - was 0.7rem */
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #C8D6E8;              /* ✅ HIGH CONTRAST - was #2D3748 (nearly invisible) */
    margin: 2.5rem 0 1.2rem 0;
    display: flex; align-items: center; gap: 0.6rem;
}
.section-hdr::after {
    content: ''; flex: 1; height: 1px;
    background: rgba(255,255,255,0.12);  /* ✅ MORE VISIBLE - was 0.05 */
}

/* ── Info Banner ── */
.info-banner {
    background: rgba(0,212,170,0.07);
    border: 1px solid rgba(0,212,170,0.25);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-bottom: 2rem;
    display: flex; gap: 0.8rem; align-items: flex-start;
}
.info-banner-icon { font-size: 1.2rem; flex-shrink: 0; margin-top: 2px; }
.info-banner-text {
    font-size: 0.92rem;
    color: #C0D4E8;              /* ✅ HIGH CONTRAST */
    line-height: 1.6;
}
.info-banner-text strong { color: #E8F4FF; }

/* ── Stat Pills — patient-friendly version ── */
.stat-row {
    display: flex; gap: 0.8rem; flex-wrap: nowrap; margin-bottom: 2rem;
}
.stat-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    text-align: center; flex: 1; min-width: 0;
}
.stat-num {
    font-family: 'DM Mono', monospace;
    font-size: 1.15rem; font-weight: 500; color: #00D4AA;
}
.stat-lbl {
    font-size: 0.75rem;          /* ✅ LARGER - was 0.65rem */
    color: #8BAEC4;              /* ✅ HIGH CONTRAST - was #4A5568 */
    text-transform: uppercase; letter-spacing: 1.5px; margin-top: 3px;
}

/* ── Result Card ── */
.result-card {
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.2rem;
    border-left: 5px solid;
    position: relative; overflow: hidden;
}
.result-grade {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;          /* ✅ SLIGHTLY LARGER */
    letter-spacing: 2px; text-transform: uppercase;
    color: #B0C4D8;              /* ✅ HIGH CONTRAST - was opacity:0.7 on dark */
    margin-bottom: 8px;
}
.result-name {
    font-family: 'Syne', sans-serif;
    font-size: 2rem; font-weight: 800;
    margin: 0 0 6px 0; letter-spacing: -0.5px;
}
.result-urgency {
    font-size: 0.95rem;          /* ✅ LARGER - was 0.78rem */
    font-weight: 600;
    color: #E8F4FF;              /* ✅ HIGH CONTRAST - was opacity:0.75 */
    margin-bottom: 0.4rem;
}
.result-urgency-plain {
    font-size: 0.85rem;
    color: #A8C0D4;
    margin-bottom: 1rem;
    font-style: italic;
}
.result-desc {
    font-size: 0.95rem;          /* ✅ LARGER - was 0.88rem */
    line-height: 1.7;
    color: #D0E0F0;              /* ✅ HIGH CONTRAST - was opacity:0.85 on dark */
    margin-bottom: 1rem;
}
.result-action {
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.88rem;
    color: #E0EEFF;              /* ✅ HIGH CONTRAST */
    font-weight: 500;
    margin-bottom: 0.8rem;
}
.result-clinical {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #7090A8;              /* ✅ IMPROVED from #2D3748 (invisible) - labeled as technical */
    line-height: 1.5;
    border-top: 1px solid rgba(255,255,255,0.1);
    padding-top: 0.8rem; margin-top: 0.8rem;
}
.clinical-label {
    font-size: 0.62rem; letter-spacing: 1.5px; text-transform: uppercase;
    color: #5A7A90; margin-bottom: 4px;
}

/* ── Confidence Bar ── */
.conf-row {
    display: flex; align-items: center; gap: 1rem; margin-top: 1rem;
}
.conf-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;          /* ✅ LARGER - was 0.65rem */
    letter-spacing: 2px; text-transform: uppercase;
    color: #8BAEC4;              /* ✅ HIGH CONTRAST - was #4A5568 */
    flex-shrink: 0;
}
.conf-bar-wrap {
    flex: 1; height: 8px;        /* ✅ TALLER - was 6px */
    background: rgba(255,255,255,0.08);
    border-radius: 10px; overflow: hidden;
}
.conf-bar-fill {
    height: 100%; border-radius: 10px; transition: width 0.6s ease;
}
.conf-value {
    font-family: 'DM Mono', monospace;
    font-size: 0.9rem; font-weight: 500;
    flex-shrink: 0; min-width: 50px; text-align: right;
}
.low-conf-warn {
    background: rgba(245,200,66,0.10);
    border: 1px solid rgba(245,200,66,0.35);
    border-radius: 10px; padding: 0.8rem 1rem;
    font-size: 0.85rem;          /* ✅ LARGER */
    color: #F5E090;              /* ✅ HIGH CONTRAST - was #F5C842 (dim on dark) */
    margin-top: 0.8rem; line-height: 1.5;
}

/* ── Severity Progress Bar ── */
.severity-track {
    background: rgba(255,255,255,0.05);
    border-radius: 50px; height: 10px; overflow: hidden; margin: 0.8rem 0;
    position: relative;
}
.severity-thumb {
    height: 100%; border-radius: 50px;
    background: linear-gradient(90deg, #00D4AA, #F5C842, #FF8C42, #FF4D6D, #C840E9);
    transition: width 0.6s ease;
}

/* ── Class Cards ── */
.class-card {
    border-radius: 12px; padding: 1rem 0.8rem;
    text-align: center; transition: transform 0.2s;
}
.class-card-icon { font-size: 1.2rem; margin-bottom: 5px; }
.class-card-pct {
    font-family: 'DM Mono', monospace;
    font-size: 1.05rem; font-weight: 500;
}
.class-card-name {
    font-size: 0.75rem;          /* ✅ LARGER - was 0.65rem */
    color: #90AAC0;              /* ✅ HIGH CONTRAST - was #4A5568 */
    margin-top: 3px; font-weight: 500;
}
.class-card-grade {
    font-size: 0.65rem;
    color: #6080A0;              /* ✅ IMPROVED - was #2D3748 (invisible) */
    font-family: 'DM Mono', monospace;
}

/* ── Empty State ── */
.empty-state {
    background: rgba(255,255,255,0.025);
    border: 1.5px dashed rgba(255,255,255,0.12);  /* ✅ MORE VISIBLE - was 0.06 */
    border-radius: 16px;
    padding: 4rem 2rem; text-align: center; margin: 1rem 0;
}
.empty-state-icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.6; }
.empty-state-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem; font-weight: 700;
    color: #A0B8CC;              /* ✅ HIGH CONTRAST - was #2D3748 (invisible) */
    margin-bottom: 0.5rem;
}
.empty-state-sub {
    font-size: 0.9rem;           /* ✅ LARGER & plainer - was 0.7rem monospace uppercase */
    color: #6A8EA8;              /* ✅ HIGH CONTRAST - was #1A2535 (nearly invisible) */
    line-height: 1.6;
}

/* ── Grade Reference Cards ── */
.ref-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 12px; padding: 1rem 0.8rem;
    text-align: center; height: 100%;
}
.ref-card-grade {
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    letter-spacing: 1.5px; margin-bottom: 5px;
}
.ref-card-name {
    font-size: 0.85rem; font-weight: 600;
    color: #C0D4E8;              /* ✅ HIGH CONTRAST - was #8892A4 */
    margin-bottom: 8px;
}
.ref-card-urgency {
    font-size: 0.75rem;          /* ✅ LARGER - was 0.68rem */
    color: #8AACC8;              /* ✅ HIGH CONTRAST - was #2D3748 (invisible) */
    line-height: 1.5;
}

/* ── How it Works ── */
.how-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 1.2rem;
    text-align: center;
}
.how-card-num {
    width: 32px; height: 32px;
    background: rgba(0,212,170,0.15); border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'DM Mono', monospace; font-size: 0.85rem;
    color: #00D4AA; font-weight: 500;
    margin: 0 auto 0.8rem auto;
}
.how-card-title {
    font-size: 0.9rem; font-weight: 600;
    color: #D0E0F0;              /* ✅ HIGH CONTRAST */
    margin-bottom: 0.4rem;
}
.how-card-desc {
    font-size: 0.8rem; color: #7A9AB4; line-height: 1.5; /* ✅ IMPROVED */
}

/* ── Footer — THE MOST IMPORTANT VISIBILITY FIX ── */
.rtn-footer {
    margin-top: 3.5rem;
    padding: 1.5rem 1.8rem;
    border-radius: 12px;
    background: rgba(255,200,60,0.06);
    border: 1px solid rgba(255,200,60,0.2);  /* ✅ Warning tone, visible */
}
.footer-disclaimer {
    font-size: 0.82rem;          /* ✅ MUCH LARGER - was 0.62rem */
    color: #D4C080;              /* ✅ HIGH CONTRAST - was #2D3748 (invisible) */
    letter-spacing: 0.3px; line-height: 1.7; margin-bottom: 0.8rem;
}
.footer-disclaimer strong { color: #F0D878; }
.footer-model {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #6A8096;              /* ✅ IMPROVED - was #1A2535 (invisible) */
    text-align: right;
}

/* ── Streamlit overrides ── */
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="stFileUploader"] > div {
    background: rgba(0,212,170,0.04) !important;
    border: 1.5px dashed rgba(0,212,170,0.25) !important;
    border-radius: 14px !important; padding: 1.5rem !important;
}
[data-testid="stFileUploader"] label { color: #8BAEC4 !important; }
div[data-testid="stImage"] img { border-radius: 12px; }
.stButton > button {
    background: linear-gradient(135deg, #00D4AA, #0099FF) !important;
    color: #080C14 !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 500 !important; border: none !important;
    border-radius: 10px !important; letter-spacing: 1px !important;
    font-size: 0.8rem !important; text-transform: uppercase !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Model Loading ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    model = RetinalCNN(num_classes=NUM_CLASSES)
    if not os.path.exists(MODEL_PATH):
        return None
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

# ─── Inference ────────────────────────────────────────────────────────────────

_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@torch.no_grad()
def predict(model, image: Image.Image):
    tensor = _tfm(image).unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
    return probs

# ─── Charts ───────────────────────────────────────────────────────────────────

def make_prob_chart(probs, pred_idx):
    colors   = [CLASS_META[c]["color"] for c in CLASS_NAMES]
    opacities = [1.0 if i == pred_idx else 0.35 for i in range(NUM_CLASSES)]
    bar_colors = [
        f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},{o})"
        for c, o in zip(colors, opacities)
    ]

    fig = go.Figure(go.Bar(
        x=probs * 100,
        y=CLASS_NAMES,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
        textfont=dict(family="DM Mono, monospace", size=12, color="#A0B8CC"),
        hovertemplate="%{y}: <b>%{x:.2f}%</b><extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=70, t=10, b=10), height=240,
        xaxis=dict(
            range=[0, 120], showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            tickfont=dict(family="DM Mono", size=11, color="#8BAEC4"),
            ticksuffix="%", zeroline=False, showline=False,
        ),
        yaxis=dict(
            tickfont=dict(family="DM Sans", size=13, color="#C0D4E8"),  # ✅ HIGH CONTRAST
            showgrid=False, zeroline=False,
        ),
        hoverlabel=dict(
            bgcolor="#0D1421",
            font=dict(family="DM Mono", size=11, color="#F0F4FF"),
            bordercolor="rgba(255,255,255,0.1)",
        ),
        showlegend=False,
    )
    return fig

def make_gauge(conf, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=conf * 100,
        number=dict(suffix="%", font=dict(family="Syne, sans-serif", size=30, color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=0, tickcolor="rgba(0,0,0,0)", visible=False),
            bar=dict(color=color, thickness=0.24),
            bgcolor="rgba(255,255,255,0.04)", borderwidth=0,
            steps=[
                dict(range=[0, 60],  color="rgba(255,77,109,0.06)"),
                dict(range=[60, 80], color="rgba(245,200,66,0.06)"),
                dict(range=[80,100], color="rgba(0,212,170,0.06)"),
            ],
        ),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=20, b=10), height=170,
        font=dict(family="DM Sans"),
    )
    return fig

# ─── Header ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="rtn-header">
    <div class="rtn-logo">👁</div>
    <div>
        <p class="rtn-title">RetinaAI</p>
        <p class="rtn-subtitle">Diabetic Retinopathy Detection System</p>
    </div>
    <div class="rtn-badge">EfficientNet-B3 · APTOS 2019</div>
</div>
""", unsafe_allow_html=True)

# ─── Model Init ──────────────────────────────────────────────────────────────

with st.spinner("Loading AI model..."):
    model = load_model()

if model is None:
    st.markdown("""
    <div style="background:rgba(255,77,109,0.10);border:1px solid rgba(255,77,109,0.4);
    border-radius:12px;padding:1.4rem 1.6rem;margin:1rem 0;">
        <div style="color:#FF8090;font-family:'DM Mono',monospace;font-size:0.85rem;
        letter-spacing:1px;margin-bottom:6px;">⚠ MODEL NOT FOUND</div>
        <div style="color:#C0D0E0;font-size:0.88rem;line-height:1.6;">
        Please run <code style="background:rgba(255,255,255,0.08);padding:2px 8px;
        border-radius:4px;color:#00D4AA;">python train.py</code> in the 
        <code style="background:rgba(255,255,255,0.08);padding:2px 8px;
        border-radius:4px;color:#00D4AA;">model_training/</code> folder first.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── Info Banner (patient-friendly intro) ─────────────────────────────────────

st.markdown("""
<div class="info-banner">
    <div class="info-banner-icon">ℹ️</div>
    <div class="info-banner-text">
        <strong>How this works:</strong> Upload a retinal fundus photo (an image of the back of your eye). 
        Our AI analyses it for signs of diabetic retinopathy — a diabetes-related eye condition. 
        Results appear instantly. <strong>This tool is for screening only — always confirm results with your doctor.</strong>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Stats Row (patient-friendly labels) ─────────────────────────────────────

st.markdown("""
<div class="stat-row">
    <div class="stat-pill">
        <div class="stat-num">72.2%</div>
        <div class="stat-lbl">AI Accuracy</div>
    </div>
    <div class="stat-pill">
        <div class="stat-num">3,662</div>
        <div class="stat-lbl">Training Images</div>
    </div>
    <div class="stat-pill">
        <div class="stat-num">5</div>
        <div class="stat-lbl">Severity Levels</div>
    </div>
    <div class="stat-pill">
        <div class="stat-num">96%</div>
        <div class="stat-lbl">Healthy Eyes Confirmed</div>
    </div>
    <div class="stat-pill">
        <div class="stat-num">93%</div>
        <div class="stat-lbl">No False Alarms</div>
    </div>
    <div class="stat-pill">
        <div class="stat-num">EfficientNet</div>
        <div class="stat-lbl">AI Model Used</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Upload ──────────────────────────────────────────────────────────────────

st.markdown('<p class="section-hdr">Upload Your Retinal Image</p>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop a retinal fundus photograph (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

# ─── Inference & Results ─────────────────────────────────────────────────────

if uploaded:
    image    = Image.open(uploaded).convert("RGB")
    probs    = predict(model, image)
    pred_idx = int(probs.argmax())
    pred_cls = CLASS_NAMES[pred_idx]
    meta     = CLASS_META[pred_cls]
    conf     = float(probs[pred_idx])
    low_conf = conf < 0.60

    st.markdown('<p class="section-hdr">Analysis Result</p>', unsafe_allow_html=True)

    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        st.markdown(f"""
        <img src="data:image/png;base64,{b64}"
             style="width:100%;height:100%;min-height:460px;
                    object-fit:cover;border-radius:16px;display:block;" />
        """, unsafe_allow_html=True)

    with col_result:
        color = meta["color"]

        st.markdown(f"""
        <div class="result-card" style="
            background:{meta['bg']};
            border-left-color:{color};
            color:#F0F4FF;
        ">
            <div class="result-grade" style="color:{color};">
                {meta['icon']} &nbsp; {meta['grade']} · Diabetic Retinopathy
            </div>
            <div class="result-name" style="color:{color};">{pred_cls}</div>
            <div class="result-urgency">{meta['urgency']}</div>
            <div class="result-urgency-plain">{meta['urgency_plain']}</div>
            <div class="result-desc">{meta['desc']}</div>
            <div class="result-action">{meta['action']}</div>
            <div class="clinical-label">Technical Finding (for doctors)</div>
            <div class="result-clinical">{meta['clinical']}</div>
        </div>
        """, unsafe_allow_html=True)

        conf_pct = conf * 100
        st.markdown(f"""
        <div class="conf-row">
            <span class="conf-label">AI Confidence</span>
            <div class="conf-bar-wrap">
                <div class="conf-bar-fill" style="
                    width:{conf_pct:.1f}%;
                    background:linear-gradient(90deg,{color}99,{color});
                "></div>
            </div>
            <span class="conf-value" style="color:{color};">{conf_pct:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

        if low_conf:
            st.markdown("""
            <div class="low-conf-warn">
                ⚠ The AI is not very certain about this result (confidence below 60%). 
                This may be a borderline case. Please consult a doctor regardless of this reading.
            </div>
            """, unsafe_allow_html=True)

    # ── Severity Visual ──────────────────────────────────────────────────────
    st.markdown('<p class="section-hdr">Severity Scale</p>', unsafe_allow_html=True)
    level = meta["severity_level"]
    track_pct = (level / 4) * 100 if level > 0 else 5

    severity_labels_html = "".join([
        f'<span style="font-size:0.72rem;color:{"#E0F0FF" if i==level else "#5A7A90"};'
        f'font-weight:{"600" if i==level else "400"};text-align:center;flex:1;">'
        f'{"→ " if i==level else ""}{CLASS_META[CLASS_NAMES[i]]["severity_label"]}</span>'
        for i in range(5)
    ])

    st.markdown(f"""
    <div style="margin-bottom:1.5rem;">
        <div class="severity-track">
            <div class="severity-thumb" style="width:{track_pct}%;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:6px;">
            {severity_labels_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Probability Chart ─────────────────────────────────────────────────────
    st.markdown('<p class="section-hdr">Probability Breakdown</p>', unsafe_allow_html=True)

    col_bar, col_gauge = st.columns([2, 1], gap="large")

    with col_bar:
        st.plotly_chart(
            make_prob_chart(probs, pred_idx),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with col_gauge:
        st.markdown("""
        <div style="text-align:center;padding-top:0.3rem;">
            <div style="font-size:0.8rem;color:#8BAEC4;letter-spacing:1.5px;
            text-transform:uppercase;margin-bottom:0.2rem;">AI Confidence</div>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(
            make_gauge(conf, meta["color"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # ── All Classes ───────────────────────────────────────────────────────────
    st.markdown('<p class="section-hdr">All Severity Scores</p>', unsafe_allow_html=True)

    cols = st.columns(5)
    for i, (col, cls) in enumerate(zip(cols, CLASS_NAMES)):
        m = CLASS_META[cls]
        c = m["color"]
        p = probs[i] * 100
        is_pred = (i == pred_idx)
        border_style = f"1.5px solid {c}" if is_pred else "1px solid rgba(255,255,255,0.08)"
        bg_style = m["bg"] if is_pred else "rgba(255,255,255,0.025)"
        with col:
            st.markdown(f"""
            <div class="class-card" style="background:{bg_style};border:{border_style};">
                <div class="class-card-icon">{m['icon']}</div>
                <div class="class-card-pct" style="color:{c};">{p:.1f}%</div>
                <div class="class-card-name">{cls}</div>
                <div class="class-card-grade">{m['grade']}</div>
            </div>
            """, unsafe_allow_html=True)

else:
    # ── Empty State ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">👁</div>
        <div class="empty-state-title">No Image Uploaded Yet</div>
        <div class="empty-state-sub">
            Upload a retinal fundus photograph above to begin your eye health analysis.<br>
            Supported formats: JPG, JPEG, PNG · Max size: 200MB
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── How It Works ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-hdr">How It Works</p>', unsafe_allow_html=True)
    hw_cols = st.columns(3)
    steps = [
        ("1", "Upload Your Image", "Drag and drop a retinal fundus photograph taken by your eye doctor or clinic."),
        ("2", "AI Analysis", "Our AI model, trained on thousands of retinal images, analyses yours in seconds."),
        ("3", "Read Your Result", "You receive a clear result showing severity level and recommended next steps."),
    ]
    for col, (num, title, desc) in zip(hw_cols, steps):
        with col:
            st.markdown(f"""
            <div class="how-card">
                <div class="how-card-num">{num}</div>
                <div class="how-card-title">{title}</div>
                <div class="how-card-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Grade Reference ───────────────────────────────────────────────────────
    st.markdown('<p class="section-hdr">DR Severity Reference Guide</p>', unsafe_allow_html=True)
    ref_cols = st.columns(5)
    for col, cls in zip(ref_cols, CLASS_NAMES):
        m = CLASS_META[cls]
        with col:
            st.markdown(f"""
            <div class="ref-card">
                <div style="font-size:1.4rem;margin-bottom:8px;">{m['icon']}</div>
                <div class="ref-card-grade" style="color:{m['color']};">{m['grade']}</div>
                <div class="ref-card-name">{cls}</div>
                <div class="ref-card-urgency">{m['urgency']}</div>
            </div>
            """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="rtn-footer">
    <div class="footer-disclaimer">
        <strong>⚠ IMPORTANT — FOR RESEARCH & EDUCATIONAL USE ONLY</strong><br>
        This system is <strong>not a certified medical device</strong> and must not replace a professional eye examination. 
        All results must be reviewed by a qualified ophthalmologist or clinician before any medical decision is made. 
        If you are experiencing vision problems, please contact your doctor immediately.
    </div>
    <div class="footer-model">
        EfficientNet-B3 · APTOS 2019<br>
        Fine-tuned · 5-class DR Grading
    </div>
</div>
""", unsafe_allow_html=True)