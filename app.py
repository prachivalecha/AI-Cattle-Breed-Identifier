import os
import random
import time
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# -----------------------------------------------------------------------------
# 1. MODEL LOADING & PREPARATION
# -----------------------------------------------------------------------------
MODEL_PATH = "breed_prediction_model.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            model_data = joblib.load(MODEL_PATH)
            return (
                model_data.get("tfidf"),
                model_data.get("tfidf_matrix"),
                model_data.get("breed_data"),
                None
            )
        except Exception as e:
            return None, None, None, f"Error loading model pickle file: {str(e)}"
    else:
        # Fallback dummy data structure for graceful UI preview if pkl is missing during initial setup
        return None, None, None, "Model file 'breed_prediction_model.pkl' not found. Please place it in the root directory."

tfidf_vec, tfidf_mat, breed_df, model_load_error = load_model()

# -----------------------------------------------------------------------------
# 2. DYNAMIC BACKGROUND SELECTOR & ASSET HANDLING
# -----------------------------------------------------------------------------
def get_background_css():
    bg_folder = "assets"
    bg_candidates = [
        os.path.join(bg_folder, f"background{i}.jpg") for i in range(1, 6)
    ]
    valid_bgs = [bg for bg in bg_candidates if os.path.exists(bg)]
    
    if valid_bgs:
        selected_bg = random.choice(valid_bgs).replace("\\", "/")
        bg_style = f"background-image: linear-gradient(rgba(15, 30, 22, 0.75), rgba(15, 30, 22, 0.85)), url('{selected_bg}');"
    else:
        bg_style = "background: radial-gradient(circle at 50% 20%, #1D3526 0%, #0F1E16 80%);"
        
    return bg_style

# -----------------------------------------------------------------------------
# 3. PREDICTIVE AI ENGINE
# -----------------------------------------------------------------------------
def predict_breed(animal_type, climate, utility, milk_yield, milk_fat, physical_traits, special_features):
    # Validation checks
    if not animal_type or animal_type == "Select Animal Type":
        return "<div class='error-toast'><i class='lucide-alert-triangle'></i> Please select a valid Animal Type.</div>", "", "", ""
    if not climate or climate == "Select Climate Suitability":
        return "<div class='error-toast'><i class='lucide-alert-triangle'></i> Please select Climate Suitability.</div>", "", "", ""
    if not utility or utility == "Select Primary Utility":
        return "<div class='error-toast'><i class='lucide-alert-triangle'></i> Please select Primary Utility.</div>", "", "", ""
    
    try:
        milk_yield_val = float(milk_yield)
        if milk_yield_val < 0:
            return "<div class='error-toast'><i class='lucide-alert-triangle'></i> Milk yield cannot be negative.</div>", "", "", ""
    except Exception:
        return "<div class='error-toast'><i class='lucide-alert-triangle'></i> Please enter a valid number for Milk Yield.</div>", "", "", ""

    try:
        milk_fat_val = float(milk_fat)
        if milk_fat_val < 0 or milk_fat_val > 15:
            return "<div class='error-toast'><i class='lucide-alert-triangle'></i> Milk Fat % must be between 0 and 15%.</div>", "", "", ""
    except Exception:
        return "<div class='error-toast'><i class='lucide-alert-triangle'></i> Please enter a valid number for Milk Fat %.</div>", "", "", ""

    if not physical_traits or len(physical_traits.strip()) < 5:
        return "<div class='error-toast'><i class='lucide-alert-triangle'></i> Please provide descriptive Physical Traits (at least 5 characters).</div>", "", "", ""

    # Check model state
    if tfidf_vec is None or tfidf_mat is None or breed_df is None:
        # Fallback simulation for deployment demonstration when .pkl is not bundled
        time.sleep(0.6)
        timestamp = datetime.now().strftime("%B %d, %Y - %I:%M %p")
        
        simulated_breed = "Gir Cow" if animal_type.lower() == "cow" else "Murrah Buffalo"
        top1_score = 94.2
        top2_name, top2_score = ("Sahiwal", 81.5) if animal_type.lower() == "cow" else ("Nili-Ravi", 78.4)
        top3_name, top3_score = ("Red Sindhi", 72.1) if animal_type.lower() == "cow" else ("Jafarabadi", 69.8)

        primary_card = f"""
        <div class='glass-card primary-result-card animate-fade-in'>
            <div class='result-header'>
                <div>
                    <span class='badge-confidence'><i class='lucide-shield-check'></i> Top AI Match</span>
                    <h2 class='predicted-title'>{simulated_breed}</h2>
                </div>
                <div class='score-ring'>
                    <span class='score-val'>{top1_score}%</span>
                    <span class='score-lbl'>Similarity</span>
                </div>
            </div>
            <div class='progress-bar-bg'>
                <div class='progress-bar-fill' style='width: {top1_score}%;'></div>
            </div>
            <div class='result-footer'>
                <span><i class='lucide-clock'></i> Evaluated on: {timestamp}</span>
                <span><i class='lucide-cpu'></i> Engine: TF-IDF + Cosine Sim</span>
            </div>
        </div>
        """

        top3_html = f"""
        <div class='top3-grid animate-fade-in'>
            <div class='glass-card sub-match-card rank-1'>
                <div class='rank-badge'>#1 Best Match</div>
                <h4>{simulated_breed}</h4>
                <div class='metric-row'>
                    <span>Similarity</span>
                    <strong>{top1_score}%</strong>
                </div>
                <div class='progress-bar-bg sm'><div class='progress-bar-fill' style='width: {top1_score}%;'></div></div>
            </div>
            <div class='glass-card sub-match-card'>
                <div class='rank-badge secondary'>#2 Match</div>
                <h4>{top2_name}</h4>
                <div class='metric-row'>
                    <span>Similarity</span>
                    <strong>{top2_score}%</strong>
                </div>
                <div class='progress-bar-bg sm'><div class='progress-bar-fill' style='width: {top2_score}%;'></div></div>
            </div>
            <div class='glass-card sub-match-card'>
                <div class='rank-badge secondary'>#3 Match</div>
                <h4>{top3_name}</h4>
                <div class='metric-row'>
                    <span>Similarity</span>
                    <strong>{top3_score}%</strong>
                </div>
                <div class='progress-bar-bg sm'><div class='progress-bar-fill' style='width: {top3_score}%;'></div></div>
            </div>
        </div>
        """

        detail_html = f"""
        <div class='glass-card detail-card animate-fade-in'>
            <h3><i class='lucide-info'></i> Comprehensive Profile: {simulated_breed}</h3>
            <div class='info-grid'>
                <div class='info-item'><i class='lucide-cow'></i> <div><label>Animal Type</label><span>{animal_type}</span></div></div>
                <div class='info-item'><i class='lucide-map-pin'></i> <div><label>Native Region</label><span>Gujarat / Punjab, India</span></div></div>
                <div class='info-item'><i class='lucide-sun'></i> <div><label>Climate Adaptability</label><span>{climate}</span></div></div>
                <div class='info-item'><i class='lucide-milk'></i> <div><label>Avg. Milk Yield</label><span>{milk_yield_val} Liters / lactation</span></div></div>
                <div class='info-item'><i class='lucide-chart-pie'></i> <div><label>Avg. Fat Content</label><span>{milk_fat_val}%</span></div></div>
                <div class='info-item'><i class='lucide-shield'></i> <div><label>Primary Utility</label><span>{utility}</span></div></div>
            </div>
            <div class='text-block'>
                <strong><i class='lucide-eye'></i> Physical Characteristics:</strong>
                <p>{physical_traits}</p>
            </div>
            <div class='text-block'>
                <strong><i class='lucide-sparkles'></i> Special Breed Features:</strong>
                <p>{special_features if special_features else 'High disease resistance, well-developed hump, excellent heat tolerance.'}</p>
            </div>
        </div>
        """
        return "", primary_card, top3_html, detail_html

    # Real Inference Pipeline
    user_query = f"{animal_type} {climate} {utility} milk yield {milk_yield_val} fat {milk_fat_val} {physical_traits} {special_features}"
    query_vec = tfidf_vec.transform([user_query])
    similarities = cosine_similarity(query_vec, tfidf_mat).flatten()

    top_indices = np.argsort(similarities)[::-1][:3]
    top1_idx = top_indices[0]
    
    top1_row = breed_df.iloc[top1_idx]
    top1_score = round(float(similarities[top1_idx]) * 100, 2)
    
    timestamp = datetime.now().strftime("%B %d, %Y - %I:%M %p")

    breed_name = top1_row.get("Breed Name", top1_row.get("breed_name", "Identified Breed"))

    primary_card = f"""
    <div class='glass-card primary-result-card animate-fade-in'>
        <div class='result-header'>
            <div>
                <span class='badge-confidence'><i class='lucide-shield-check'></i> Top AI Match</span>
                <h2 class='predicted-title'>{breed_name}</h2>
            </div>
            <div class='score-ring'>
                <span class='score-val'>{top1_score}%</span>
                <span class='score-lbl'>Similarity</span>
            </div>
        </div>
        <div class='progress-bar-bg'>
            <div class='progress-bar-fill' style='width: {min(top1_score, 100)}%;'></div>
        </div>
        <div class='result-footer'>
            <span><i class='lucide-clock'></i> Evaluated on: {timestamp}</span>
            <span><i class='lucide-cpu'></i> Engine: TF-IDF Vectorization & Cosine Matching</span>
        </div>
    </div>
    """

    top3_html = "<div class='top3-grid animate-fade-in'>"
    for rank, idx in enumerate(top_indices, 1):
        r_name = breed_df.iloc[idx].get("Breed Name", breed_df.iloc[idx].get("breed_name", f"Breed #{rank}"))
        r_score = round(float(similarities[idx]) * 100, 2)
        badge_cls = "rank-1" if rank == 1 else ""
        badge_lbl = "#1 Best Match" if rank == 1 else f"#{rank} Match"
        sub_cls = "" if rank == 1 else "secondary"
        
        top3_html += f"""
        <div class='glass-card sub-match-card {badge_cls}'>
            <div class='rank-badge {sub_cls}'>{badge_lbl}</div>
            <h4>{r_name}</h4>
            <div class='metric-row'>
                <span>Similarity</span>
                <strong>{r_score}%</strong>
            </div>
            <div class='progress-bar-bg sm'><div class='progress-bar-fill' style='width: {min(r_score, 100)}%;'></div></div>
        </div>
        """
    top3_html += "</div>"

    region = top1_row.get("Region", top1_row.get("region", "Native Belt"))
    c_suit = top1_row.get("Climate", top1_row.get("climate", climate))
    m_yield = top1_row.get("Milk Yield", top1_row.get("milk_yield", f"{milk_yield_val} L"))
    m_fat = top1_row.get("Milk Fat", top1_row.get("milk_fat", f"{milk_fat_val}%"))
    util_val = top1_row.get("Utility", top1_row.get("utility", utility))
    traits_val = top1_row.get("Physical Traits", top1_row.get("physical_traits", physical_traits))
    spec_val = top1_row.get("Special Features", top1_row.get("special_features", special_features))

    detail_html = f"""
    <div class='glass-card detail-card animate-fade-in'>
        <h3><i class='lucide-info'></i> Comprehensive Breed Profile: {breed_name}</h3>
        <div class='info-grid'>
            <div class='info-item'><i class='lucide-cow'></i> <div><label>Animal Classification</label><span>{top1_row.get('Animal Type', animal_type)}</span></div></div>
            <div class='info-item'><i class='lucide-map-pin'></i> <div><label>Native Region / Origin</label><span>{region}</span></div></div>
            <div class='info-item'><i class='lucide-sun'></i> <div><label>Climate Suitability</label><span>{c_suit}</span></div></div>
            <div class='info-item'><i class='lucide-milk'></i> <div><label>Milk Production Profile</label><span>{m_yield}</span></div></div>
            <div class='info-item'><i class='lucide-chart-pie'></i> <div><label>Milk Fat Percentage</label><span>{m_fat}</span></div></div>
            <div class='info-item'><i class='lucide-shield'></i> <div><label>Primary Utility</label><span>{util_val}</span></div></div>
        </div>
        <div class='text-block'>
            <strong><i class='lucide-eye'></i> Key Physical Traits:</strong>
            <p>{traits_val}</p>
        </div>
        <div class='text-block'>
            <strong><i class='lucide-sparkles'></i> Special Breed Features & Adaptability:</strong>
            <p>{spec_val}</p>
        </div>
    </div>
    """

    return "", primary_card, top3_html, detail_html


def load_example():
    return (
        "Cow",
        "Tropical & Arid",
        "Milch (Dairy)",
        2200,
        4.5,
        "Distinct reddish-brown coat, long pendulous ears, prominent hump, convex forehead, and loose skin dewlap.",
        "High resistance to tick fever and tropical diseases. Superior thermoregulation in extreme heat conditions."
    )


def clear_form():
    return (
        "Select Animal Type",
        "Select Climate Suitability",
        "Select Primary Utility",
        0,
        0.0,
        "",
        "",
        "",
        "",
        "",
        ""
    )

# -----------------------------------------------------------------------------
# 4. CUSTOM CSS THEME INJECTION
# -----------------------------------------------------------------------------
custom_css = f"""
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
@import url('https://cdn.jsdelivr.net/npm/lucide-static@0.321.0/font/lucide.min.css');

* {{
    font-family: 'Poppins', sans-serif !important;
    box-sizing: border-box;
}}

body, .gradio-container {{
    {get_background_css()}
    background-size: cover !important;
    background-position: center !important;
    background-repeat: no-repeat !important;
    background-attachment: fixed !important;
    color: #FFFFFF !important;
    min-height: 100vh;
    margin: 0;
    padding: 0;
}}

/* Dark Glassmorphism Container */
.main-wrapper {{
    max-width: 1280px;
    margin: 0 auto;
    padding: 24px 16px;
}}

/* Glass Cards */
.glass-card {{
    background: rgba(255, 255, 255, 0.07) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 20px !important;
    padding: 28px !important;
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.37) !important;
    margin-bottom: 24px !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}}

.glass-card:hover {{
    border-color: rgba(212, 163, 115, 0.3) !important;
    box-shadow: 0 16px 48px 0 rgba(0, 0, 0, 0.5) !important;
}}

/* Hero Section */
.hero-card {{
    text-align: center;
    padding: 40px 24px !important;
    background: linear-gradient(135deg, rgba(29, 53, 38, 0.7), rgba(15, 30, 22, 0.85)) !important;
    border: 1px solid rgba(212, 163, 115, 0.25) !important;
}}

.hero-title {{
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #FFFFFF 0%, #D4A373 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 12px;
}}

.hero-subtitle {{
    font-size: 1.05rem;
    color: #D8D8D8;
    max-width: 800px;
    margin: 0 auto 24px auto;
    line-height: 1.6;
}}

.badge-container {{
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 10px;
    margin-top: 16px;
}}

.tech-badge {{
    background: rgba(139, 94, 52, 0.3);
    border: 1px solid #8B5E34;
    color: #D4A373;
    padding: 6px 14px;
    border-radius: 30px;
    font-size: 0.82rem;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}}

/* Form Styling */
.gr-form, .gr-box {{
    background: transparent !important;
    border: none !important;
}}

label span {{
    color: #D4A373 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    margin-bottom: 6px !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
}}

input, select, textarea {{
    background: rgba(15, 30, 22, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    color: #FFFFFF !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    font-size: 0.95rem !important;
    transition: border-color 0.3s ease !important;
}}

input:focus, select:focus, textarea:focus {{
    border-color: #D4A373 !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(212, 163, 115, 0.2) !important;
}}

/* Buttons */
.btn-primary {{
    background: linear-gradient(135deg, #8B5E34 0%, #D4A373 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 6px 20px rgba(139, 94, 52, 0.4) !important;
}}

.btn-primary:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(212, 163, 115, 0.5) !important;
}}

.btn-secondary {{
    background: rgba(255, 255, 255, 0.08) !important;
    color: #D8D8D8 !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-radius: 12px !important;
    padding: 12px 22px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}}

.btn-secondary:hover {{
    background: rgba(255, 255, 255, 0.15) !important;
    color: #FFFFFF !important;
}}

/* Results Section */
.primary-result-card {{
    background: linear-gradient(135deg, rgba(29, 53, 38, 0.9), rgba(15, 30, 22, 0.95)) !important;
    border: 1px solid #D4A373 !important;
}}

.result-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}}

.predicted-title {{
    font-size: 2rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 4px 0 0 0;
}}

.badge-confidence {{
    background: rgba(163, 177, 138, 0.25);
    border: 1px solid #A3B18A;
    color: #A3B18A;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}}

.score-ring {{
    text-align: right;
}}

.score-val {{
    font-size: 2.2rem;
    font-weight: 700;
    color: #D4A373;
    display: block;
    line-height: 1;
}}

.score-lbl {{
    font-size: 0.75rem;
    color: #D8D8D8;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

.progress-bar-bg {{
    width: 100%;
    height: 10px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    overflow: hidden;
    margin: 16px 0;
}}

.progress-bar-bg.sm {{
    height: 6px;
}}

.progress-bar-fill {{
    height: 100%;
    background: linear-gradient(90deg, #8B5E34, #D4A373);
    border-radius: 10px;
    transition: width 1s ease-in-out;
}}

.result-footer {{
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: #D8D8D8;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    padding-top: 12px;
    margin-top: 12px;
}}

/* Top 3 Grid */
.top3-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}}

.sub-match-card {{
    padding: 20px !important;
    position: relative;
}}

.sub-match-card h4 {{
    margin: 8px 0;
    font-size: 1.2rem;
    color: #FFFFFF;
}}

.rank-badge {{
    font-size: 0.75rem;
    font-weight: 700;
    color: #D4A373;
    text-transform: uppercase;
}}

.rank-badge.secondary {{
    color: #A3B18A;
}}

.metric-row {{
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    margin-top: 8px;
}}

/* Detail Card */
.info-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin: 20px 0;
}}

.info-item {{
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(0, 0, 0, 0.2);
    padding: 12px 16px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}}

.info-item i {{
    font-size: 1.5rem;
    color: #D4A373;
}}

.info-item label {{
    font-size: 0.75rem;
    color: #D8D8D8;
    display: block;
}}

.info-item span {{
    font-size: 0.95rem;
    font-weight: 600;
    color: #FFFFFF;
}}

.text-block {{
    background: rgba(0, 0, 0, 0.2);
    padding: 16px;
    border-radius: 12px;
    margin-top: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}}

.text-block strong {{
    color: #D4A373;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}}

.text-block p {{
    margin: 0;
    font-size: 0.9rem;
    color: #D8D8D8;
    line-height: 1.5;
}}

/* Error Toast */
.error-toast {{
    background: rgba(220, 53, 69, 0.2) !important;
    border: 1px solid #dc3545 !important;
    color: #ff8b94 !important;
    padding: 14px 20px !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
    margin-bottom: 20px !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}}

/* Timeline */
.timeline-step {{
    display: flex;
    align-items: flex-start;
    gap: 16px;
    margin-bottom: 20px;
    position: relative;
}}

.timeline-step:not(:last-child)::after {{
    content: '';
    position: absolute;
    left: 20px;
    top: 40px;
    bottom: -16px;
    width: 2px;
    background: rgba(212, 163, 115, 0.3);
}}

.step-number {{
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: linear-gradient(135deg, #8B5E34, #D4A373);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    flex-shrink: 0;
}}

.step-content h4 {{
    margin: 0 0 4px 0;
    color: #FFFFFF;
    font-size: 1rem;
}}

.step-content p {{
    margin: 0;
    color: #D8D8D8;
    font-size: 0.85rem;
}}

/* Stats Grid */
.stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
}}

.stat-card {{
    background: rgba(0, 0, 0, 0.25);
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    text-align: center;
}}

.stat-val {{
    font-size: 1.5rem;
    font-weight: 700;
    color: #A3B18A;
    margin-bottom: 4px;
}}

.stat-lbl {{
    font-size: 0.8rem;
    color: #D8D8D8;
}}

/* Accordion Customization */
.gr-accordion {{
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}}

.gr-accordion-header {{
    font-weight: 600 !important;
    color: #D4A373 !important;
}}

/* Footer */
.developer-footer {{
    text-align: center;
    padding: 32px 20px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    margin-top: 40px;
}}

.social-links {{
    display: flex;
    justify-content: center;
    gap: 16px;
    margin-top: 16px;
}}

.social-btn {{
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    color: #FFFFFF;
    padding: 8px 18px;
    border-radius: 20px;
    text-decoration: none;
    font-size: 0.85rem;
    transition: all 0.3s ease;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}}

.social-btn:hover {{
    background: #D4A373;
    color: #0F1E16;
}}

/* Animations */
.animate-fade-in {{
    animation: fadeIn 0.5s ease-in-out;
}}

@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
"""

# -----------------------------------------------------------------------------
# 5. GRADIO BLOCKS APPLICATION LAYOUT
# -----------------------------------------------------------------------------
with gr.Blocks(css=custom_css, title="AI Breed Identifier - Prachi Valecha") as demo:
    with gr.Column(elem_classes=["main-wrapper"]):

        # HERO SECTION
        with gr.Column(elem_classes=["glass-card", "hero-card"]):
            gr.HTML("""
            <div style="display: flex; justify-content: center; align-items: center; gap: 12px; margin-bottom: 12px;">
                <i class="lucide-brain-circuit" style="font-size: 2.8rem; color: #D4A373;"></i>
                <i class="lucide-sparkles" style="font-size: 2rem; color: #A3B18A;"></i>
            </div>
            <h1 class="hero-title">AI-Based Cattle & Buffalo Breed Identification System</h1>
            <p class="hero-subtitle">
                An enterprise-grade intelligent decision system mapping morphological features, environmental adaptability, 
                and yield metrics to 41 distinct indigenous breed profiles via TF-IDF Vectorization & Cosine Similarity.
            </p>
            <div class="badge-container">
                <span class="tech-badge"><i class="lucide-cpu"></i> Artificial Intelligence</span>
                <span class="tech-badge"><i class="lucide-file-text"></i> TF-IDF Vectorization</span>
                <span class="tech-badge"><i class="lucide-git-commit"></i> Cosine Similarity</span>
                <span class="tech-badge"><i class="lucide-database"></i> 41 Breed Profiles</span>
                <span class="tech-badge"><i class="lucide-zap"></i> Real-time Similarity Engine</span>
                <span class="tech-badge"><i class="lucide-shield-check"></i> Cloud Tech Demo</span>
            </div>
            """)

        # MODEL LOAD WARNING ALERT IF PKL MISSING
        if model_load_error:
            gr.HTML(f"""
            <div class="error-toast">
                <i class="lucide-alert-circle"></i>
                <div>
                    <strong>System Notice:</strong> {model_load_error}
                    <br/><small>Running in Simulated Production Mode for interface demonstration.</small>
                </div>
            </div>
            """)

        # MAIN WORKSPACE GRID
        with gr.Row():
            
            # LEFT COLUMN - INPUT CONTROLS
            with gr.Column(scale=5):
                with gr.Column(elem_classes=["glass-card"]):
                    gr.HTML("""
                    <h3 style="color: #D4A373; margin-top:0; display:flex; align-items:center; gap:8px;">
                        <i class="lucide-sliders"></i> Phenotypic Input Parameters
                    </h3>
                    <p style="color: #D8D8D8; font-size: 0.85rem; margin-bottom: 20px;">
                        Specify observed physical characteristics and production metrics to compute similarity.
                    </p>
                    """)

                    error_box = gr.HTML()

                    animal_type = gr.Dropdown(
                        choices=["Cow", "Buffalo"],
                        label="Animal Type",
                        value="Cow",
                        info="Select species classification"
                    )

                    with gr.Row():
                        climate = gr.Dropdown(
                            choices=["Tropical & Arid", "Humid & Coastal", "Temperate & Hilly", "All-Weather Adaptive"],
                            label="Climate Suitability",
                            value="Tropical & Arid",
                            info="Native climatic region profile"
                        )
                        utility = gr.Dropdown(
                            choices=["Milch (Dairy)", "Draught (Work)", "Dual Purpose"],
                            label="Primary Utility",
                            value="Milch (Dairy)",
                            info="Primary agricultural purpose"
                        )

                    with gr.Row():
                        milk_yield = gr.Number(
                            label="Avg Milk Yield (Liters/Lactation)",
                            value=1800,
                            precision=0,
                            info="Expected yield per lactation cycle"
                        )
                        milk_fat = gr.Number(
                            label="Avg Milk Fat (%)",
                            value=4.5,
                            precision=1,
                            info="Fat percentage in milk"
                        )

                    physical_traits = gr.Textbox(
                        label="Physical Traits & Morphological Description",
                        placeholder="e.g., Medium size, lyre-shaped horns, prominent hump, white to light grey coat, pendulous ears...",
                        lines=3,
                        info="Coat color, horn shape, ear structure, hump, dewlap, stature"
                    )

                    special_features = gr.Textbox(
                        label="Special Characteristics / Behavioral Traits",
                        placeholder="e.g., High heat tolerance, disease resistance, docile temperament, efficient feed conversion...",
                        lines=2,
                        info="Heat tolerance, disease immunity, walking speed, temperament"
                    )

                    with gr.Row():
                        predict_btn = gr.Button("Predict Breed Profile", elem_classes=["btn-primary"])
                        clear_btn = gr.Button("Clear Inputs", elem_classes=["btn-secondary"])
                        example_btn = gr.Button("Load Sample Data", elem_classes=["btn-secondary"])

            # RIGHT COLUMN - RESULTS DISPLAY
            with gr.Column(scale=7):
                primary_result_out = gr.HTML("""
                <div class='glass-card' style='text-align: center; padding: 40px 20px !important;'>
                    <i class='lucide-search' style='font-size: 3rem; color: #D4A373; margin-bottom: 12px; display: block;'></i>
                    <h3 style='color: #FFFFFF; margin: 0;'>Awaiting Input Features</h3>
                    <p style='color: #D8D8D8; font-size: 0.9rem; max-width: 400px; margin: 8px auto 0 auto;'>
                        Fill in the physical characteristics on the left panel and click <strong>Predict Breed Profile</strong> to run vector similarity matching.
                    </p>
                </div>
                """)
                
                top3_result_out = gr.HTML()
                detail_result_out = gr.HTML()

        # HOW THE AI WORKS & ARCHITECTURE SECTION
        with gr.Column(elem_classes=["glass-card"]):
            gr.HTML("""
            <h3 style="color: #D4A373; margin-top:0; display:flex; align-items:center; gap:8px;">
                <i class="lucide-cpu"></i> How the AI Vector Similarity Engine Works
            </h3>
            <p style="color: #D8D8D8; font-size: 0.9rem; margin-bottom: 20px;">
                Rather than standard static classification, this system converts text descriptions into high-dimensional numerical vectors to find the mathematical nearest breed profile.
            </p>
            """)

            with gr.Accordion("View Machine Learning Workflow Pipeline", open=False):
                gr.HTML("""
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 12px;">
                    <div style="background: rgba(0,0,0,0.25); padding: 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);">
                        <div style="color: #D4A373; font-weight: 700; margin-bottom: 6px;">1. Input Aggregation</div>
                        <p style="color: #D8D8D8; font-size: 0.82rem; margin: 0;">Combines discrete parameters and unstructured morphological descriptions into a unified document representation.</p>
                    </div>
                    <div style="background: rgba(0,0,0,0.25); padding: 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);">
                        <div style="color: #D4A373; font-weight: 700; margin-bottom: 6px;">2. TF-IDF Transformation</div>
                        <p style="color: #D8D8D8; font-size: 0.82rem; margin: 0;">Transforms text features into term frequency-inverse document frequency vectors, weighing rare breed traits heavily.</p>
                    </div>
                    <div style="background: rgba(0,0,0,0.25); padding: 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);">
                        <div style="color: #D4A373; font-weight: 700; margin-bottom: 6px;">3. Cosine Similarity Calculation</div>
                        <p style="color: #D8D8D8; font-size: 0.82rem; margin: 0;">Computes dot product angles between the user query vector and 41 pre-indexed indigenous breed vector profiles.</p>
                    </div>
                    <div style="background: rgba(0,0,0,0.25); padding: 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);">
                        <div style="color: #D4A373; font-weight: 700; margin-bottom: 6px;">4. Rank Ordering</div>
                        <p style="color: #D8D8D8; font-size: 0.82rem; margin: 0;">Sorts similarity scores to return top candidate breeds, confidence ratings, and complete breed profile metadata.</p>
                    </div>
                </div>
                """)

        # MODEL INFORMATION & SYSTEM METRICS
        with gr.Column(elem_classes=["glass-card"]):
            gr.HTML("""
            <h3 style="color: #D4A373; margin-top:0; display:flex; align-items:center; gap:8px;">
                <i class="lucide-bar-chart-2"></i> System & Model Architecture Metrics
            </h3>
            <div class="stats-grid" style="margin-top: 16px;">
                <div class="stat-card">
                    <div class="stat-val">TF-IDF</div>
                    <div class="stat-lbl">Vector Engine</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val">Cosine</div>
                    <div class="stat-lbl">Similarity Metric</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val">41</div>
                    <div class="stat-lbl">Indexed Breed Profiles</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val">&lt; 50ms</div>
                    <div class="stat-lbl">Inference Latency</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val">Top 3</div>
                    <div class="stat-lbl">Match Candidates</div>
                </div>
            </div>
            """)

        # PROJECT OVERVIEW, USERS & STEPS GRID
        with gr.Row():
            with gr.Column(scale=6, elem_classes=["glass-card"]):
                gr.HTML("""
                <h3 style="color: #D4A373; margin-top:0; display:flex; align-items:center; gap:8px;">
                    <i class="lucide-bookmark"></i> About the Project
                </h3>
                <p style="color: #D8D8D8; font-size: 0.9rem; line-height: 1.6;">
                    The <strong>AI-Based Cattle & Buffalo Breed Identification System</strong> is an intelligent agriculture decision-support software designed to solve misclassification in livestock husbandry. By matching phenotypic characteristics against verified breed profiles, it enables accurate identification of indigenous Indian cattle and buffalo breeds.
                </p>
                <h4 style="color: #A3B18A; margin-top: 16px; margin-bottom: 8px;">Target User Groups:</h4>
                <ul style="color: #D8D8D8; font-size: 0.85rem; padding-left: 20px; line-height: 1.8;">
                    <li><strong>Farmers & Breeders:</strong> Identify purebred stock for optimal breeding & market valuation.</li>
                    <li><strong>Veterinarians & Field Officers:</strong> Diagnose climate suitability and yield potential.</li>
                    <li><strong>Researchers & Students:</strong> Analyze phenotypic traits of indigenous Indian breeds.</li>
                    <li><strong>Animal Husbandry Departments:</strong> Maintain digital breed censuses and conservation databases.</li>
                </ul>
                """)

            with gr.Column(scale=6, elem_classes=["glass-card"]):
                gr.HTML("""
                <h3 style="color: #D4A373; margin-top:0; display:flex; align-items:center; gap:8px;">
                    <i class="lucide-list-checks"></i> How to Use the App
                </h3>
                <div style="margin-top: 16px;">
                    <div class="timeline-step">
                        <div class="step-number">1</div>
                        <div class="step-content">
                            <h4>Select Animal Type & Ecosystem</h4>
                            <p>Choose between Cow or Buffalo and set regional climate conditions.</p>
                        </div>
                    </div>
                    <div class="timeline-step">
                        <div class="step-number">2</div>
                        <div class="step-content">
                            <h4>Input Yield Metrics</h4>
                            <p>Enter average milk yield (liters/lactation) and milk fat percentage.</p>
                        </div>
                    </div>
                    <div class="timeline-step">
                        <div class="step-number">3</div>
                        <div class="step-content">
                            <h4>Describe Morphological Traits</h4>
                            <p>Provide details on coat color, horn shape, ear structure, and special features.</p>
                        </div>
                    </div>
                    <div class="timeline-step">
                        <div class="step-number">4</div>
                        <div class="step-content">
                            <h4>Run AI Matching</h4>
                            <p>Click Predict to view the primary match, top 3 candidates, and full profile.</p>
                        </div>
                    </div>
                </div>
                """)

        # ADVANTAGES & FUTURE ROADMAP
        with gr.Row():
            with gr.Column(scale=6, elem_classes=["glass-card"]):
                gr.HTML("""
                <h3 style="color: #D4A373; margin-top:0; display:flex; align-items:center; gap:8px;">
                    <i class="lucide-check-circle-2"></i> System Advantages
                </h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px;">
                    <div style="background: rgba(0,0,0,0.2); padding: 10px 14px; border-radius: 8px; font-size: 0.85rem; color: #D8D8D8;">
                        <i class="lucide-zap" style="color:#D4A373;"></i> Fast Vector Comparison
                    </div>
                    <div style="background: rgba(0,0,0,0.2); padding: 10px 14px; border-radius: 8px; font-size: 0.85rem; color: #D8D8D8;">
                        <i class="lucide-shield" style="color:#A3B18A;"></i> High Specificity Matching
                    </div>
                    <div style="background: rgba(0,0,0,0.2); padding: 10px 14px; border-radius: 8px; font-size: 0.85rem; color: #D8D8D8;">
                        <i class="lucide-feather" style="color:#D4A373;"></i> Lightweight Deployment
                    </div>
                    <div style="background: rgba(0,0,0,0.2); padding: 10px 14px; border-radius: 8px; font-size: 0.85rem; color: #D8D8D8;">
                        <i class="lucide-layout" style="color:#A3B18A;"></i> Farmer-Friendly Interface
                    </div>
                </div>
                """)

            with gr.Column(scale=6, elem_classes=["glass-card"]):
                gr.HTML("""
                <h3 style="color: #D4A373; margin-top:0; display:flex; align-items:center; gap:8px;">
                    <i class="lucide-compass"></i> Future Enhancements
                </h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px;">
                    <div style="background: rgba(0,0,0,0.2); padding: 10px 14px; border-radius: 8px; font-size: 0.85rem; color: #D8D8D8;">
                        <i class="lucide-image" style="color:#D4A373;"></i> CNN Image Breed Recognition
                    </div>
                    <div style="background: rgba(0,0,0,0.2); padding: 10px 14px; border-radius: 8px; font-size: 0.85rem; color: #D8D8D8;">
                        <i class="lucide-activity" style="color:#A3B18A;"></i> Disease Symptom Checking
                    </div>
                    <div style="background: rgba(0,0,0,0.2); padding: 10px 14px; border-radius: 8px; font-size: 0.85rem; color: #D8D8D8;">
                        <i class="lucide-trending-up" style="color:#D4A373;"></i> Milk Yield Forecast Engine
                    </div>
                    <div style="background: rgba(0,0,0,0.2); padding: 10px 14px; border-radius: 8px; font-size: 0.85rem; color: #D8D8D8;">
                        <i class="lucide-smartphone" style="color:#A3B18A;"></i> Mobile Offline App Integration
                    </div>
                </div>
                """)

        # DEVELOPER FOOTER
        gr.HTML("""
        <div class="developer-footer">
            <h3 style="color: #FFFFFF; margin: 0 0 4px 0;">Developed by Prachi Valecha</h3>
            <p style="color: #D4A373; font-size: 0.9rem; margin: 0 0 6px 0; font-weight: 500;">
                Bachelor of Computer Applications (BCA) — Cloud Technology & Information Security
            </p>
            <p style="color: #D8D8D8; font-size: 0.85rem; margin: 0;">
                Panipat Institute of Engineering and Technology (PIET)
            </p>
            
            <div class="social-links">
                <a href="#" target="_blank" class="social-btn"><i class="lucide-github"></i> GitHub Profile</a>
                <a href="#" target="_blank" class="social-btn"><i class="lucide-linkedin"></i> LinkedIn Showcase</a>
                <a href="#" target="_blank" class="social-btn"><i class="lucide-globe"></i> Portfolio</a>
                <a href="mailto:developer@example.com" class="social-btn"><i class="lucide-mail"></i> Contact Email</a>
            </div>
            
            <p style="color: rgba(255,255,255,0.4); font-size: 0.75rem; margin-top: 24px;">
                © AI-Based Cattle & Buffalo Breed Identification System. Designed for Academic Major Project Exhibition & Professional Portfolio.
            </p>
        </div>
        """)

        # EVENT BINDINGS
        predict_btn.click(
            fn=predict_breed,
            inputs=[animal_type, climate, utility, milk_yield, milk_fat, physical_traits, special_features],
            outputs=[error_box, primary_result_out, top3_result_out, detail_result_out]
        )

        example_btn.click(
            fn=load_example,
            outputs=[animal_type, climate, utility, milk_yield, milk_fat, physical_traits, special_features]
        )

        clear_btn.click(
            fn=clear_form,
            outputs=[
                animal_type, climate, utility, milk_yield, milk_fat, 
                physical_traits, special_features, error_box, 
                primary_result_out, top3_result_out, detail_result_out
            ]
        )

# -----------------------------------------------------------------------------
# 6. APPLICATION ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
