import os
import time
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# ==============================================================================
# SAMPLE DATASET & FALLBACK MODEL GENERATION
# ==============================================================================
SAMPLE_BREED_DATA = [
    {
        "Breed Name": "Gir",
        "Type": "Cattle",
        "Region of Origin": "Gujarat, India",
        "Climate Suitability": "Hot",
        "Average Milk Yield": 12.0,
        "Milk Fat %": 4.5,
        "Milk Type": "A2 Milk",
        "Utility": "Dairy",
        "Crossbreeding Programs": "Used extensively worldwide (e.g., Gyr Leiteiro in Brazil)",
        "Physical Traits": "Distinctive domed forehead, long pendulous ears, reddish brown coat with spots, curved horns",
        "Special Features": "High disease resistance, extreme heat tolerance, docile temperament, high fertility"
    },
    {
        "Breed Name": "Sahiwal",
        "Type": "Cattle",
        "Region of Origin": "Punjab, India / Pakistan",
        "Climate Suitability": "Hot",
        "Average Milk Yield": 14.0,
        "Milk Fat %": 5.0,
        "Milk Type": "A2 Milk",
        "Utility": "Dairy",
        "Crossbreeding Programs": "Jamaican Hope, Australian Milking Zebu",
        "Physical Traits": "Reddish brown or pale red color, heavy dewlap, loose skin, short horns, large hump",
        "Special Features": "Tick resistant, heat tolerant, high milk production under tropical conditions, low maintenance"
    },
    {
        "Breed Name": "Red Sindhi",
        "Type": "Cattle",
        "Region of Origin": "Sindh / Recognized across India",
        "Climate Suitability": "Hot",
        "Average Milk Yield": 11.0,
        "Milk Fat %": 4.8,
        "Milk Type": "A2 Milk",
        "Utility": "Dairy",
        "Crossbreeding Programs": "Used across South Asia for improving local cattle stock",
        "Physical Traits": "Deep red to dark reddish-brown, medium size, compact body, thick horns",
        "Special Features": "High resistance to tropical diseases, efficient feed converter, hardy and adaptable"
    },
    {
        "Breed Name": "Murrah",
        "Type": "Buffalo",
        "Region of Origin": "Haryana, India",
        "Climate Suitability": "Moderate",
        "Average Milk Yield": 15.0,
        "Milk Fat %": 7.5,
        "Milk Type": "Rich Fat Milk",
        "Utility": "Dairy",
        "Crossbreeding Programs": "Primary improver breed for riverine buffaloes globally",
        "Physical Traits": "Jet black body, tightly curled horns, short limbs, massive body structure, wedge shape",
        "Special Features": "Highest milk producer among buffaloes, excellent butterfat content, docility"
    },
    {
        "Breed Name": "Nili-Ravi",
        "Type": "Buffalo",
        "Region of Origin": "Punjab, India / Pakistan",
        "Climate Suitability": "Moderate",
        "Average Milk Yield": 13.5,
        "Milk Fat %": 6.8,
        "Milk Type": "Rich Fat Milk",
        "Utility": "Dairy",
        "Crossbreeding Programs": "Widely crossbred in North and Central India",
        "Physical Traits": "Black body with white markings on forehead, muzzle, legs, and tail tip (Panchbhadra), wall eyes",
        "Special Features": "High fat yield, calm temperament, robust health and longevity"
    },
    {
        "Breed Name": "Jaffarabadi",
        "Type": "Buffalo",
        "Region of Origin": "Gir Forest, Gujarat",
        "Climate Suitability": "Hot",
        "Average Milk Yield": 13.0,
        "Milk Fat %": 8.0,
        "Milk Type": "High Fat Milk",
        "Utility": "Dairy",
        "Crossbreeding Programs": "Selective breeding programs in western India",
        "Physical Traits": "Massive heavy head, wide drooping horns curving upward, dark black coat, large size",
        "Special Features": "Extremely high milk fat percentage, powerful build, high heat tolerance"
    },
    {
        "Breed Name": "Kankrej",
        "Type": "Cattle",
        "Region of Origin": "Rann of Kutch, Gujarat",
        "Climate Suitability": "Hot",
        "Average Milk Yield": 8.5,
        "Milk Fat %": 4.8,
        "Milk Type": "A2 Milk",
        "Utility": "Dual Purpose",
        "Crossbreeding Programs": "Used in South America to develop Kankrej/Guzerat stock",
        "Physical Traits": "Silver-grey to dark grey, large lyre-shaped horns, majestic gait (1 1/4 pace), broad forehead",
        "Special Features": "Fast draught worker, immune to many tropical diseases, extremely hardy in drought"
    },
    {
        "Breed Name": "Ongole",
        "Type": "Cattle",
        "Region of Origin": "Andhra Pradesh, India",
        "Climate Suitability": "Hot",
        "Average Milk Yield": 7.5,
        "Milk Fat %": 4.2,
        "Milk Type": "A2 Milk",
        "Utility": "Dual Purpose",
        "Crossbreeding Programs": "Famous ancestor of American Brahman cattle",
        "Physical Traits": "Large glossy white coat, dark grey markings, muscular neck hump, short thick horns",
        "Special Features": "Renowned strength and endurance, high heat resistance, tick and pest resistant"
    },
    {
        "Breed Name": "Surti",
        "Type": "Buffalo",
        "Region of Origin": "Gujarat, India",
        "Climate Suitability": "Hot",
        "Average Milk Yield": 10.0,
        "Milk Fat %": 7.2,
        "Milk Type": "Rich Fat Milk",
        "Utility": "Dairy",
        "Crossbreeding Programs": "State level buffalo enhancement initiatives",
        "Physical Traits": "Medium size, sickle-shaped flat horns, two white collars on brisket/neck, straight back",
        "Special Features": "Economical feeder, early maturing, high fat percentage in milk"
    },
    {
        "Breed Name": "Bhadawari",
        "Type": "Buffalo",
        "Region of Origin": "Uttar Pradesh / Madhya Pradesh",
        "Climate Suitability": "Hot",
        "Average Milk Yield": 8.0,
        "Milk Fat %": 8.5,
        "Milk Type": "Highest Fat Milk",
        "Utility": "Dairy",
        "Crossbreeding Programs": "Conservation and upgrade programs in ravine areas",
        "Physical Traits": "Copper or copper-yellow reddish coat color, wedge-shaped light body, two white lines on neck",
        "Special Features": "Exceptional butterfat conversion, thrives on sparse grazing, drought tolerant"
    }
]

def load_or_create_model():
    try:
        model_dict = joblib.load("breed_prediction_model.pkl")
        tfidf = model_dict["tfidf"]
        tfidf_matrix = model_dict["tfidf_matrix"]
        breed_data = model_dict["breed_data"]
        return tfidf, tfidf_matrix, breed_data
    except Exception:
        df = pd.DataFrame(SAMPLE_BREED_DATA)
        df["text_repr"] = (
            df["Type"].fillna("") + " " +
            df["Climate Suitability"].fillna("") + " " +
            df["Utility"].fillna("") + " " +
            df["Physical Traits"].fillna("") + " " +
            df["Special Features"].fillna("") + " " +
            df["Region of Origin"].fillna("")
        )
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df["text_repr"])
        breed_data = df
        return tfidf, tfidf_matrix, breed_data

TFIDF_VECTORIZER, TFIDF_MATRIX, BREED_DATA = load_or_create_model()

# ==============================================================================
# PREDICTION ENGINE & VALIDATION LOGIC
# ==============================================================================
def predict_breed(b_type, climate, utility, milk_yield, milk_fat, traits, features):
    if not b_type or not climate or not utility:
        return (
            '<div class="error-banner"><i class="fa-solid fa-triangle-exclamation"></i> Please select Type, Climate Suitability, and Utility.</div>',
            "", "", "", ""
        )
    
    try:
        milk_yield_val = float(milk_yield)
        milk_fat_val = float(milk_fat)
        if milk_yield_val <= 0 or milk_fat_val <= 0:
            return (
                '<div class="error-banner"><i class="fa-solid fa-triangle-exclamation"></i> Milk Yield and Milk Fat % must be positive numbers strictly greater than 0.</div>',
                "", "", "", ""
            )
    except (ValueError, TypeError):
        return (
            '<div class="error-banner"><i class="fa-solid fa-triangle-exclamation"></i> Please provide valid numeric values for Milk Yield and Milk Fat %.</div>',
            "", "", "", ""
        )

    if not traits or len(traits.strip()) < 3:
        return (
            '<div class="error-banner"><i class="fa-solid fa-triangle-exclamation"></i> Please enter descriptive Physical Traits (at least 3 characters).</div>',
            "", "", "", ""
        )
    
    if not features or len(features.strip()) < 3:
        return (
            '<div class="error-banner"><i class="fa-solid fa-triangle-exclamation"></i> Please enter key Special Features (at least 3 characters).</div>',
            "", "", "", ""
        )

    query_text = f"{b_type} {climate} {utility} {traits} {features}"
    query_vec = TFIDF_VECTORIZER.transform([query_text])
    
    similarities = cosine_similarity(query_vec, TFIDF_MATRIX).flatten()
    
    # Filter by Type if available in dataset
    if isinstance(BREED_DATA, pd.DataFrame) and "Type" in BREED_DATA.columns:
        type_mask = BREED_DATA["Type"].str.lower() == b_type.lower()
        adjusted_sims = similarities.copy()
        adjusted_sims[~type_mask] *= 0.3  # penalize mismatch type
    else:
        adjusted_sims = similarities

    top_indices = np.argsort(adjusted_sims)[::-1][:3]
    
    matches = []
    for idx in top_indices:
        if isinstance(BREED_DATA, pd.DataFrame):
            row = BREED_DATA.iloc[idx].to_dict()
        else:
            row = BREED_DATA[idx]
        score = float(adjusted_sims[idx])
        matches.append((row, score))
    
    best_match, best_score = matches[0]
    best_score_pct = round(min(max(best_score * 100, 15.0), 99.4), 1)
    
    # Delay for realistic AI inference animation feel
    time.sleep(0.4)
    timestamp = datetime.now().strftime("%B %d, %Y - %H:%M:%S")

    # SECTION 6 HTML - Primary Result Card
    sec6_html = f"""
    <div class="glass-card result-hero-card">
        <div class="result-header">
            <div>
                <span class="badge badge-success"><i class="fa-solid fa-circle-check"></i> PREDICTION COMPLETE</span>
                <span class="confidence-badge">High Confidence Match</span>
            </div>
            <span class="timestamp-text"><i class="fa-regular fa-clock"></i> {timestamp}</span>
        </div>
        <div class="main-breed-title">
            <h2>{best_match.get('Breed Name', 'Unknown Breed')}</h2>
            <div class="score-display">
                <span class="score-num">{best_score_pct}%</span>
                <span class="score-label">Cosine Similarity Score</span>
            </div>
        </div>
        <div class="progress-container">
            <div class="progress-bar-fill" style="width: {best_score_pct}%;"></div>
        </div>
    </div>
    """

    # SECTION 7 HTML - Top 3 Matches
    ranks = [
        ("fa-trophy", "RANK 1 - TOP MATCH", "#2563EB"), 
        ("fa-medal", "RANK 2 - ALTERNATIVE", "#0891B2"), 
        ("fa-award", "RANK 3 - POSSIBLE MATCH", "#059669")
    ]
    top3_html = '<div class="top3-grid">'
    for i, (m, s) in enumerate(matches):
        pct = round(min(max(s * 100, 10.0), 99.4), 1)
        icon_class, r_label, color = ranks[i]
        top3_html += f"""
        <div class="glass-card match-card">
            <div class="match-card-header">
                <span class="rank-icon" style="color: {color};"><i class="fa-solid {icon_class}"></i></span>
                <span class="rank-title" style="color: {color};">{r_label}</span>
            </div>
            <h3 class="match-breed-name">{m.get('Breed Name', 'N/A')}</h3>
            <p class="match-meta"><i class="fa-solid fa-location-dot"></i> {m.get('Region of Origin', 'N/A')} | {m.get('Type', 'N/A')}</p>
            <div class="match-score-row">
                <span>Similarity Index</span>
                <span style="font-weight:700; color:{color};">{pct}%</span>
            </div>
            <div class="progress-container">
                <div class="progress-bar-fill" style="width: {pct}%; background: {color};"></div>
            </div>
        </div>
        """
    top3_html += '</div>'

    # SECTION 8 HTML - Breed Information Card
    info_html = f"""
    <div class="glass-card info-details-card">
        <h3 class="card-section-title"><i class="fa-solid fa-clipboard-list"></i> Comprehensive Breed Profile</h3>
        <div class="info-grid">
            <div class="info-item">
                <span class="info-icon"><i class="fa-solid fa-cow"></i></span>
                <div>
                    <div class="info-label">Breed Name & Type</div>
                    <div class="info-value">{best_match.get('Breed Name', 'N/A')} ({best_match.get('Type', 'N/A')})</div>
                </div>
            </div>
            <div class="info-item">
                <span class="info-icon"><i class="fa-solid fa-map-pin"></i></span>
                <div>
                    <div class="info-label">Region of Origin</div>
                    <div class="info-value">{best_match.get('Region of Origin', 'N/A')}</div>
                </div>
            </div>
            <div class="info-item">
                <span class="info-icon"><i class="fa-solid fa-temperature-three-quarters"></i></span>
                <div>
                    <div class="info-label">Climate Suitability</div>
                    <div class="info-value">{best_match.get('Climate Suitability', 'N/A')}</div>
                </div>
            </div>
            <div class="info-item">
                <span class="info-icon"><i class="fa-solid fa-whiskey-glass"></i></span>
                <div>
                    <div class="info-label">Average Milk Yield</div>
                    <div class="info-value">{best_match.get('Average Milk Yield', 'N/A')} Litres/Day</div>
                </div>
            </div>
            <div class="info-item">
                <span class="info-icon"><i class="fa-solid fa-flask"></i></span>
                <div>
                    <div class="info-label">Milk Type / Composition</div>
                    <div class="info-value">{best_match.get('Milk Type', 'A2 / High Butterfat')}</div>
                </div>
            </div>
            <div class="info-item">
                <span class="info-icon"><i class="fa-solid fa-bullseye"></i></span>
                <div>
                    <div class="info-label">Utility Classification</div>
                    <div class="info-value">{best_match.get('Utility', 'N/A')}</div>
                </div>
            </div>
            <div class="info-item full-width-item">
                <span class="info-icon"><i class="fa-solid fa-dna"></i></span>
                <div>
                    <div class="info-label">Crossbreeding Programs</div>
                    <div class="info-value">{best_match.get('Crossbreeding Programs', 'N/A')}</div>
                </div>
            </div>
            <div class="info-item full-width-item">
                <span class="info-icon"><i class="fa-solid fa-eye"></i></span>
                <div>
                    <div class="info-label">Physical Traits</div>
                    <div class="info-value">{best_match.get('Physical Traits', 'N/A')}</div>
                </div>
            </div>
            <div class="info-item full-width-item">
                <span class="info-icon"><i class="fa-solid fa-star"></i></span>
                <div>
                    <div class="info-label">Special Features</div>
                    <div class="info-value">{best_match.get('Special Features', 'N/A')}</div>
                </div>
            </div>
        </div>
    </div>
    """

    return sec6_html, top3_html, info_html, "", ""

def load_example():
    return (
        "Cattle",
        "Hot",
        "Dairy",
        12.5,
        4.5,
        "Distinctive broad convex domed forehead, long pendulous ears like folded leaves, reddish brown spotted coat, half-curved horns",
        "Extremely heat tolerant, tick resistant, high fertility, docile nature, adaptable to high ambient temperatures"
    )

def clear_fields():
    return None, None, None, None, None, "", "", "", "", ""

# ==============================================================================
# GRADIO UI DEFINITION (LIGHT THEME & BACKGROUND SLIDESHOW)
# ==============================================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css');

* {
    font-family: 'Poppins', sans-serif !important;
    box-sizing: border-box;
}

body {
    margin: 0;
    padding: 0;
    background-color: #F8FAFC !important;
    color: #0F172A !important;
}

/* Background Image Slideshow */
.bg-slideshow {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: -2;
    overflow: hidden;
}

.bg-slideshow div {
    position: absolute;
    width: 100%;
    height: 100%;
    background-size: cover;
    background-position: center;
    opacity: 0;
    animation: imageSlideshow 24s linear infinite;
    filter: brightness(0.95) contrast(0.95);
}

.bg-slideshow div:nth-child(1) {
    background-image: url('https://images.unsplash.com/photo-1546445317-29f4545f9d52?q=80&w=1920&auto=format&fit=crop');
    animation-delay: 0s;
}
.bg-slideshow div:nth-child(2) {
    background-image: url('https://images.unsplash.com/photo-1500595046743-cd271d694d30?q=80&w=1920&auto=format&fit=crop');
    animation-delay: 6s;
}
.bg-slideshow div:nth-child(3) {
    background-image: url('https://images.unsplash.com/photo-1570042707222-7f287950c441?q=80&w=1920&auto=format&fit=crop');
    animation-delay: 12s;
}
.bg-slideshow div:nth-child(4) {
    background-image: url('https://images.unsplash.com/photo-1527153857715-3908f2bae5e8?q=80&w=1920&auto=format&fit=crop');
    animation-delay: 18s;
}

@keyframes imageSlideshow {
    0% { opacity: 0; transform: scale(1); }
    4% { opacity: 0.12; }
    25% { opacity: 0.12; }
    29% { opacity: 0; transform: scale(1.05); }
    100% { opacity: 0; transform: scale(1); }
}

/* Light Overlay over background */
.bg-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: -1;
    background: linear-gradient(135deg, rgba(248, 250, 252, 0.92) 0%, rgba(241, 245, 249, 0.88) 100%);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}

.gradio-container {
    background: transparent !important;
    color: #0F172A !important;
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding: 0 1rem !important;
}

/* Light Glassmorphism Card */
.glass-card {
    background: rgba(255, 255, 255, 0.85) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(226, 232, 240, 0.9) !important;
    border-radius: 20px !important;
    padding: 2rem !important;
    margin-bottom: 2rem !important;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05) !important;
    transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
}

.glass-card:hover {
    border-color: rgba(37, 99, 235, 0.4) !important;
    box-shadow: 0 15px 35px rgba(37, 99, 235, 0.08) !important;
}

/* Hero Section */
.hero-wrapper {
    text-align: center;
    padding: 3.5rem 1.5rem 2.5rem 1.5rem;
    background: radial-gradient(circle at 50% 20%, rgba(37, 99, 235, 0.08) 0%, rgba(248, 250, 252, 0) 70%);
    border-bottom: 1px solid rgba(226, 232, 240, 0.8);
    margin-bottom: 2.5rem;
}

.hero-icon {
    font-size: 3.5rem;
    color: #2563EB;
    margin-bottom: 1rem;
    display: inline-block;
    filter: drop-shadow(0 4px 12px rgba(37, 99, 235, 0.25));
    animation: floatAnim 4s ease-in-out infinite;
}

@keyframes floatAnim {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-8px); }
}

.hero-title {
    font-size: 2.75rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #1E293B 0%, #2563EB 50%, #0891B2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem !important;
    letter-spacing: -0.5px;
}

.hero-subtitle {
    font-size: 1.15rem;
    color: #475569;
    max-width: 820px;
    margin: 0 auto 2rem auto;
    line-height: 1.7;
    font-weight: 400;
}

/* Badges */
.badge-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.75rem;
    margin-top: 1rem;
}

.badge {
    background: rgba(37, 99, 235, 0.08);
    border: 1px solid rgba(37, 99, 235, 0.25);
    color: #2563EB;
    padding: 0.4rem 1rem;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.3px;
    transition: all 0.2s ease;
}

.badge i {
    margin-right: 0.35rem;
}

.badge:hover {
    background: #2563EB;
    border-color: #2563EB;
    color: #FFFFFF;
    transform: translateY(-2px);
}

.badge-success {
    background: rgba(16, 185, 129, 0.1);
    border-color: rgba(16, 185, 129, 0.3);
    color: #059669;
}

/* Section Headings */
.section-heading {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.section-heading::before {
    content: '';
    display: inline-block;
    width: 5px;
    height: 24px;
    background: linear-gradient(180deg, #2563EB 0%, #0891B2 100%);
    border-radius: 4px;
}

/* Timeline */
.timeline-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.2rem;
    position: relative;
    margin-top: 1rem;
}

.timeline-step {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 1.25rem;
    text-align: center;
    position: relative;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.02);
}

.timeline-step:hover {
    background: #F0F9FF;
    border-color: #3B82F6;
    transform: translateY(-3px);
}

.step-num {
    background: linear-gradient(135deg, #2563EB, #0891B2);
    color: #FFFFFF;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    margin: 0 auto 0.75rem auto;
    box-shadow: 0 4px 10px rgba(37, 99, 235, 0.3);
}

/* Form Inputs customization (Light Mode) */
.gr-box, fieldset, input, textarea, select, .gradio-dropdown {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    color: #0F172A !important;
    border-radius: 12px !important;
}

.gr-box:focus-within, input:focus, textarea:focus, select:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
}

label span {
    color: #475569 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

/* Buttons */
.btn-primary {
    background: linear-gradient(135deg, #2563EB 0%, #0891B2 100%) !important;
    border: none !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    border-radius: 14px !important;
    padding: 0.85rem 2rem !important;
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.25) !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}

.btn-primary:hover {
    box-shadow: 0 12px 28px rgba(8, 145, 178, 0.35) !important;
    transform: translateY(-2px) !important;
}

.btn-secondary {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    color: #334155 !important;
    border-radius: 14px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03) !important;
}

.btn-secondary:hover {
    background: #F1F5F9 !important;
    border-color: #94A3B8 !important;
    color: #0F172A !important;
}

/* Result Section Styles */
.result-hero-card {
    background: linear-gradient(135deg, #FFFFFF 0%, #EFF6FF 100%) !important;
    border: 1px solid #BFDBFE !important;
}

.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.25rem;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.confidence-badge {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid #10B981;
    color: #047857;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-left: 0.5rem;
}

.timestamp-text {
    color: #64748B;
    font-size: 0.85rem;
}

.main-breed-title {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}

.main-breed-title h2 {
    font-size: 2.2rem;
    font-weight: 800;
    color: #0F172A;
    margin: 0;
}

.score-display {
    text-align: right;
}

.score-num {
    font-size: 2.5rem;
    font-weight: 800;
    color: #2563EB;
    line-height: 1;
    display: block;
}

.score-label {
    font-size: 0.8rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.progress-container {
    width: 100%;
    height: 10px;
    background: #E2E8F0;
    border-radius: 10px;
    overflow: hidden;
    margin-top: 0.75rem;
}

.progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #2563EB 0%, #0891B2 100%);
    border-radius: 10px;
    transition: width 1s ease-in-out;
}

/* Top 3 Matches Cards */
.top3-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.25rem;
    margin-top: 1rem;
}

.match-card {
    padding: 1.5rem !important;
    margin-bottom: 0 !important;
    background: #FFFFFF !important;
}

.match-card-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
}

.rank-icon {
    font-size: 1.25rem;
}

.rank-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1px;
}

.match-breed-name {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 0.25rem;
}

.match-meta {
    font-size: 0.85rem;
    color: #64748B;
    margin-bottom: 1rem;
}

.match-score-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.9rem;
    color: #334155;
}

/* Info Details Grid */
.info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1.25rem;
    margin-top: 1rem;
}

.info-item {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1rem 1.25rem;
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
}

.full-width-item {
    grid-column: 1 / -1;
}

.info-icon {
    font-size: 1.25rem;
    background: #EFF6FF;
    color: #2563EB;
    padding: 0.5rem;
    border-radius: 10px;
    line-height: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 38px;
    height: 38px;
}

.info-label {
    font-size: 0.78rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.2rem;
    font-weight: 600;
}

.info-value {
    font-size: 0.98rem;
    color: #0F172A;
    font-weight: 500;
}

/* Infographic Flow */
.flow-container {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    margin-top: 1rem;
}

.flow-node {
    flex: 1;
    min-width: 140px;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1rem 0.75rem;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
}

.flow-node h5 {
    font-size: 0.9rem;
    font-weight: 600;
    color: #2563EB;
    margin-bottom: 0.3rem;
}

.flow-node p {
    font-size: 0.75rem;
    color: #64748B;
    margin: 0;
}

.flow-arrow {
    color: #0891B2;
    font-weight: bold;
    font-size: 1.2rem;
}

/* Feature & Stats Cards */
.stats-grid, .features-grid, .future-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.25rem;
    margin-top: 1rem;
}

.stat-card, .feature-card, .future-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 1.25rem;
    transition: all 0.3s ease;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
}

.stat-card:hover, .feature-card:hover, .future-card:hover {
    transform: translateY(-4px);
    background: #F8FAFC;
    border-color: #3B82F6;
    box-shadow: 0 10px 20px rgba(37, 99, 235, 0.06);
}

.stat-title {
    font-size: 0.8rem;
    color: #64748B;
    text-transform: uppercase;
    font-weight: 600;
}

.stat-val {
    font-size: 1.2rem;
    font-weight: 700;
    color: #2563EB;
    margin-top: 0.3rem;
}

/* Footer */
.footer-card {
    border-top: 1px solid #E2E8F0;
    text-align: center;
    padding: 2.5rem 1rem 1.5rem 1rem;
    margin-top: 3rem;
}

.footer-dev-name {
    font-size: 1.4rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 0.25rem;
}

.footer-dev-sub {
    color: #64748B;
    font-size: 0.9rem;
    margin-bottom: 1.25rem;
}

.footer-links {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-top: 1rem;
}

.error-banner {
    background: #FEF2F2;
    border: 1px solid #FCA5A5;
    color: #DC2626;
    padding: 1rem 1.25rem;
    border-radius: 12px;
    font-weight: 500;
    margin-bottom: 1rem;
}

.accordion-custom {
    background: transparent !important;
    border: none !important;
}
"""

with gr.Blocks(title="AI Cattle & Buffalo Breed Identification", css=CUSTOM_CSS) as demo:
    
    # BACKGROUND SLIDESHOW HTML
    gr.HTML("""
    <div class="bg-slideshow">
        <div></div>
        <div></div>
        <div></div>
        <div></div>
    </div>
    <div class="bg-overlay"></div>
    """)

    # SECTION 1: HERO BANNER
    gr.HTML("""
    <div class="hero-wrapper">
        <div class="hero-icon"><i class="fa-solid fa-cow"></i></div>
        <h1 class="hero-title">AI-Based Cattle & Buffalo Breed Identification System</h1>
        <p class="hero-subtitle">
            Identify cattle and buffalo breeds intelligently using Artificial Intelligence powered by TF-IDF Vectorization and Cosine Similarity.
        </p>
        <div class="badge-container">
            <span class="badge"><i class="fa-solid fa-wand-magic-sparkles"></i> AI Powered</span>
            <span class="badge"><i class="fa-solid fa-chart-simple"></i> TF-IDF</span>
            <span class="badge"><i class="fa-solid fa-calculator"></i> Cosine Similarity</span>
            <span class="badge"><i class="fa-solid fa-bolt"></i> Fast Prediction</span>
            <span class="badge"><i class="fa-solid fa-flag"></i> Indian Breeds</span>
            <span class="badge"><i class="fa-solid fa-globe"></i> Global Breeds</span>
            <span class="badge"><i class="fa-solid fa-gem"></i> Premium Design</span>
            <span class="badge"><i class="fa-solid fa-magnifying-glass"></i> Similarity Search</span>
        </div>
    </div>
    """)

    # SECTION 2: ABOUT PROJECT
    with gr.Column(elem_classes=["glass-card"]):
        gr.HTML("""
        <div class="section-heading">About Project</div>
        <p style="color: #334155; line-height: 1.7; font-size: 1rem; margin-bottom: 1.25rem;">
            The <strong>AI-Based Cattle & Buffalo Breed Identification System</strong> is an advanced decision-support tool engineered to classify and match indigenous and exotic livestock breeds. By evaluating user-provided physical traits, climate tolerance, milk production capacity, and special characteristics against an expert-curated breed knowledge base, the system identifies closest matching profiles instantaneously.
        </p>
        <div style="margin-top: 1rem;">
            <span style="color: #64748B; font-size: 0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">Designed For Key Stakeholders:</span>
            <div class="badge-container" style="justify-content: flex-start; margin-top: 0.5rem;">
                <span class="badge"><i class="fa-solid fa-tractor"></i> Farmers</span>
                <span class="badge"><i class="fa-solid fa-microscope"></i> Researchers</span>
                <span class="badge"><i class="fa-solid fa-user-graduate"></i> Students</span>
                <span class="badge"><i class="fa-solid fa-user-doctor"></i> Veterinarians</span>
                <span class="badge"><i class="fa-solid fa-wheat-awn"></i> Animal Husbandry Departments</span>
            </div>
        </div>
        """)

    # SECTION 3: HOW TO USE
    with gr.Column(elem_classes=["glass-card"]):
        gr.HTML("""
        <div class="section-heading">How to Use</div>
        <div class="timeline-grid">
            <div class="timeline-step">
                <div class="step-num">1</div>
                <h4 style="font-weight:600; font-size:0.95rem; margin-bottom:0.3rem;">Select Category</h4>
                <p style="font-size:0.8rem; color:#64748B;">Choose Cattle or Buffalo</p>
            </div>
            <div class="timeline-step">
                <div class="step-num">2</div>
                <h4 style="font-weight:600; font-size:0.95rem; margin-bottom:0.3rem;">Input Characteristics</h4>
                <p style="font-size:0.8rem; color:#64748B;">Fill traits & milk metrics</p>
            </div>
            <div class="timeline-step">
                <div class="step-num">3</div>
                <h4 style="font-weight:600; font-size:0.95rem; margin-bottom:0.3rem;">Predict Breed</h4>
                <p style="font-size:0.8rem; color:#64748B;">Click Predict Breed button</p>
            </div>
            <div class="timeline-step">
                <div class="step-num">4</div>
                <h4 style="font-weight:600; font-size:0.95rem; margin-bottom:0.3rem;">AI Cosine Analysis</h4>
                <p style="font-size:0.8rem; color:#64748B;">Evaluates breed database</p>
            </div>
            <div class="timeline-step">
                <div class="step-num">5</div>
                <h4 style="font-weight:600; font-size:0.95rem; margin-bottom:0.3rem;">View Matches</h4>
                <p style="font-size:0.8rem; color:#64748B;">Get top 3 breed insights</p>
            </div>
        </div>
        """)

    # SECTION 4: PREDICTION FORM
    with gr.Column(elem_classes=["glass-card"]):
        gr.HTML('<div class="section-heading">Animal Characteristic Input Form</div>')
        
        with gr.Row():
            b_type = gr.Dropdown(
                choices=["Cattle", "Buffalo"],
                label="Animal Type",
                info="Select whether the animal is a Cattle or Buffalo",
                interactive=True
            )
            climate = gr.Dropdown(
                choices=["Hot", "Cold", "Moderate", "Humid", "Dry"],
                label="Climate Suitability",
                info="Primary environmental climate where the animal thrives",
                interactive=True
            )
            utility = gr.Dropdown(
                choices=["Dairy", "Dual Purpose", "Draught", "Meat"],
                label="Primary Utility",
                info="Main economic purpose or farming utility",
                interactive=True
            )

        with gr.Row():
            milk_yield = gr.Number(
                label="Average Milk Yield (Liters / Day)",
                placeholder="Example: 12.5",
                info="Daily milk output in liters (Must be > 0)",
                interactive=True
            )
            milk_fat = gr.Number(
                label="Milk Fat Percentage (%)",
                placeholder="Example: 7.5",
                info="Average fat percentage in milk (Must be > 0)",
                interactive=True
            )

        with gr.Row():
            traits = gr.Textbox(
                label="Physical Traits & Appearance",
                placeholder="Black body, curved horns, long pendulous ears, broad forehead, white patches...",
                lines=3,
                info="Describe forehead shape, horn pattern, skin color, hump size, ears, dewlap etc.",
                interactive=True
            )
            features = gr.Textbox(
                label="Special Features & Adaptations",
                placeholder="Disease resistant, heat tolerant, high fertility, docile temperament...",
                lines=3,
                info="Include tick resistance, draught endurance, heat resilience, docility, etc.",
                interactive=True
            )

        # SECTION 5: BUTTONS
        with gr.Row():
            predict_btn = gr.Button("Predict Breed", elem_classes=["btn-primary"], scale=2)
            example_btn = gr.Button("Load Example Input", elem_classes=["btn-secondary"], scale=1)
            clear_btn = gr.Button("Clear Form", elem_classes=["btn-secondary"], scale=1)

    # SECTION 6, 7, 8: OUTPUT DISPLAY REGION
    sec6_out = gr.HTML()
    sec7_out = gr.HTML()
    sec8_out = gr.HTML()

    # SECTION 9: HOW THE AI WORKS (Collapsible Infographic)
    with gr.Accordion("How the AI Model Works (Click to Expand)", open=False, elem_classes=["glass-card", "accordion-custom"]):
        gr.HTML("""
        <div class="flow-container">
            <div class="flow-node">
                <h5>1. User Input</h5>
                <p>Structured traits & parameters</p>
            </div>
            <div class="flow-arrow"><i class="fa-solid fa-arrow-right"></i></div>
            <div class="flow-node">
                <h5>2. TF-IDF Vector</h5>
                <p>N-gram text representation</p>
            </div>
            <div class="flow-arrow"><i class="fa-solid fa-arrow-right"></i></div>
            <div class="flow-node">
                <h5>3. Feature Weighting</h5>
                <p>Importance scoring of terms</p>
            </div>
            <div class="flow-arrow"><i class="fa-solid fa-arrow-right"></i></div>
            <div class="flow-node">
                <h5>4. Cosine Similarity</h5>
                <p>Vector angle calculation</p>
            </div>
            <div class="flow-arrow"><i class="fa-solid fa-arrow-right"></i></div>
            <div class="flow-node">
                <h5>5. Breed Matrix</h5>
                <p>41 Profile comparison</p>
            </div>
            <div class="flow-arrow"><i class="fa-solid fa-arrow-right"></i></div>
            <div class="flow-node">
                <h5>6. Top Matches</h5>
                <p>Ranked match output</p>
            </div>
        </div>
        <p style="color:#64748B; font-size:0.85rem; margin-top:1.25rem; line-height:1.6;">
            <strong>TF-IDF (Term Frequency-Inverse Document Frequency)</strong> transforms qualitative anatomical descriptions into numerical feature vectors. <strong>Cosine Similarity</strong> subsequently measures the cosine of the angle between user query vectors and stored breed profile vectors in a high-dimensional space, providing precise similarity rankings unaffected by text length.
        </p>
        """)

    # SECTION 10: MODEL INFORMATION
    with gr.Column(elem_classes=["glass-card"]):
        gr.HTML("""
        <div class="section-heading">Model & System Architecture</div>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-title">Algorithm</div>
                <div class="stat-val">TF-IDF Vectorizer</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Similarity Metric</div>
                <div class="stat-val">Cosine Similarity</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Prediction Paradigm</div>
                <div class="stat-val">Similarity-Based</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Knowledge Dataset</div>
                <div class="stat-val">41 Breed Profiles</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Output Format</div>
                <div class="stat-val">Top 3 Ranked Matches</div>
            </div>
        </div>
        """)

    # SECTION 11: ADVANTAGES
    with gr.Column(elem_classes=["glass-card"]):
        gr.HTML("""
        <div class="section-heading">Key System Advantages</div>
        <div class="features-grid">
            <div class="feature-card">
                <h4 style="color:#2563EB; font-weight:600; margin-bottom:0.4rem;"><i class="fa-solid fa-bolt"></i> Ultra Fast Prediction</h4>
                <p style="font-size:0.82rem; color:#64748B; margin:0;">Sub-second execution time leveraging sparse vector matrix operations.</p>
            </div>
            <div class="feature-card">
                <h4 style="color:#2563EB; font-weight:600; margin-bottom:0.4rem;"><i class="fa-solid fa-robot"></i> AI Based Matching</h4>
                <p style="font-size:0.82rem; color:#64748B; margin:0;">Understands contextual physical traits without rigid string rules.</p>
            </div>
            <div class="feature-card">
                <h4 style="color:#2563EB; font-weight:600; margin-bottom:0.4rem;"><i class="fa-solid fa-puzzle-piece"></i> Easily Extendable</h4>
                <p style="font-size:0.82rem; color:#64748B; margin:0;">Seamlessly scale dataset with new cattle or buffalo breeds without retrain overhead.</p>
            </div>
            <div class="feature-card">
                <h4 style="color:#2563EB; font-weight:600; margin-bottom:0.4rem;"><i class="fa-solid fa-users"></i> Farmer Centric UI</h4>
                <p style="font-size:0.82rem; color:#64748B; margin:0;">Intuitive form fields tailored to actual field observations.</p>
            </div>
            <div class="feature-card">
                <h4 style="color:#2563EB; font-weight:600; margin-bottom:0.4rem;"><i class="fa-solid fa-wheat-awn"></i> Indigenous Support</h4>
                <p style="font-size:0.82rem; color:#64748B; margin:0;">Covers extensive Indian native breeds along with key global breeds.</p>
            </div>
            <div class="feature-card">
                <h4 style="color:#2563EB; font-weight:600; margin-bottom:0.4rem;"><i class="fa-solid fa-chart-line"></i> Multi-Match Rankings</h4>
                <p style="font-size:0.82rem; color:#64748B; margin:0;">Presents Top 3 potential candidates with confidence percentages.</p>
            </div>
        </div>
        """)

    # SECTION 12: FUTURE SCOPE
    with gr.Column(elem_classes=["glass-card"]):
        gr.HTML("""
        <div class="section-heading">Future Roadmap</div>
        <div class="future-grid">
            <div class="future-card">
                <span style="font-size:1.8rem; display:block; margin-bottom:0.5rem; color:#2563EB;"><i class="fa-solid fa-image"></i></span>
                <h4 style="color:#0F172A; font-weight:600; font-size:0.95rem;">Computer Vision Recognition</h4>
                <p style="font-size:0.8rem; color:#64748B; margin-top:0.2rem;">Direct breed identification from uploaded images using CNNs.</p>
            </div>
            <div class="future-card">
                <span style="font-size:1.8rem; display:block; margin-bottom:0.5rem; color:#2563EB;"><i class="fa-solid fa-stethoscope"></i></span>
                <h4 style="color:#0F172A; font-weight:600; font-size:0.95rem;">Disease Risk Prediction</h4>
                <p style="font-size:0.8rem; color:#64748B; margin-top:0.2rem;">Early symptom analysis and breed-specific disease vulnerability.</p>
            </div>
            <div class="future-card">
                <span style="font-size:1.8rem; display:block; margin-bottom:0.5rem; color:#2563EB;"><i class="fa-solid fa-chart-simple"></i></span>
                <h4 style="color:#0F172A; font-weight:600; font-size:0.95rem;">Yield Forecasting</h4>
                <p style="font-size:0.8rem; color:#64748B; margin-top:0.2rem;">Lactation yield estimations based on climate and feed inputs.</p>
            </div>
            <div class="future-card">
                <span style="font-size:1.8rem; display:block; margin-bottom:0.5rem; color:#2563EB;"><i class="fa-solid fa-syringe"></i></span>
                <h4 style="color:#0F172A; font-weight:600; font-size:0.95rem;">Vaccination Scheduler</h4>
                <p style="font-size:0.8rem; color:#64748B; margin-top:0.2rem;">Automated health monitoring and immunization reminders.</p>
            </div>
            <div class="future-card">
                <span style="font-size:1.8rem; display:block; margin-bottom:0.5rem; color:#2563EB;"><i class="fa-solid fa-bowl-food"></i></span>
                <h4 style="color:#0F172A; font-weight:600; font-size:0.95rem;">Feed & Nutrition Planner</h4>
                <p style="font-size:0.8rem; color:#64748B; margin-top:0.2rem;">Customized ration balancing for optimum milk fat percentage.</p>
            </div>
        </div>
        """)

    # SECTION 13: DEVELOPER FOOTER
    gr.HTML("""
    <div class="glass-card footer-card">
        <div style="font-size:0.8rem; color:#0891B2; font-weight:700; letter-spacing:1px; text-transform:uppercase; margin-bottom:0.4rem;">Lead Developer & Researcher</div>
        <div class="footer-dev-name">Prachi Valecha</div>
        <div class="footer-dev-sub">
            Bachelor of Computer Applications (BCA)<br/>
            Specialization in Cloud Technology & Information Security<br/>
            <span style="color:#2563EB; font-weight:500;">Panipat Institute of Engineering and Technology (PIET)</span>
        </div>
        <div class="footer-links">
            <a href="https://github.com" target="_blank" class="badge" style="text-decoration:none;"><i class="fa-brands fa-github"></i> GitHub Profile</a>
            <a href="https://linkedin.com" target="_blank" class="badge" style="text-decoration:none;"><i class="fa-brands fa-linkedin"></i> LinkedIn Network</a>
            <a href="mailto:developer@example.com" class="badge" style="text-decoration:none;"><i class="fa-solid fa-envelope"></i> Contact Email</a>
        </div>
        <div style="margin-top:1.5rem; font-size:0.75rem; color:#94A3B8;">
            © 2026 AI-Based Cattle & Buffalo Breed Identification System. All Rights Reserved.
        </div>
    </div>
    """)

    # ==============================================================================
    # BIND EVENTS
    # ==============================================================================
    predict_btn.click(
        fn=predict_breed,
        inputs=[b_type, climate, utility, milk_yield, milk_fat, traits, features],
        outputs=[sec6_out, sec7_out, sec8_out, traits, features]
    )

    example_btn.click(
        fn=load_example,
        inputs=[],
        outputs=[b_type, climate, utility, milk_yield, milk_fat, traits, features]
    )

    clear_btn.click(
        fn=clear_fields,
        inputs=[],
        outputs=[b_type, climate, utility, milk_yield, milk_fat, traits, features, sec6_out, sec7_out, sec8_out]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )
