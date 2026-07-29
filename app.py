import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

print("=" * 50)
print("Current directory:", os.getcwd())
print("Files:", os.listdir())
print("Model exists:", os.path.exists("breed_prediction_model.pkl"))
print("=" * 50)
# ==============================================================================
# MODEL LOADING & FALLBACK HANDLER
# ==============================================================================
MODEL_FILE = "breed_prediction_model.pkl"

def load_model():
    """
    Attempts to load the pre-trained TF-IDF model dictionary.
    Expected structure of loaded dict:
    {
        'tfidf': TfidfVectorizer instance,
        'tfidf_matrix': scipy sparse matrix or numpy array,
        'breed_data': pandas DataFrame containing breed records
    }
    """
    if os.path.exists(MODEL_FILE):
        try:
            model_data = joblib.load(MODEL_FILE)
            if isinstance(model_data, dict) and 'tfidf' in model_data and 'tfidf_matrix' in model_data and 'breed_data' in model_data:
                return model_data, True, "Model loaded successfully from " + MODEL_FILE
            else:
                return None, False, "Loaded file does not contain required keys ('tfidf', 'tfidf_matrix', 'breed_data')."
        except Exception as e:
            return None, False, f"Error loading pickle file: {str(e)}"
    else:
        return None, False, f"Model file '{MODEL_FILE}' not found. Please place it in the application directory."

model_obj, MODEL_LOADED, MODEL_STATUS_MSG = load_model()

# ==============================================================================
# PREDICTION & VALIDATION LOGIC
# ==============================================================================
def predict_breed(animal_type, climate, utility, milk_yield_str, milk_fat_str, physical_traits, special_features):
    """
    Validates user input, formats user feature text, computes TF-IDF vectorization,
    calculates Cosine Similarity, and returns structured result components.
    """
    # Inline validation rules
    errors = []
    
    if not animal_type or animal_type == "Select Animal Type":
        errors.append("Please select a valid Animal Type (Cow or Buffalo).")
        
    if not climate or climate == "Select Climate":
        errors.append("Please select a valid Climate Suitability.")
        
    if not utility or utility == "Select Utility":
        errors.append("Please select a Primary Utility.")
        
    try:
        milk_yield = float(milk_yield_str)
        if milk_yield <= 0:
            errors.append("Average Milk Yield must be a positive number greater than 0.")
    except (ValueError, TypeError):
        errors.append("Please enter a valid numeric value for Average Milk Yield.")

    try:
        milk_fat = float(milk_fat_str)
        if milk_fat <= 0 or milk_fat > 25:
            errors.append("Milk Fat percentage must be a valid positive number (e.g., between 1% and 20%).")
    except (ValueError, TypeError):
        errors.append("Please enter a valid numeric value for Milk Fat %.")

    if not physical_traits or len(physical_traits.strip()) < 3:
        errors.append("Please provide descriptive Physical Traits (minimum 3 characters).")

    if not special_features or len(special_features.strip()) < 3:
        errors.append("Please provide descriptive Special Features or behavior.")

    if errors:
        error_html = "<div class='validation-card-error'>"
        error_html += "<div class='val-header'><svg width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='#be185d' stroke-width='2.2'><path d='M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z'></path><line x1='12' y1='9' x2='12' y2='13'></line><line x1='12' y1='17' x2='12.01' y2='17'></line></svg><span>Input Validation Notice</span></div><ul>"
        for err in errors:
            error_html += f"<li>{err}</li>"
        error_html += "</ul></div>"
        return error_html, "", "", "", ""

    if not MODEL_LOADED:
        error_html = f"<div class='validation-card-error'><div class='val-header'><svg width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='#be185d' stroke-width='2.2'><circle cx='12' cy='12' r='10'></circle><line x1='12' y1='8' x2='12' y2='12'></line><line x1='12' y1='16' x2='12.01' y2='16'></line></svg><span>Model Initialization Error</span></div><p>{MODEL_STATUS_MSG}</p></div>"
        return error_html, "", "", "", ""

    try:
        tfidf_vec = model_obj['tfidf']
        tfidf_mat = model_obj['tfidf_matrix']
        df = model_obj['breed_data']

        # Combine user inputs into unified representation
        combined_user_text = f"Animal: {animal_type}. Climate: {climate}. Utility: {utility}. Milk Yield: {milk_yield} kg/day. Fat: {milk_fat}%. Physical Traits: {physical_traits}. Special Features: {special_features}."
        
        # Transform and compute cosine similarity
        user_vector = tfidf_vec.transform([combined_user_text])
        similarities = cosine_similarity(user_vector, tfidf_mat).flatten()

        # Filtering by animal type if column exists in dataset
        filtered_indices = np.argsort(similarities)[::-1]
        if 'Animal' in df.columns or 'Type' in df.columns:
            type_col = 'Animal' if 'Animal' in df.columns else 'Type'
            type_matches = [i for i in filtered_indices if str(df.iloc[i][type_col]).strip().lower() == animal_type.strip().lower()]
            if type_matches:
                filtered_indices = type_matches

        top_indices = filtered_indices[:3]
        best_idx = top_indices[0]
        best_score = float(similarities[best_idx])
        score_pct = round(best_score * 100, 1)

        best_row = df.iloc[best_idx]
        best_breed_name = best_row.get('Breed', best_row.get('Breed Name', f'Breed #{best_idx+1}'))

        # Result Summary HTML
        result_summary_html = f"""
        <div class="result-success-box">
            <div class="result-badge-hdr">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#be185d" stroke-width="2.5"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                <span>Prediction Successful</span>
            </div>
            <div class="predicted-main-title">{best_breed_name}</div>
            <div class="confidence-container">
                <div class="confidence-label-row">
                    <span class="conf-title">AI Similarity Confidence</span>
                    <span class="conf-value">{score_pct}% Match</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: {score_pct}%;"></div>
                </div>
            </div>
        </div>
        """

        # Top 3 Matching Breeds Cards HTML
        top3_html = "<div class='top3-grid'>"
        for rank, idx in enumerate(top_indices, 1):
            row = df.iloc[idx]
            b_name = row.get('Breed', row.get('Breed Name', f'Breed #{idx+1}'))
            b_score = round(float(similarities[idx]) * 100, 1)
            b_type = row.get('Animal', row.get('Type', animal_type))
            b_origin = row.get('Origin', row.get('Region', 'Native India'))
            
            top3_html += f"""
            <div class="top3-card {'top3-card-first' if rank==1 else ''}">
                <div class="top3-rank-badge">#{rank} Candidate</div>
                <div class="top3-breed-title">{b_name}</div>
                <div class="top3-meta">
                    <span><strong>Type:</strong> {b_type}</span>
                    <span><strong>Origin:</strong> {b_origin}</span>
                </div>
                <div class="top3-score-pill">{b_score}% Similarity</div>
            </div>
            """
        top3_html += "</div>"

        # Detailed Breed Info Card HTML
        b_climate = best_row.get('Climate', best_row.get('Climate Suitability', climate))
        b_yield = best_row.get('Milk Yield', best_row.get('Average Milk Yield', f"{milk_yield} kg/day"))
        b_fat = best_row.get('Milk Fat', best_row.get('Fat %', f"{milk_fat}%"))
        b_utility = best_row.get('Utility', utility)
        b_cross = best_row.get('Crossbreeding', best_row.get('Crossbreeding Programs', 'Extensively used in indigenous improvement programs.'))
        b_traits = best_row.get('Physical Traits', physical_traits)
        b_special = best_row.get('Special Features', special_features)
        b_origin = best_row.get('Origin', best_row.get('Region of Origin', 'Pan-India native tract'))

        breed_info_html = f"""
        <div class="breed-detail-card">
            <div class="detail-header">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#be185d" stroke-width="2.2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                <h3>Comprehensive Breed Dossier: {best_breed_name}</h3>
            </div>
            <div class="detail-grid">
                <div class="detail-item">
                    <div class="d-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#be185d" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg></div>
                    <div class="d-content"><strong>Region of Origin</strong><span>{b_origin}</span></div>
                </div>
                <div class="detail-item">
                    <div class="d-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#be185d" stroke-width="2"><path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"></path></svg></div>
                    <div class="d-content"><strong>Climate Suitability</strong><span>{b_climate}</span></div>
                </div>
                <div class="detail-item">
                    <div class="d-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#be185d" stroke-width="2"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"></path></svg></div>
                    <div class="d-content"><strong>Average Milk Yield</strong><span>{b_yield}</span></div>
                </div>
                <div class="detail-item">
                    <div class="d-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#be185d" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><path d="M12 6v6l4 2"></path></svg></div>
                    <div class="d-content"><strong>Milk Fat Content</strong><span>{b_fat}</span></div>
                </div>
                <div class="detail-item">
                    <div class="d-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#be185d" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg></div>
                    <div class="d-content"><strong>Primary Utility</strong><span>{b_utility}</span></div>
                </div>
                <div class="detail-item">
                    <div class="d-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#be185d" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg></div>
                    <div class="d-content"><strong>Crossbreeding Value</strong><span>{b_cross}</span></div>
                </div>
            </div>
            <div class="detail-text-block">
                <strong>Physical Traits:</strong>
                <p>{b_traits}</p>
            </div>
            <div class="detail-text-block">
                <strong>Special Distinctive Features:</strong>
                <p>{b_special}</p>
            </div>
        </div>
        """

        return "", result_summary_html, top3_html, breed_info_html, "Section successfully evaluated."

    except Exception as ex:
        err_msg = f"<div class='validation-card-error'><div class='val-header'><svg width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='#be185d' stroke-width='2.2'><circle cx='12' cy='12' r='10'></circle><line x1='15' y1='9' x2='9' y2='15'></line><line x1='9' y1='9' x2='15' y2='15'></line></svg><span>Runtime Processing Error</span></div><p>{str(ex)}</p></div>"
        return err_msg, "", "", "", ""

# ==============================================================================
# GRADIO APPLICATION INTERFACE
# ==============================================================================
custom_css = """
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&family=Outfit:wght@600;700;800&display=swap');

/* Global Reset & Base Styling */
* {
    box-sizing: border-box;
    scroll-behavior: smooth;
}

body {
    background-color: #fffafc !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #111827 !important;
    margin: 0 !important;
    padding: 0 !important;
    min-height: 100vh;
}

/* TRANSPARENCY FIX: Ensures background slideshow is fully visible through Gradio wrappers */
.gradio-container, .gradio-container > .main, div[class*="app"], .gradio-app {
    background: transparent !important;
    background-color: transparent !important;
}

/* HIGH CONTRAST TYPOGRAPHY FIXES FOR GRADIO LABELS */
label, .gradio-container label, .gr-form label, span.text-gray-500, .gr-input-label {
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    opacity: 1 !important;
}

.gradio-container p, .gradio-container span, .gradio-container div {
    color: #1e293b;
}

/* Fullscreen Crossfade Background Slideshow with Real Cow/Buffalo Photography */
.slideshow-bg {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: -1;
    overflow: hidden;
}

.slideshow-slide {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-size: cover;
    background-position: center;
    opacity: 0;
    animation: imageFade 28s infinite ease-in-out;
    filter: blur(2px) brightness(0.95);
    transform: scale(1.02);
}

.slideshow-slide:nth-child(1) {
    background-image: url('https://images.unsplash.com/photo-1546445317-29f4545f9d52?q=80&w=1920&auto=format&fit=crop');
    animation-delay: 0s;
}
.slideshow-slide:nth-child(2) {
    background-image: url('https://images.unsplash.com/photo-1594042831518-a15d2a9009eb?q=80&w=1920&auto=format&fit=crop');
    animation-delay: 7s;
}
.slideshow-slide:nth-child(3) {
    background-image: url('https://images.unsplash.com/photo-1570042225831-d98fa7577f1e?q=80&w=1920&auto=format&fit=crop');
    animation-delay: 14s;
}
.slideshow-slide:nth-child(4) {
    background-image: url('https://images.unsplash.com/photo-1527153857715-3908f2bae5e8?q=80&w=1920&auto=format&fit=crop');
    animation-delay: 21s;
}

@keyframes imageFade {
    0% { opacity: 0; transform: scale(1.02); }
    8% { opacity: 0.65; }
    25% { opacity: 0.65; }
    33% { opacity: 0; transform: scale(1.05); }
    100% { opacity: 0; }
}

.slideshow-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: -1;
    background: linear-gradient(180deg, rgba(255, 250, 252, 0.75) 0%, rgba(253, 242, 248, 0.85) 100%);
    backdrop-filter: blur(6px);
}

/* Glassmorphism White & Soft Light Pink Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.92) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
    border: 1px solid rgba(251, 207, 232, 0.8) !important;
    border-radius: 26px !important;
    box-shadow: 0 16px 40px -12px rgba(190, 24, 93, 0.08), 0 4px 12px rgba(0, 0, 0, 0.03) !important;
    padding: 32px !important;
    margin-bottom: 32px !important;
    transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 22px 48px -12px rgba(190, 24, 93, 0.14), 0 8px 20px rgba(0, 0, 0, 0.04) !important;
    border-color: rgba(244, 114, 182, 0.6) !important;
}

/* Hero Section Styling */
.hero-wrapper {
    text-align: center;
    padding: 40px 20px 24px 20px;
    background: rgba(255, 255, 255, 0.95) !important;
}

.hero-logo-box {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 68px;
    height: 68px;
    background: linear-gradient(135deg, #be185d 0%, #9d174d 100%);
    border-radius: 20px;
    box-shadow: 0 10px 25px -5px rgba(190, 24, 93, 0.35);
    margin-bottom: 20px;
}

.hero-title {
    font-family: 'Outfit', 'Manrope', sans-serif !important;
    font-size: 2.7rem !important;
    font-weight: 800 !important;
    color: #881337 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.2 !important;
    margin-bottom: 14px !important;
}

.hero-subtitle {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.15rem !important;
    color: #4c0519 !important;
    max-width: 780px !important;
    margin: 0 auto 28px auto !important;
    line-height: 1.6 !important;
    font-weight: 500 !important;
}

.badges-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 10px;
    margin-top: 16px;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 18px;
    background: #fdf2f8;
    border: 1px solid rgba(244, 114, 182, 0.4);
    border-radius: 50px;
    font-size: 0.88rem;
    font-weight: 700;
    color: #9d174d;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
    transition: all 0.25s ease;
}

.hero-badge:hover {
    background: #be185d;
    color: #ffffff;
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(190, 24, 93, 0.25);
}

/* Header & Section Titles */
.section-hdr-title {
    font-family: 'Manrope', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    color: #881337 !important;
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    margin-bottom: 22px !important;
}

/* Guidance Section Grid */
.guide-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 18px;
    margin-bottom: 24px;
}

.guide-item {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 18px;
    background: #fff;
    border: 1px solid #fbcfe8;
    border-radius: 18px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
}

.guide-icon-box {
    width: 40px;
    height: 40px;
    border-radius: 12px;
    background: #fce7f3;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.guide-text h4 {
    margin: 0 0 4px 0;
    font-size: 1rem;
    font-weight: 700;
    color: #881337;
}

.guide-text p {
    margin: 0;
    font-size: 0.88rem;
    color: #334155;
    line-height: 1.45;
    font-weight: 500;
}

.info-alert-banner {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 20px;
    background: #fdf2f8;
    border: 1px solid #fbcfe8;
    border-radius: 16px;
    color: #9d174d;
    font-size: 0.95rem;
    font-weight: 700;
}

/* Inputs & Form Elements Text Color Fix */
input, select, textarea, .gr-input, .gr-select {
    color: #0f172a !important;
    font-weight: 600 !important;
    background: #ffffff !important;
    border: 1px solid #fbcfe8 !important;
    border-radius: 14px !important;
    padding: 12px 14px !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02) !important;
    transition: all 0.25s ease !important;
}

input:focus, select:focus, textarea:focus {
    border-color: #be185d !important;
    box-shadow: 0 0 0 3px rgba(190, 24, 93, 0.18) !important;
}

/* Primary Action Button (White & Pink Light Luxury) */
.predict-btn {
    background: linear-gradient(135deg, #be185d 0%, #9d174d 100%) !important;
    color: #ffffff !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    border-radius: 16px !important;
    padding: 16px 32px !important;
    border: none !important;
    box-shadow: 0 10px 24px -6px rgba(190, 24, 93, 0.35) !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    width: 100% !important;
    margin-top: 14px !important;
}

.predict-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 14px 30px -6px rgba(190, 24, 93, 0.48) !important;
    background: linear-gradient(135deg, #d946ef 0%, #be185d 100%) !important;
}

/* Validation and Error Display */
.validation-card-error {
    background: #fff1f2;
    border: 1px solid #fecdd3;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 20px;
}

.val-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 800;
    color: #9f1239;
    margin-bottom: 8px;
}

.validation-card-error ul {
    margin: 4px 0 0 20px;
    padding: 0;
    color: #be123c;
    font-size: 0.92rem;
    font-weight: 600;
}

/* Result Section Styling */
.result-success-box {
    background: linear-gradient(135deg, #fdf2f8 0%, #fce7f3 100%);
    border: 1px solid #fbcfe8;
    border-radius: 20px;
    padding: 26px;
    text-align: center;
    margin-bottom: 24px;
}

.result-badge-hdr {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: #9d174d;
    font-weight: 800;
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}

.predicted-main-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.3rem;
    font-weight: 800;
    color: #881337;
    margin-bottom: 16px;
}

.confidence-container {
    max-width: 480px;
    margin: 0 auto;
}

.confidence-label-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.95rem;
    font-weight: 700;
    color: #9d174d;
    margin-bottom: 6px;
}

.progress-bar-bg {
    width: 100%;
    height: 12px;
    background: #ffffff;
    border-radius: 20px;
    overflow: hidden;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.06);
    border: 1px solid #fbcfe8;
}

.progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #f472b6 0%, #be185d 100%);
    border-radius: 20px;
    transition: width 1s ease-in-out;
}

/* Top 3 Cards Grid */
.top3-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}

.top3-card {
    background: #ffffff;
    border: 1px solid #fbcfe8;
    border-radius: 18px;
    padding: 20px;
    position: relative;
    transition: all 0.25s ease;
}

.top3-card-first {
    border-color: #be185d;
    box-shadow: 0 8px 20px -6px rgba(190, 24, 93, 0.2);
}

.top3-rank-badge {
    font-size: 0.78rem;
    font-weight: 800;
    text-transform: uppercase;
    color: #be185d;
    margin-bottom: 6px;
}

.top3-breed-title {
    font-family: 'Manrope', sans-serif;
    font-size: 1.2rem;
    font-weight: 800;
    color: #881337;
    margin-bottom: 8px;
}

.top3-meta span {
    display: block;
    font-size: 0.85rem;
    color: #334155;
    margin-bottom: 4px;
}

.top3-score-pill {
    display: inline-block;
    margin-top: 8px;
    padding: 5px 12px;
    background: #fce7f3;
    color: #9d174d;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 800;
}

/* Detailed Dossier Card */
.breed-detail-card {
    background: #ffffff;
    border-radius: 20px;
    padding: 26px;
    border: 1px solid #fbcfe8;
}

.detail-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
    border-bottom: 1px solid #fce7f3;
    padding-bottom: 14px;
}

.detail-header h3 {
    margin: 0;
    font-family: 'Manrope', sans-serif;
    font-size: 1.35rem;
    color: #881337;
    font-weight: 800;
}

.detail-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 20px;
}

.detail-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px;
    background: #fdf2f8;
    border-radius: 14px;
    border: 1px solid #fbcfe8;
}

.d-icon {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background: #fce7f3;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.d-content strong {
    display: block;
    font-size: 0.78rem;
    color: #9d174d;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    font-weight: 800;
}

.d-content span {
    font-size: 0.92rem;
    font-weight: 700;
    color: #0f172a;
}

.detail-text-block {
    margin-top: 14px;
    padding: 16px;
    background: #fdf2f8;
    border-radius: 14px;
    border: 1px solid #fbcfe8;
}

.detail-text-block strong {
    display: block;
    font-size: 0.88rem;
    color: #be185d;
    margin-bottom: 4px;
    font-weight: 800;
}

.detail-text-block p {
    margin: 0;
    font-size: 0.92rem;
    color: #1e293b;
    line-height: 1.55;
    font-weight: 500;
}

/* Timeline Layout */
.timeline-container {
    display: flex;
    flex-direction: column;
    gap: 16px;
    position: relative;
    margin-top: 10px;
}

.timeline-step {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    background: #ffffff;
    border: 1px solid #fbcfe8;
    border-radius: 18px;
    padding: 20px;
    transition: all 0.25s ease;
}

.timeline-step:hover {
    transform: translateX(4px);
    border-color: #f472b6;
}

.step-num {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: linear-gradient(135deg, #be185d 0%, #9d174d 100%);
    color: #ffffff;
    font-weight: 800;
    font-size: 1rem;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 4px 10px rgba(190, 24, 93, 0.25);
}

.step-content h4 {
    margin: 0 0 4px 0;
    font-size: 1.1rem;
    font-weight: 800;
    color: #881337;
}

.step-content p {
    margin: 0;
    font-size: 0.9rem;
    color: #334155;
    line-height: 1.5;
    font-weight: 500;
}

/* About Model Card */
.model-info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin-top: 16px;
}

.model-info-item {
    padding: 16px;
    background: #ffffff;
    border-radius: 14px;
    border: 1px solid #fbcfe8;
}

.model-info-item label {
    font-size: 0.78rem;
    text-transform: uppercase;
    color: #9d174d;
    font-weight: 800;
    display: block;
    margin-bottom: 4px;
}

.model-info-item span {
    font-size: 0.95rem;
    font-weight: 700;
    color: #0f172a;
}

.status-badge-ok {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 16px;
    background: #fce7f3;
    color: #9d174d;
    border: 1px solid #fbcfe8;
    border-radius: 30px;
    font-size: 0.88rem;
    font-weight: 800;
}

/* Footer Section */
.footer-card {
    text-align: center;
    padding: 36px 20px !important;
    margin-top: 40px !important;
    background: rgba(255, 255, 255, 0.95) !important;
}

.footer-dev-name {
    font-family: 'Outfit', sans-serif;
    font-size: 1.35rem;
    font-weight: 800;
    color: #881337;
    margin-bottom: 4px;
}

.footer-dev-title {
    font-size: 0.92rem;
    color: #475569;
    margin-bottom: 20px;
    font-weight: 600;
}

.footer-links {
    display: flex;
    justify-content: center;
    gap: 14px;
    margin-bottom: 22px;
}

.btn-social {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 22px;
    background: #ffffff;
    border: 1px solid #fbcfe8;
    border-radius: 50px;
    color: #be185d;
    font-size: 0.9rem;
    font-weight: 700;
    text-decoration: none;
    transition: all 0.25s ease;
}

.btn-social:hover {
    background: #be185d;
    color: #ffffff;
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(190, 24, 93, 0.25);
}

.footer-note {
    font-size: 0.85rem;
    color: #64748b;
    font-weight: 500;
}

/* Back to Top Floating Button */
.back-to-top {
    position: fixed;
    bottom: 24px;
    right: 24px;
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background: #be185d;
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 8px 20px rgba(190, 24, 93, 0.35);
    cursor: pointer;
    text-decoration: none;
    transition: all 0.25s ease;
    z-index: 999;
}

.back-to-top:hover {
    transform: translateY(-4px);
    background: #9d174d;
}
"""

with gr.Blocks(title="AI Breed Identification System", css=custom_css) as demo:
    
    # Background Animated Slideshow HTML using direct Unsplash cattle photography links
    gr.HTML("""
    <div class="slideshow-bg">
        <div class="slideshow-slide"></div>
        <div class="slideshow-slide"></div>
        <div class="slideshow-slide"></div>
        <div class="slideshow-slide"></div>
    </div>
    <div class="slideshow-overlay"></div>
    <a href="#top" class="back-to-top" title="Back to Top">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="18 15 12 9 6 15"></polyline></svg>
    </a>
    """)

    with gr.Column(elem_id="top"):
        
        # 1. HERO SECTION
        with gr.Column(elem_classes=["glass-card", "hero-wrapper"]):
            gr.HTML("""
            <div class="hero-logo-box">
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z"></path><path d="M12 6a6 6 0 1 0 6 6 6 6 0 0 0-6-6zm0 10a4 4 0 1 1 4-4 4 4 0 0 1-4 4z"></path></svg>
            </div>
            <h1 class="hero-title">AI-Based Cattle & Buffalo Breed Identification System</h1>
            <p class="hero-subtitle">Identify the most probable cattle or buffalo breed using Artificial Intelligence powered by TF-IDF Vectorization and Cosine Similarity.</p>
            <div class="badges-container">
                <span class="hero-badge"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="12" cy="12" r="10"></circle><polygon points="10 8 16 12 10 16 10 8"></polygon></svg> AI Powered</span>
                <span class="hero-badge"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path></svg> TF-IDF</span>
                <span class="hero-badge"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg> Cosine Similarity</span>
                <span class="hero-badge"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg> Fast Prediction</span>
                <span class="hero-badge"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path></svg> Indian Breeds</span>
                <span class="hero-badge"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="12" cy="12" r="10"></circle><path d="M2 12h20"></path></svg> Global Breeds</span>
            </div>
            """)

        # 2. HOW TO FILL DETAILS SECTION
        with gr.Column(elem_classes=["glass-card"]):
            gr.HTML("""
            <div class="section-hdr-title">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#be185d" stroke-width="2.2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                <span>How to Fill the Details</span>
            </div>
            <div class="guide-grid">
                <div class="guide-item">
                    <div class="guide-icon-box">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#be185d" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
                    </div>
                    <div class="guide-text">
                        <h4>Animal Type</h4>
                        <p>Select whether the subject is a Cow or Buffalo from the dropdown.</p>
                    </div>
                </div>
                <div class="guide-item">
                    <div class="guide-icon-box">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#be185d" stroke-width="2"><path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"></path></svg>
                    </div>
                    <div class="guide-text">
                        <h4>Climate Suitability</h4>
                        <p>Choose the primary ecological region (e.g., Arid, Tropical, Humid).</p>
                    </div>
                </div>
                <div class="guide-item">
                    <div class="guide-icon-box">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#be185d" stroke-width="2"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"></path></svg>
                    </div>
                    <div class="guide-text">
                        <h4>Average Milk Yield</h4>
                        <p>Enter average daily milk yield in kilograms (e.g., 14.5 kg/day).</p>
                    </div>
                </div>
                <div class="guide-item">
                    <div class="guide-icon-box">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#be185d" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><path d="M12 6v6l4 2"></path></svg>
                    </div>
                    <div class="guide-text">
                        <h4>Milk Fat %</h4>
                        <p>Enter the percentage of butterfat content in milk (e.g., 4.5% to 8%).</p>
                    </div>
                </div>
                <div class="guide-item">
                    <div class="guide-icon-box">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#be185d" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path></svg>
                    </div>
                    <div class="guide-text">
                        <h4>Physical Traits</h4>
                        <p>Describe hump size, horn style, coat color, body shape, and forehead profile.</p>
                    </div>
                </div>
                <div class="guide-item">
                    <div class="guide-icon-box">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#be185d" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
                    </div>
                    <div class="guide-text">
                        <h4>Utility & Features</h4>
                        <p>Specify primary use (Milch, Draught, Dual) along with distinct behaviors.</p>
                    </div>
                </div>
            </div>
            <div class="info-alert-banner">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                <span>The more detailed your information, the more accurate the prediction becomes.</span>
            </div>
            """)

        # 3. PREDICTION INPUT SECTION
        with gr.Column(elem_classes=["glass-card"]):
            gr.HTML("""
            <div class="section-hdr-title">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#be185d" stroke-width="2.2"><path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
                <span>Input Animal Attributes</span>
            </div>
            """)
            
            validation_out = gr.HTML()
            
            with gr.Row():
                with gr.Column():
                    animal_input = gr.Dropdown(
                        choices=["Cow", "Buffalo"],
                        label="Animal Type",
                        value="Cow"
                    )
                    climate_input = gr.Dropdown(
                        choices=["Arid & Semi-Arid", "Hot & Humid Tropical", "Moderate & Temperate", "All-Weather Adaptable"],
                        label="Climate Suitability",
                        value="Arid & Semi-Arid"
                    )
                    utility_input = gr.Dropdown(
                        choices=["Milch (Dairy)", "Draught (Work)", "Dual Purpose"],
                        label="Utility",
                        value="Milch (Dairy)"
                    )
                with gr.Column():
                    yield_input = gr.Textbox(
                        label="Average Milk Yield (kg/day)",
                        placeholder="e.g., 14.5",
                        value="14.0"
                    )
                    fat_input = gr.Textbox(
                        label="Milk Fat %",
                        placeholder="e.g., 4.5",
                        value="4.5"
                    )

            with gr.Row():
                with gr.Column():
                    traits_input = gr.TextArea(
                        label="Physical Traits",
                        placeholder="Describe coat color, horn style, hump size, ears...",
                        value="Reddish brown coat, white speckles, pendulous ears, prominent broad forehead, medium curved horns.",
                        lines=3
                    )
                with gr.Column():
                    features_input = gr.TextArea(
                        label="Special Features",
                        placeholder="Describe heat tolerance, disease resistance, docile temperament...",
                        value="Extremely high heat tolerance, resistant to tropical tick diseases, docile temperament.",
                        lines=3
                    )

            predict_btn = gr.Button("Predict Breed", elem_classes=["predict-btn"])

        # 4. RESULT SECTION
        with gr.Column(elem_classes=["glass-card"]):
            gr.HTML("""
            <div class="section-hdr-title">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#be185d" stroke-width="2.2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                <span>AI Identification Results</span>
            </div>
            """)
            
            res_summary_out = gr.HTML("""
            <div style="text-align:center; padding: 24px; color: #475569; font-size:0.95rem; font-weight:600;">
                Submit animal attributes above to compute TF-IDF cosine vector similarities and display match predictions.
            </div>
            """)
            
            top3_out = gr.HTML()
            
            # 5. BREED INFORMATION CARD
            dossier_out = gr.HTML()
            
            status_out = gr.HTML(visible=False)

        # 6. HOW THE SYSTEM WORKS SECTION
        with gr.Column(elem_classes=["glass-card"]):
            gr.HTML("""
            <div class="section-hdr-title">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#be185d" stroke-width="2.2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
                <span>How the System Works</span>
            </div>
            <div class="timeline-container">
                <div class="timeline-step">
                    <div class="step-num">1</div>
                    <div class="step-content">
                        <h4>User Enters Animal Details</h4>
                        <p>Physical attributes, climate suitability, daily milk yield, and distinct traits are collected through the structured interface.</p>
                    </div>
                </div>
                <div class="timeline-step">
                    <div class="step-num">2</div>
                    <div class="step-content">
                        <h4>TF-IDF Converts Text Into Vectors</h4>
                        <p>Term Frequency-Inverse Document Frequency transforms raw textual attributes into high-dimensional numerical feature vectors.</p>
                    </div>
                </div>
                <div class="timeline-step">
                    <div class="step-num">3</div>
                    <div class="step-content">
                        <h4>Cosine Similarity Analysis</h4>
                        <p>The mathematical cosine distance evaluates angular alignment between the input vector and all standard breed dossiers in the dataset.</p>
                    </div>
                </div>
                <div class="timeline-step">
                    <div class="step-num">4</div>
                    <div class="step-content">
                        <h4>Highest Similarity Selection</h4>
                        <p>Statistical ranking determines the primary breed candidate exhibiting the closest feature affinity score.</p>
                    </div>
                </div>
                <div class="timeline-step">
                    <div class="step-num">5</div>
                    <div class="step-content">
                        <h4>Top 3 Matches Displayed</h4>
                        <p>Comprehensive result panels present the best fit along with alternative candidate probabilities and full trait dossiers.</p>
                    </div>
                </div>
            </div>
            """)

        # 7. ABOUT THE AI MODEL
        with gr.Column(elem_classes=["glass-card"]):
            status_badge = "<span class='status-badge-ok'><svg width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='3'><polyline points='20 6 9 17 4 12'></polyline></svg> Loaded Successfully</span>" if MODEL_LOADED else f"<span style='color:#be123c; font-weight:700;'>Error Loading Pickle ({MODEL_STATUS_MSG})</span>"
            
            gr.HTML(f"""
            <div class="section-hdr-title">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#be185d" stroke-width="2.2"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect><line x1="8" y1="21" x2="16" y2="21"></line><line x1="12" y1="17" x2="12" y2="21"></line></svg>
                <span>About the AI Model</span>
            </div>
            <div style="margin-bottom: 16px;">
                <strong style="color:#0f172a;">Model Status:</strong> {status_badge}
            </div>
            <div class="model-info-grid">
                <div class="model-info-item">
                    <label>Prediction Technique</label>
                    <span>TF-IDF Vectorization</span>
                </div>
                <div class="model-info-item">
                    <label>Similarity Algorithm</label>
                    <span>Cosine Similarity</span>
                </div>
                <div class="model-info-item">
                    <label>Prediction Type</label>
                    <span>Similarity-Based Identification</span>
                </div>
                <div class="model-info-item">
                    <label>Dataset</label>
                    <span>Breed Information Dataset</span>
                </div>
                <div class="model-info-item">
                    <label>Output Format</label>
                    <span>Top Matching Breed + Match %</span>
                </div>
            </div>
            """)

        # 8. FOOTER SECTION
        with gr.Column(elem_classes=["glass-card", "footer-card"]):
            gr.HTML("""
            <div class="footer-dev-name">Developer: Prachi Valecha</div>
            <div class="footer-dev-title">
                Bachelor of Computer Applications (BCA)<br>
                Specialization in Cloud Technology & Information Security<br>
                Panipat Institute of Engineering and Technology
            </div>
            <div class="footer-links">
                <a href="https://github.com/yourusername" target="_blank" class="btn-social">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path></svg>
                    <span>GitHub Profile</span>
                </a>
                <a href="https://linkedin.com/in/yourusername" target="_blank" class="btn-social">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path><rect x="2" y="9" width="4" height="12"></rect><circle cx="4" cy="4" r="2"></circle></svg>
                    <span>LinkedIn Profile</span>
                </a>
            </div>
            <div class="footer-note">
                Made with Python, Gradio, TF-IDF and Cosine Similarity.
            </div>
            """)

    # Event binding
    predict_btn.click(
        fn=predict_breed,
        inputs=[
            animal_input,
            climate_input,
            utility_input,
            yield_input,
            fat_input,
            traits_input,
            features_input
        ],
        outputs=[
            validation_out,
            res_summary_out,
            top3_out,
            dossier_out,
            status_out
        ]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        show_error=True
    )
