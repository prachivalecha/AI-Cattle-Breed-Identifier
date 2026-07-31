import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

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
        error_html += "<div class='val-header'><svg width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='#b91c1c' stroke-width='2.5'><path d='M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z'></path><line x1='12' y1='9' x2='12' y2='13'></line><line x1='12' y1='17' x2='12.01' y2='17'></line></svg><span>Input Validation Notice</span></div><ul>"
        for err in errors:
            error_html += f"<li>{err}</li>"
        error_html += "</ul></div>"
        return error_html, "", "", "", ""

    if not MODEL_LOADED:
        error_html = f"<div class='validation-card-error'><div class='val-header'><svg width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='#b91c1c' stroke-width='2.5'><circle cx='12' cy='12' r='10'></circle><line x1='12' y1='8' x2='12' y2='12'></line><line x1='12' y1='16' x2='12.01' y2='16'></line></svg><span>Model Initialization Error</span></div><p>{MODEL_STATUS_MSG}</p></div>"
        return error_html, "", "", "", ""

    try:
        tfidf_vec = model_obj['tfidf']
        tfidf_mat = model_obj['tfidf_matrix']
        df = model_obj['breed_data']

        combined_user_text = f"Animal: {animal_type}. Climate: {climate}. Utility: {utility}. Milk Yield: {milk_yield} kg/day. Fat: {milk_fat}%. Physical Traits: {physical_traits}. Special Features: {special_features}."
        
        user_vector = tfidf_vec.transform([combined_user_text])
        similarities = cosine_similarity(user_vector, tfidf_mat).flatten()

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

        result_summary_html = f"""
        <div class="result-success-box">
            <div class="result-badge-hdr">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#15803d" stroke-width="2.5"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
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
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#15803d" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                <h3>Comprehensive Breed Dossier: {best_breed_name}</h3>
            </div>
            <div class="detail-grid">
                <div class="detail-item">
                    <div class="d-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#15803d" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg></div>
                    <div class="d-content"><strong>Region of Origin</strong><span>{b_origin}</span></div>
                </div>
                <div class="detail-item">
                    <div class="d-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#15803d" stroke-width="2"><path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"></path></svg></div>
                    <div class="d-content"><strong>Climate Suitability</strong><span>{b_climate}</span></div>
                </div>
                <div class="detail-item">
                    <div class="d-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#15803d" stroke-width="2"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"></path></svg></div>
                    <div class="d-content"><strong>Average Milk Yield</strong><span>{b_yield}</span></div>
                </div>
                <div class="detail-item">
                    <div class="d-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#15803d" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><path d="M12 6v6l4 2"></path></svg></div>
                    <div class="d-content"><strong>Milk Fat Content</strong><span>{b_fat}</span></div>
                </div>
                <div class="detail-item">
                    <div class="d-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#15803d" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg></div>
                    <div class="d-content"><strong>Primary Utility</strong><span>{b_utility}</span></div>
                </div>
                <div class="detail-item">
                    <div class="d-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#15803d" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg></div>
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
        err_msg = f"<div class='validation-card-error'><div class='val-header'><svg width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='#b91c1c' stroke-width='2.5'><circle cx='12' cy='12' r='10'></circle><line x1='15' y1='9' x2='9' y2='15'></line><line x1='9' y1='9' x2='15' y2='15'></line></svg><span>Runtime Processing Error</span></div><p>{str(ex)}</p></div>"
        return err_msg, "", "", "", ""

# ==============================================================================
# GRADIO APPLICATION INTERFACE (CLEAN ULTRA-HIGH CONTRAST THEME)
# ==============================================================================
custom_css = """
/* Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Manrope:wght@600;700;800&family=Outfit:wght@600;700;800&display=swap');

/* Master Reset & Root Canvas Fix */
html, body {
    background-color: #f1f5f9 !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: #0f172a !important;
    margin: 0 !important;
    padding: 0 !important;
    min-height: 100vh;
}

/* Force Gradio wrapper transparency so full-screen background slideshow works */
.gradio-container, .gradio-container > .main, div[class*="app"], .gradio-app, .gradio-container > div {
    background: transparent !important;
    background-color: transparent !important;
}

/* HARDCODED ULTRA-BLACK CONTRAST FOR ALL GRADIO LABELS & INPUT TEXT */
label, .gradio-container label, .gr-form label, span.text-gray-500, .gr-input-label, span, p, h1, h2, h3, h4, h5 {
    color: #020617 !important;
    font-weight: 800 !important;
    opacity: 1 !important;
}

/* Form Inputs / Textareas / Dropdowns Hard Styling */
input, select, textarea, .gr-input, .gr-select, .gr-box, div[data-testid="textbox"], div[data-testid="dropdown"] {
    color: #020617 !important;
    background-color: #ffffff !important;
    border: 2px solid #cbd5e1 !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
}

input:focus, select:focus, textarea:focus {
    border-color: #15803d !important;
    box-shadow: 0 0 0 3px rgba(21, 128, 61, 0.2) !important;
}

/* Fullscreen Crossfade Background Slideshow */
.slideshow-bg {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: -2;
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
    animation: imageFade 24s infinite ease-in-out;
    filter: brightness(0.9) contrast(1.05);
}

.slideshow-slide:nth-child(1) {
    background-image: url('https://images.unsplash.com/photo-1546445317-29f4545f9d52?q=80&w=1920&auto=format&fit=crop');
    animation-delay: 0s;
}
.slideshow-slide:nth-child(2) {
    background-image: url('https://images.unsplash.com/photo-1594042831518-a15d2a9009eb?q=80&w=1920&auto=format&fit=crop');
    animation-delay: 6s;
}
.slideshow-slide:nth-child(3) {
    background-image: url('https://images.unsplash.com/photo-1570042225831-d98fa7577f1e?q=80&w=1920&auto=format&fit=crop');
    animation-delay: 12s;
}
.slideshow-slide:nth-child(4) {
    background-image: url('https://images.unsplash.com/photo-1527153857715-3908f2bae5e8?q=80&w=1920&auto=format&fit=crop');
    animation-delay: 18s;
}

@keyframes imageFade {
    0% { opacity: 0; transform: scale(1.0); }
    10% { opacity: 0.85; }
    25% { opacity: 0.85; }
    35% { opacity: 0; transform: scale(1.03); }
    100% { opacity: 0; }
}

.slideshow-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: -1;
    background: rgba(248, 250, 252, 0.65);
    backdrop-filter: blur(4px);
}

/* Clean Solid White Cards (No Washout / Perfect Readability) */
.glass-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.1) !important;
    padding: 30px !important;
    margin-bottom: 28px !important;
}

/* Hero Section */
.hero-wrapper {
    text-align: center;
    background: #ffffff !important;
}

.hero-logo-box {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 64px;
    height: 64px;
    background: linear-gradient(135deg, #166534 0%, #15803d 100%);
    border-radius: 18px;
    margin-bottom: 16px;
    box-shadow: 0 8px 18px rgba(22, 101, 52, 0.3);
}

.hero-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2.5rem !important;
    font-weight: 900 !important;
    color: #14532d !important;
    margin-bottom: 12px !important;
}

.hero-subtitle {
    font-size: 1.1rem !important;
    color: #334155 !important;
    max-width: 750px !important;
    margin: 0 auto 24px auto !important;
    font-weight: 600 !important;
}

.badges-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 10px;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 50px;
    font-size: 0.88rem;
    font-weight: 800;
    color: #166534;
}

/* Section Titles */
.section-hdr-title {
    font-family: 'Manrope', sans-serif !important;
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    color: #14532d !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    margin-bottom: 20px !important;
}

/* Guidance Section Grid */
.guide-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 16px;
    margin-bottom: 20px;
}

.guide-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 16px;
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 14px;
}

.guide-icon-box {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: #dcfce7;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.guide-text h4 {
    margin: 0 0 4px 0;
    font-size: 0.98rem;
    font-weight: 800;
    color: #14532d;
}

.guide-text p {
    margin: 0;
    font-size: 0.88rem;
    color: #334155;
    font-weight: 600;
}

.info-alert-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 14px 18px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 12px;
    color: #1e40af;
    font-size: 0.92rem;
    font-weight: 800;
}

/* Primary Action Button */
.predict-btn {
    background: linear-gradient(135deg, #15803d 0%, #166534 100%) !important;
    color: #ffffff !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.15rem !important;
    font-weight: 800 !important;
    border-radius: 14px !important;
    padding: 16px 32px !important;
    border: none !important;
    box-shadow: 0 8px 20px rgba(21, 128, 61, 0.3) !important;
    cursor: pointer !important;
    width: 100% !important;
    margin-top: 12px !important;
}

.predict-btn:hover {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 100%) !important;
}

/* Validation and Error Display */
.validation-card-error {
    background: #fef2f2;
    border: 2px solid #fca5a5;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 20px;
}

.val-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 800;
    color: #991b1b;
    margin-bottom: 6px;
}

.validation-card-error ul {
    margin: 4px 0 0 20px;
    padding: 0;
    color: #b91c1c;
    font-size: 0.92rem;
    font-weight: 700;
}

/* Results Formatting */
.result-success-box {
    background: #f0fdf4;
    border: 2px solid #86efac;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin-bottom: 20px;
}

.result-badge-hdr {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #15803d;
    font-weight: 800;
    font-size: 0.9rem;
    text-transform: uppercase;
}

.predicted-main-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem;
    font-weight: 900;
    color: #14532d;
    margin-bottom: 12px;
}

.confidence-container {
    max-width: 450px;
    margin: 0 auto;
}

.confidence-label-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.9rem;
    font-weight: 800;
    color: #166534;
    margin-bottom: 6px;
}

.progress-bar-bg {
    width: 100%;
    height: 12px;
    background: #ffffff;
    border-radius: 20px;
    border: 1px solid #bbf7d0;
    overflow: hidden;
}

.progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #22c55e 0%, #15803d 100%);
    border-radius: 20px;
}

.top3-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 14px;
    margin-bottom: 20px;
}

.top3-card {
    background: #f8fafc;
    border: 2px solid #e2e8f0;
    border-radius: 14px;
    padding: 16px;
}

.top3-card-first {
    border-color: #15803d;
    background: #f0fdf4;
}

.top3-rank-badge {
    font-size: 0.75rem;
    font-weight: 800;
    text-transform: uppercase;
    color: #15803d;
}

.top3-breed-title {
    font-size: 1.1rem;
    font-weight: 800;
    color: #0f172a;
    margin: 4px 0 6px 0;
}

.top3-meta span {
    display: block;
    font-size: 0.85rem;
    color: #334155;
    font-weight: 600;
}

.top3-score-pill {
    display: inline-block;
    margin-top: 8px;
    padding: 4px 10px;
    background: #dcfce7;
    color: #14532d;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 800;
}

.breed-detail-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 20px;
    border: 2px solid #e2e8f0;
}

.detail-header h3 {
    margin: 0 0 16px 0;
    font-size: 1.25rem;
    color: #14532d;
    font-weight: 800;
}

.detail-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
}

.detail-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px;
    background: #f8fafc;
    border-radius: 10px;
    border: 1px solid #cbd5e1;
}

.d-icon {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: #dcfce7;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.d-content strong {
    display: block;
    font-size: 0.75rem;
    color: #15803d;
    text-transform: uppercase;
    font-weight: 800;
}

.d-content span {
    font-size: 0.9rem;
    font-weight: 700;
    color: #020617;
}

.detail-text-block {
    margin-top: 12px;
    padding: 14px;
    background: #f8fafc;
    border-radius: 10px;
    border: 1px solid #cbd5e1;
}

.detail-text-block strong {
    display: block;
    font-size: 0.85rem;
    color: #15803d;
    margin-bottom: 4px;
    font-weight: 800;
}

.detail-text-block p {
    margin: 0;
    font-size: 0.9rem;
    color: #0f172a;
    font-weight: 600;
}

.timeline-container {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.timeline-step {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 14px;
    padding: 16px;
}

.step-num {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: #15803d;
    color: #ffffff;
    font-weight: 900;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.step-content h4 {
    margin: 0 0 4px 0;
    font-size: 1rem;
    font-weight: 800;
    color: #14532d;
}

.step-content p {
    margin: 0;
    font-size: 0.88rem;
    color: #334155;
    font-weight: 600;
}

.model-info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-top: 12px;
}

.model-info-item {
    padding: 14px;
    background: #f8fafc;
    border-radius: 10px;
    border: 1px solid #cbd5e1;
}

.model-info-item label {
    font-size: 0.75rem;
    text-transform: uppercase;
    color: #15803d;
    font-weight: 800;
    display: block;
    margin-bottom: 2px;
}

.model-info-item span {
    font-size: 0.9rem;
    font-weight: 700;
    color: #020617;
}

.status-badge-ok {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    background: #dcfce7;
    color: #14532d;
    border: 1px solid #86efac;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 800;
}

.footer-card {
    text-align: center;
    background: #ffffff !important;
}

.footer-dev-name {
    font-size: 1.3rem;
    font-weight: 900;
    color: #14532d;
}

.footer-dev-title {
    font-size: 0.9rem;
    color: #475569;
    margin: 4px 0 16px 0;
    font-weight: 700;
}

.footer-links {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-bottom: 16px;
}

.btn-social {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 18px;
    background: #ffffff;
    border: 2px solid #cbd5e1;
    border-radius: 40px;
    color: #15803d;
    font-size: 0.88rem;
    font-weight: 800;
    text-decoration: none;
}

.btn-social:hover {
    background: #15803d;
    color: #ffffff;
}

.footer-note {
    font-size: 0.82rem;
    color: #64748b;
    font-weight: 600;
}
"""

with gr.Blocks(title="AI Breed Identification System", css=custom_css) as demo:
    
    # HTML Background Slideshow
    gr.HTML("""
    <div class="slideshow-bg">
        <div class="slideshow-slide"></div>
        <div class="slideshow-slide"></div>
        <div class="slideshow-slide"></div>
        <div class="slideshow-slide"></div>
    </div>
    <div class="slideshow-overlay"></div>
    """)

    with gr.Column(elem_id="top"):
        
        # 1. HERO SECTION
        with gr.Column(elem_classes=["glass-card", "hero-wrapper"]):
            gr.HTML("""
            <div class="hero-logo-box">
                <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.5"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z"></path><path d="M12 6a6 6 0 1 0 6 6 6 6 0 0 0-6-6zm0 10a4 4 0 1 1 4-4 4 4 0 0 1-4 4z"></path></svg>
            </div>
            <h1 class="hero-title">AI-Based Cattle & Buffalo Breed Identification System</h1>
            <p class="hero-subtitle">Identify the most probable cattle or buffalo breed using Artificial Intelligence powered by TF-IDF Vectorization and Cosine Similarity.</p>
            <div class="badges-container">
                <span class="hero-badge">AI Powered</span>
                <span class="hero-badge">TF-IDF</span>
                <span class="hero-badge">Cosine Similarity</span>
                <span class="hero-badge">Fast Prediction</span>
                <span class="hero-badge">Indian Breeds</span>
                <span class="hero-badge">Global Breeds</span>
            </div>
            """)

        # 2. HOW TO FILL DETAILS SECTION
        with gr.Column(elem_classes=["glass-card"]):
            gr.HTML("""
            <div class="section-hdr-title">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#15803d" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                <span>How to Fill the Details</span>
            </div>
            <div class="guide-grid">
                <div class="guide-item">
                    <div class="guide-icon-box">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#15803d" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
                    </div>
                    <div class="guide-text">
                        <h4>Animal Type</h4>
                        <p>Select whether the subject is a Cow or Buffalo from the dropdown.</p>
                    </div>
                </div>
                <div class="guide-item">
                    <div class="guide-icon-box">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#15803d" stroke-width="2"><path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"></path></svg>
                    </div>
                    <div class="guide-text">
                        <h4>Climate Suitability</h4>
                        <p>Choose the primary ecological region (e.g., Arid, Tropical, Humid).</p>
                    </div>
                </div>
                <div class="guide-item">
                    <div class="guide-icon-box">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#15803d" stroke-width="2"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"></path></svg>
                    </div>
                    <div class="guide-text">
                        <h4>Average Milk Yield</h4>
                        <p>Enter average daily milk yield in kilograms (e.g., 14.5 kg/day).</p>
                    </div>
                </div>
                <div class="guide-item">
                    <div class="guide-icon-box">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#15803d" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><path d="M12 6v6l4 2"></path></svg>
                    </div>
                    <div class="guide-text">
                        <h4>Milk Fat %</h4>
                        <p>Enter the percentage of butterfat content in milk (e.g., 4.5% to 8%).</p>
                    </div>
                </div>
                <div class="guide-item">
                    <div class="guide-icon-box">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#15803d" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path></svg>
                    </div>
                    <div class="guide-text">
                        <h4>Physical Traits</h4>
                        <p>Describe hump size, horn style, coat color, body shape, and forehead profile.</p>
                    </div>
                </div>
                <div class="guide-item">
                    <div class="guide-icon-box">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#15803d" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
                    </div>
                    <div class="guide-text">
                        <h4>Utility & Features</h4>
                        <p>Specify primary use (Milch, Draught, Dual) along with distinct behaviors.</p>
                    </div>
                </div>
            </div>
            <div class="info-alert-banner">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                <span>The more detailed your information, the more accurate the prediction becomes.</span>
            </div>
            """)

        # 3. PREDICTION INPUT SECTION
        with gr.Column(elem_classes=["glass-card"]):
            gr.HTML("""
            <div class="section-hdr-title">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#15803d" stroke-width="2.5"><path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
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
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#15803d" stroke-width="2.5"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                <span>AI Identification Results</span>
            </div>
            """)
            
            res_summary_out = gr.HTML("""
            <div style="text-align:center; padding: 20px; color: #334155; font-size:0.95rem; font-weight:700;">
                Submit animal attributes above to compute TF-IDF cosine vector similarities and display match predictions.
            </div>
            """)
            
            top3_out = gr.HTML()
            dossier_out = gr.HTML()
            status_out = gr.HTML(visible=False)

        # 5. HOW THE SYSTEM WORKS SECTION
        with gr.Column(elem_classes=["glass-card"]):
            gr.HTML("""
            <div class="section-hdr-title">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#15803d" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
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

        # 6. ABOUT THE AI MODEL
        with gr.Column(elem_classes=["glass-card"]):
            status_badge = "<span class='status-badge-ok'>Loaded Successfully</span>" if MODEL_LOADED else f"<span style='color:#b91c1c; font-weight:800;'>Error Loading Pickle ({MODEL_STATUS_MSG})</span>"
            
            gr.HTML(f"""
            <div class="section-hdr-title">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#15803d" stroke-width="2.5"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect><line x1="8" y1="21" x2="16" y2="21"></line><line x1="12" y1="17" x2="12" y2="21"></line></svg>
                <span>About the AI Model</span>
            </div>
            <div style="margin-bottom: 14px;">
                <strong style="color:#020617; font-size: 1rem;">Model Status:</strong> {status_badge}
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

        # 7. FOOTER SECTION
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
    demo.launch()
