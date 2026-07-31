import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# Suppress version & deprecation warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# MODEL LOADING
# ==============================================================================
MODEL_FILE = "breed_prediction_model.pkl"

def load_model():
    if os.path.exists(MODEL_FILE):
        try:
            model_data = joblib.load(MODEL_FILE)
            if isinstance(model_data, dict) and 'tfidf' in model_data and 'tfidf_matrix' in model_data and 'breed_data' in model_data:
                return model_data, True, "Model loaded successfully."
            else:
                return None, False, "Loaded file missing required keys ('tfidf', 'tfidf_matrix', 'breed_data')."
        except Exception as e:
            return None, False, f"Error loading pickle file: {str(e)}"
    else:
        return None, False, f"Model file '{MODEL_FILE}' not found."

model_obj, MODEL_LOADED, MODEL_STATUS_MSG = load_model()

# ==============================================================================
# IMPROVED PREDICTION & VALIDATION LOGIC
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
            errors.append("Average Milk Yield must be greater than 0.")
    except (ValueError, TypeError):
        errors.append("Please enter a valid numeric value for Average Milk Yield.")

    try:
        milk_fat = float(milk_fat_str)
        if milk_fat <= 0 or milk_fat > 25:
            errors.append("Milk Fat percentage must be between 1% and 25%.")
    except (ValueError, TypeError):
        errors.append("Please enter a valid numeric value for Milk Fat %.")

    if not physical_traits or len(physical_traits.strip()) < 3:
        errors.append("Please provide descriptive Physical Traits (minimum 3 characters).")

    if not special_features or len(special_features.strip()) < 3:
        errors.append("Please provide descriptive Special Features.")

    if errors:
        error_html = "<div class='validation-card-error'>"
        error_html += "<div class='val-header' style='color:#b91c1c !important; font-weight:800; font-size:1.05rem;'><span>⚠️ Input Validation Notice</span></div><ul style='margin-top:6px;'>"
        for err in errors:
            error_html += f"<li style='color:#b91c1c !important; font-weight:700;'>{err}</li>"
        error_html += "</ul></div>"
        return error_html, "", "", "", ""

    if not MODEL_LOADED:
        error_html = f"<div class='validation-card-error'><div class='val-header' style='color:#b91c1c !important;'><span>⚠️ Model Error</span></div><p style='color:#b91c1c !important;'>{MODEL_STATUS_MSG}</p></div>"
        return error_html, "", "", "", ""

    try:
        tfidf_vec = model_obj['tfidf']
        tfidf_mat = model_obj['tfidf_matrix']
        df = model_obj['breed_data']

        # Weighted text construction: Boost physical traits and features 3x so physical details drive the match
        weighted_user_text = f"{animal_type} {animal_type} {climate} {utility} " \
                             f"{physical_traits} {physical_traits} {physical_traits} " \
                             f"{special_features} {special_features} " \
                             f"milk yield {milk_yield} fat {milk_fat} percent"
        
        # Transform user text into TF-IDF vector
        user_vector = tfidf_vec.transform([weighted_user_text])
        similarities = cosine_similarity(user_vector, tfidf_mat).flatten()

        # Keyword Boost Logic for exact trait alignment (e.g. domed, pendulous, hump, horns)
        traits_lower = (physical_traits + " " + special_features).lower()
        keywords_to_check = ['pendulous', 'domed', 'hump', 'lyre', 'dewlap', 'curved', 'broad', 'horns', 'heat']
        
        boosts = np.zeros(len(df))
        for key in keywords_to_check:
            if key in traits_lower:
                for idx, row in df.iterrows():
                    row_text = " ".join([str(val) for val in row.values]).lower()
                    if key in row_text:
                        boosts[idx] += 0.04  # Extra similarity boost for specific trait matches

        adjusted_similarities = similarities + boosts
        top_indices = np.argsort(adjusted_similarities)[::-1]

        # Filter indices by Animal Type if category column exists
        type_col = None
        for col in ['Animal', 'Type', 'Animal Type', 'Category']:
            if col in df.columns:
                type_col = col
                break
        
        if type_col:
            filtered = [i for i in top_indices if str(df.iloc[i][type_col]).strip().lower() == animal_type.strip().lower()]
            if filtered:
                top_indices = filtered

        top3_indices = top_indices[:3]
        best_idx = top3_indices[0]
        best_score = float(adjusted_similarities[best_idx])
        # Cap confidence score display logically
        score_pct = round(min(best_score * 100, 98.5), 1)

        # Retrieve breed name safely
        best_row = df.iloc[best_idx]
        breed_col = None
        for col in ['Breed', 'Breed Name', 'Name', 'Breed_Name']:
            if col in df.columns:
                breed_col = col
                break
        
        best_breed_name = str(best_row[breed_col]) if breed_col else f"Breed #{best_idx+1}"

        # Result Summary HTML with High Contrast Styling
        result_summary_html = f"""
        <div class="result-success-box">
            <div class="result-badge-hdr" style="color: #15803d !important; font-weight: 900; font-size: 0.95rem;">
                <span>✓ PREDICTION SUCCESSFUL</span>
            </div>
            <div class="predicted-main-title" style="color: #14532d !important; font-size: 2.3rem; font-weight: 900; margin: 8px 0;">{best_breed_name}</div>
            <div class="confidence-container" style="max-width: 480px; margin: 0 auto;">
                <div class="confidence-label-row" style="color: #0f172a !important; font-weight: 800; font-size: 1rem; display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #0f172a !important;">AI Match Confidence</span>
                    <span style="color: #15803d !important; font-weight: 900;">{score_pct}% Match</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: {score_pct}%;"></div>
                </div>
            </div>
        </div>
        """

        # Top 3 Candidates HTML with Dark Text Overrides
        top3_html = "<div class='top3-grid'>"
        for rank, idx in enumerate(top3_indices, 1):
            row = df.iloc[idx]
            b_name = str(row[breed_col]) if breed_col else f"Breed #{idx+1}"
            b_score = round(min(float(adjusted_similarities[idx]) * 100, 98.5), 1)
            b_type = row.get('Animal', row.get('Type', animal_type))
            b_origin = row.get('Origin', row.get('Region', 'Native India'))
            
            top3_html += f"""
            <div class="top3-card {'top3-card-first' if rank==1 else ''}">
                <div class="top3-rank-badge" style="color: #15803d !important; font-weight: 900; font-size: 0.82rem; text-transform: uppercase;">Candidate #{rank}</div>
                <div class="top3-breed-title" style="color: #0f172a !important; font-weight: 900; font-size: 1.2rem; margin: 4px 0;">{b_name}</div>
                <div class="top3-meta" style="color: #1e293b !important; font-weight: 700; font-size: 0.9rem; line-height: 1.5;">
                    <span style="color: #1e293b !important; display: block;"><strong style="color: #15803d !important;">Type:</strong> {b_type}</span>
                    <span style="color: #1e293b !important; display: block;"><strong style="color: #15803d !important;">Origin:</strong> {b_origin}</span>
                </div>
                <div class="top3-score-pill" style="color: #14532d !important; font-weight: 900; background: #dcfce7; padding: 4px 12px; border-radius: 20px; display: inline-block; margin-top: 8px; font-size: 0.85rem;">{b_score}% Similarity</div>
            </div>
            """
        top3_html += "</div>"

        # Safe Extraction for Comprehensive Dossier
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
                <h3 style="color: #14532d !important; font-weight: 900; font-size: 1.3rem; margin-bottom: 14px;">Comprehensive Breed Dossier: {best_breed_name}</h3>
            </div>
            <div class="detail-grid">
                <div class="detail-item">
                    <div class="d-content"><strong style="color: #15803d !important; display: block; font-size:0.78rem; text-transform:uppercase;">Region of Origin</strong><span style="color: #0f172a !important; font-weight: 800; font-size:0.95rem;">{b_origin}</span></div>
                </div>
                <div class="detail-item">
                    <div class="d-content"><strong style="color: #15803d !important; display: block; font-size:0.78rem; text-transform:uppercase;">Climate Suitability</strong><span style="color: #0f172a !important; font-weight: 800; font-size:0.95rem;">{b_climate}</span></div>
                </div>
                <div class="detail-item">
                    <div class="d-content"><strong style="color: #15803d !important; display: block; font-size:0.78rem; text-transform:uppercase;">Average Milk Yield</strong><span style="color: #0f172a !important; font-weight: 800; font-size:0.95rem;">{b_yield}</span></div>
                </div>
                <div class="detail-item">
                    <div class="d-content"><strong style="color: #15803d !important; display: block; font-size:0.78rem; text-transform:uppercase;">Milk Fat Content</strong><span style="color: #0f172a !important; font-weight: 800; font-size:0.95rem;">{b_fat}</span></div>
                </div>
                <div class="detail-item">
                    <div class="d-content"><strong style="color: #15803d !important; display: block; font-size:0.78rem; text-transform:uppercase;">Primary Utility</strong><span style="color: #0f172a !important; font-weight: 800; font-size:0.95rem;">{b_utility}</span></div>
                </div>
                <div class="detail-item">
                    <div class="d-content"><strong style="color: #15803d !important; display: block; font-size:0.78rem; text-transform:uppercase;">Crossbreeding Value</strong><span style="color: #0f172a !important; font-weight: 800; font-size:0.95rem;">{b_cross}</span></div>
                </div>
            </div>
            <div class="detail-text-block">
                <strong style="color: #15803d !important; font-weight: 900; display: block; margin-bottom: 4px;">Physical Traits:</strong>
                <p style="color: #0f172a !important; font-weight: 700; margin: 0; font-size:0.92rem;">{b_traits}</p>
            </div>
            <div class="detail-text-block" style="margin-top: 10px;">
                <strong style="color: #15803d !important; font-weight: 900; display: block; margin-bottom: 4px;">Special Distinctive Features:</strong>
                <p style="color: #0f172a !important; font-weight: 700; margin: 0; font-size:0.92rem;">{b_special}</p>
            </div>
        </div>
        """

        return "", result_summary_html, top3_html, breed_info_html, "Evaluated."

    except Exception as ex:
        err_msg = f"<div class='validation-card-error'><div class='val-header'><span style='color:#b91c1c !important;'>⚠️ Prediction Error</span></div><p style='color:#b91c1c !important;'>{str(ex)}</p></div>"
        return err_msg, "", "", "", ""

# ==============================================================================
# GRADIO APPLICATION INTERFACE (STRICT HIGH CONTRAST & FORCE LIGHT THEME)
# ==============================================================================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* DISABLE DARK MODE ENTIRELY AT ALL LEVEL SELECTORS */
:root, html, body, .dark, .gradio-container, .gradio-container * {
    color-scheme: light !important;
    --body-text-color: #0f172a !important;
    --block-label-text-color: #0f172a !important;
    --block-title-text-color: #0f172a !important;
    --block-background-fill: #ffffff !important;
    --input-background-fill: #ffffff !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f8fafc !important;
    --border-color-primary: #cbd5e1 !important;
}

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background-color: #f8fafc !important;
    color: #0f172a !important;
}

/* HARD OVERRIDE GRADIO INTERNAL DARK CONTAINERS & CARDS */
div[class*="block"], 
div[class*="cell"], 
div[class*="form"],
div[class*="gr-box"],
div[class*="input"],
div[data-testid="textbox"],
div[data-testid="dropdown"],
.block, .form, .gr-form, .gr-box {
    background-color: #ffffff !important;
    background: #ffffff !important;
    border-color: #cbd5e1 !important;
}

/* ALL FORM INPUTS & TEXTAREAS CLEAN WHITE WITH DARK TEXT */
input, select, textarea, 
.gradio-container input, 
.gradio-container select, 
.gradio-container textarea,
.gradio-container .input-container,
div[data-testid="textbox"] textarea,
div[data-testid="dropdown"] input,
div[data-testid="dropdown"] .single-select {
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1.5px solid #94a3b8 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
}

/* ALL LABELS IN BOLD CHARCOAL */
label span, .gradio-container label span, .gradio-container .block-title {
    color: #0f172a !important;
    font-weight: 800 !important;
    font-size: 0.92rem !important;
}

/* Background Slideshow Layer */
.slideshow-bg {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 0;
    pointer-events: none;
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
    animation: imageFade 20s infinite ease-in-out;
}

.slideshow-slide:nth-child(1) {
    background-image: url('https://images.unsplash.com/photo-1546445317-29f4545f9d52?q=80&w=1920&auto=format&fit=crop');
    animation-delay: 0s;
}
.slideshow-slide:nth-child(2) {
    background-image: url('https://images.unsplash.com/photo-1500595046743-cd271d694d30?q=80&w=1920&auto=format&fit=crop');
    animation-delay: 5s;
}
.slideshow-slide:nth-child(3) {
    background-image: url('https://images.unsplash.com/photo-1570042225831-d98fa7577f1e?q=80&w=1920&auto=format&fit=crop');
    animation-delay: 10s;
}
.slideshow-slide:nth-child(4) {
    background-image: url('https://images.unsplash.com/photo-1527153857715-3908f2bae5e8?q=80&w=1920&auto=format&fit=crop');
    animation-delay: 15s;
}

@keyframes imageFade {
    0% { opacity: 0; }
    15% { opacity: 0.4; }
    35% { opacity: 0.4; }
    50% { opacity: 0; }
    100% { opacity: 0; }
}

/* Glassmorphic Cards Overlay */
.glass-card {
    background: rgba(255, 255, 255, 0.96) !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06) !important;
    padding: 24px !important;
    margin-bottom: 24px !important;
    position: relative;
    z-index: 10;
}

.hero-title {
    font-size: 2.2rem !important;
    font-weight: 900 !important;
    color: #14532d !important;
    margin-bottom: 8px !important;
    text-align: center;
}

.hero-subtitle {
    font-size: 1rem !important;
    color: #0f172a !important;
    text-align: center;
    max-width: 750px !important;
    margin: 0 auto 16px auto !important;
    font-weight: 700 !important;
}

.badges-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
}

.hero-badge {
    padding: 6px 14px;
    background: #f0fdf4 !important;
    border: 1px solid #bbf7d0 !important;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 800 !important;
    color: #166534 !important;
}

.section-hdr-title {
    font-size: 1.25rem !important;
    font-weight: 900 !important;
    color: #14532d !important;
    margin-bottom: 16px !important;
}

.guide-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
}

.guide-item {
    padding: 12px;
    background: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px;
}

.guide-item h4 {
    margin: 0 0 4px 0;
    font-size: 0.95rem;
    color: #14532d !important;
    font-weight: 900 !important;
}

.guide-item p {
    margin: 0;
    font-size: 0.85rem;
    color: #0f172a !important;
    font-weight: 700 !important;
}

.predict-btn {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 100%) !important;
    color: #ffffff !important;
    font-size: 1.1rem !important;
    font-weight: 900 !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(22, 163, 74, 0.3) !important;
    cursor: pointer !important;
    width: 100% !important;
    margin-top: 12px !important;
}

.result-success-box {
    background: #f0fdf4 !important;
    border: 2px solid #86efac !important;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 16px;
}

.predicted-main-title {
    font-size: 2rem;
    font-weight: 900;
    color: #14532d !important;
    margin: 6px 0;
}

.progress-bar-bg {
    width: 100%;
    height: 10px;
    background: #ffffff !important;
    border-radius: 20px;
    border: 1px solid #bbf7d0 !important;
    overflow: hidden;
}

.progress-bar-fill {
    height: 100%;
    background: #16a34a !important;
    border-radius: 20px;
}

.top3-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
}

.top3-card {
    background: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 12px;
    padding: 14px;
}

.top3-card-first {
    border-color: #16a34a !important;
    background: #f0fdf4 !important;
}

.breed-detail-card {
    background: #ffffff !important;
    border-radius: 12px;
    padding: 18px;
    border: 1.5px solid #cbd5e1 !important;
}

.detail-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px;
    margin-bottom: 14px;
}

.detail-item {
    padding: 10px;
    background: #f8fafc !important;
    border-radius: 8px;
    border: 1px solid #cbd5e1 !important;
}

.timeline-step {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    background: #f8fafc !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 8px;
}

.step-num {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #16a34a !important;
    color: #ffffff !important;
    font-weight: 900 !important;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 0.85rem;
}

.footer-card {
    text-align: center;
    background: #ffffff !important;
}

.footer-social-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 18px;
    margin: 4px;
    background: #f0fdf4;
    border: 1.5px solid #bbf7d0;
    border-radius: 50px;
    color: #14532d !important;
    font-weight: 800 !important;
    text-decoration: none;
    font-size: 0.88rem;
}

.footer-social-btn:hover {
    background: #15803d;
    color: #ffffff !important;
}

.validation-card-error {
    background: #fef2f2;
    border: 1px solid #fca5a5;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 12px;
}
"""

js_remove_dark = """
function() {
    document.documentElement.classList.remove('dark');
    document.body.classList.remove('dark');
    const observer = new MutationObserver(function() {
        if (document.documentElement.classList.contains('dark')) {
            document.documentElement.classList.remove('dark');
        }
        if (document.body.classList.contains('dark')) {
            document.body.classList.remove('dark');
        }
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
}
"""

with gr.Blocks(title="AI Breed Identification System", theme=gr.themes.Default(primary_hue="green"), js=js_remove_dark) as demo:
    
    # Render Background Slideshow Layer
    gr.HTML("""
    <div class="slideshow-bg">
        <div class="slideshow-slide"></div>
        <div class="slideshow-slide"></div>
        <div class="slideshow-slide"></div>
        <div class="slideshow-slide"></div>
    </div>
    """)

    with gr.Column():
        
        # 1. HERO SECTION
        with gr.Column(elem_classes=["glass-card"]):
            gr.HTML("""
            <h1 class="hero-title">🐄 AI Cattle & Buffalo Breed Identification</h1>
            <p class="hero-subtitle">Identify the most probable cattle or buffalo breed using TF-IDF Vectorization and Cosine Similarity.</p>
            <div class="badges-container">
                <span class="hero-badge">AI Powered</span>
                <span class="hero-badge">TF-IDF</span>
                <span class="hero-badge">Cosine Similarity</span>
                <span class="hero-badge">Fast Prediction</span>
                <span class="hero-badge">Indian Breeds</span>
            </div>
            """)

        # 2. HOW TO FILL DETAILS SECTION
        with gr.Column(elem_classes=["glass-card"]):
            gr.HTML("""
            <div class="section-hdr-title">📌 How to Fill the Details</div>
            <div class="guide-grid">
                <div class="guide-item">
                    <h4>Animal Type</h4>
                    <p>Select Cow or Buffalo.</p>
                </div>
                <div class="guide-item">
                    <h4>Climate Suitability</h4>
                    <p>Choose region (Arid, Tropical, Humid).</p>
                </div>
                <div class="guide-item">
                    <h4>Milk Yield</h4>
                    <p>Daily yield in kg (e.g. 14.5 kg/day).</p>
                </div>
                <div class="guide-item">
                    <h4>Milk Fat %</h4>
                    <p>Butterfat percentage (e.g. 4.5%).</p>
                </div>
            </div>
            """)

        # 3. PREDICTION INPUT SECTION
        with gr.Column(elem_classes=["glass-card"]):
            gr.HTML("<div class='section-hdr-title'>📝 Input Animal Attributes</div>")
            
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
                        placeholder="Describe coat color, horns, hump...",
                        value="Reddish brown coat, white speckles, pendulous ears, prominent broad forehead.",
                        lines=3
                    )
                with gr.Column():
                    features_input = gr.TextArea(
                        label="Special Features",
                        placeholder="Describe heat tolerance, temperament...",
                        value="Extremely high heat tolerance, resistant to diseases, docile temperament.",
                        lines=3
                    )

            predict_btn = gr.Button("Predict Breed", elem_classes=["predict-btn"])

        # 4. RESULT SECTION
        with gr.Column(elem_classes=["glass-card"]):
            gr.HTML("<div class='section-hdr-title'>📊 AI Identification Results</div>")
            
            res_summary_out = gr.HTML("""
            <div style="text-align:center; padding: 12px; color: #0f172a; font-weight:800;">
                Fill details above and click 'Predict Breed'.
            </div>
            """)
            
            top3_out = gr.HTML()
            dossier_out = gr.HTML()
            status_out = gr.HTML(visible=False)

        # 5. HOW THE SYSTEM WORKS SECTION
        with gr.Column(elem_classes=["glass-card"]):
            gr.HTML("""
            <div class="section-hdr-title">⚙️ How the System Works</div>
            <div>
                <div class="timeline-step">
                    <div class="step-num">1</div>
                    <div><strong style="color: #14532d;">Input:</strong> User enters animal parameters.</div>
                </div>
                <div class="timeline-step">
                    <div class="step-num">2</div>
                    <div><strong style="color: #14532d;">TF-IDF Vectorization:</strong> Converts input into mathematical vectors.</div>
                </div>
                <div class="timeline-step">
                    <div class="step-num">3</div>
                    <div><strong style="color: #14532d;">Cosine Similarity:</strong> Calculates distance against dataset breeds.</div>
                </div>
                <div class="timeline-step">
                    <div class="step-num">4</div>
                    <div><strong style="color: #14532d;">Output:</strong> Top 3 matched breeds displayed with percentage match.</div>
                </div>
            </div>
            """)

        # 6. ABOUT THE AI MODEL
        with gr.Column(elem_classes=["glass-card"]):
            status_badge = "<span style='color:#16a34a; font-weight:900;'>✓ Loaded Successfully</span>" if MODEL_LOADED else f"<span style='color:#b91c1c; font-weight:900;'>Error ({MODEL_STATUS_MSG})</span>"
            
            gr.HTML(f"""
            <div class="section-hdr-title">🤖 About the AI Model</div>
            <p style="margin-bottom: 12px; color:#0f172a; font-weight:800;"><strong>Status:</strong> {status_badge}</p>
            <div class="guide-grid">
                <div class="guide-item">
                    <h4>Technique</h4>
                    <p>TF-IDF Vectorization</p>
                </div>
                <div class="guide-item">
                    <h4>Algorithm</h4>
                    <p>Cosine Similarity</p>
                </div>
                <div class="guide-item">
                    <h4>Type</h4>
                    <p>Content-based Identification</p>
                </div>
            </div>
            """)

        # 7. FOOTER SECTION WITH UPDATED SOCIAL LINKS
        with gr.Column(elem_classes=["glass-card", "footer-card"]):
            gr.HTML("""
            <div style="font-size: 1.2rem; font-weight: 900; color: #14532d;">Developer: Prachi Valecha</div>
            <div style="font-size: 0.88rem; color: #0f172a; font-weight: 800; margin-top: 4px; margin-bottom: 12px;">
                Bachelor of Computer Applications (BCA)<br>
                Panipat Institute of Engineering and Technology
            </div>
            <div style="margin-bottom: 10px;">
                <a href="https://github.com/prachivalecha" target="_blank" class="footer-social-btn">
                    <svg width="16" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path></svg>
                    GitHub Profile
                </a>
                <a href="https://www.linkedin.com/in/prachi-valecha-a21898322?utm_source=share_via&utm_content=profile&utm_medium=member_ios" target="_blank" class="footer-social-btn">
                    <svg width="16" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path><rect x="2" y="9" width="4" height="12"></rect><circle cx="4" cy="4" r="2"></circle></svg>
                    LinkedIn Profile
                </a>
            </div>
            <div style="font-size: 0.82rem; color: #475569; font-weight: 700;">
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
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, css=custom_css)
