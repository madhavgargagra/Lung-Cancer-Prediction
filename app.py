import os
import pickle
import numpy as np
import pandas as pd
import re
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder='templates', static_folder='static')

# Define the exact feature names and order expected by the model
FEATURE_NAMES = [
    'Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy', 
    'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease', 'Balanced Diet', 
    'Obesity', 'Smoking', 'Passive Smoker', 'Chest Pain', 'Coughing of Blood', 
    'Fatigue', 'Weight Loss', 'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty', 
    'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring'
]

# Global variables for model and preprocessors
model = None
scaler = None
classes = ['High', 'Low', 'Medium']  # Default fallback classes

def load_assets():
    global model, scaler, classes
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(base_dir, 'model.pkl')
    scaler_path = os.path.join(base_dir, 'scaler.pkl')
    classes_path = os.path.join(base_dir, 'classes.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model assets (model.pkl or scaler.pkl) are missing. Please run train_and_save_model.py first.")
        
    print("Loading model and scaler assets...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    if os.path.exists(classes_path):
        with open(classes_path, 'rb') as f:
            classes = pickle.load(f).tolist()
            
    print(f"Loaded assets. Classes: {classes}")

# Load assets immediately on start
try:
    load_assets()
except Exception as e:
    print(f"Error loading assets: {e}")

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'success': True}), 200
        
    global model, scaler, classes
    if model is None or scaler is None:
        return jsonify({
            'success': False, 
            'error': 'Model is not loaded. Check server logs.'
        }), 500
        
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No input data provided'}), 400
            
        # Extract features and format them in the exact order and names
        input_data = {}
        missing_features = []
        for feature in FEATURE_NAMES:
            if feature not in data:
                # Age defaults to 40, Gender to 1, others to 1 (lowest scale)
                if feature == 'Age':
                    input_data[feature] = 40
                elif feature == 'Gender':
                    input_data[feature] = 1
                else:
                    input_data[feature] = 1
                missing_features.append(feature)
            else:
                input_data[feature] = float(data[feature])
                
        if missing_features:
            print(f"Warning: Missing features defaulted: {missing_features}")
            
        # Create a 1-row DataFrame matching the column list
        features_df = pd.DataFrame([input_data])[FEATURE_NAMES]
        
        # Scale the features
        features_scaled = scaler.transform(features_df)
        
        # Perform prediction and get probability distribution
        pred_idx = int(model.predict(features_scaled)[0])
        pred_prob = model.predict_proba(features_scaled)[0]
        
        predicted_class = classes[pred_idx]
        confidence = float(pred_prob[pred_idx])
        
        # Structure the breakdown of probabilities
        prob_breakdown = {}
        for idx, class_name in enumerate(classes):
            prob_breakdown[class_name] = float(pred_prob[idx])
            
        # Determine recommendations based on threat level
        recommendations = []
        if predicted_class == 'High':
            recommendations = [
                "URGENT: Please consult a pulmonologist or oncologist immediately for professional screening (e.g., Low-Dose CT scan).",
                "Minimize any exposure to primary risk factors such as active smoking and passive smoke.",
                "Seek advice on pulmonary rehabilitation or specialized tests if experiencing chronic coughing or chest pain."
            ]
        elif predicted_class == 'Medium':
            recommendations = [
                "Schedule a routine health check-up and discuss lung health screening options with a physician.",
                "Implement positive lifestyle changes: cease smoking, increase physical activity, and eat an antioxidant-rich diet.",
                "Monitor any persistent symptoms like wheezing, persistent cold, or fatigue, and report them to a doctor."
            ]
        else:
            recommendations = [
                "Continue maintaining a healthy lifestyle, a balanced diet, and regular exercise.",
                "Avoid environmental hazards such as air pollution, passive smoking, and dust exposures where possible.",
                "Schedule standard periodic health check-ups."
            ]

        # Calculate local contributions for features relative to risk probability (Medium + High threat = 1 - Low threat)
        try:
            low_idx = classes.index('Low')
        except ValueError:
            low_idx = 1
            
        original_risk = 1.0 - float(pred_prob[low_idx])
        contributions = []
        
        for feature in FEATURE_NAMES:
            val = input_data[feature]
            
            # Baseline value (healthy reference point)
            if feature == 'Age':
                baseline = 40.0
            elif feature == 'Gender':
                baseline = 1.0  # male
            elif feature == 'Balanced Diet':
                baseline = 7.0  # High diet is protective
            else:
                baseline = 1.0  # Low exposure/symptom is healthy
                
            if val == baseline:
                continue
                
            # Create a copy with the feature set to baseline
            perturbed_data = input_data.copy()
            perturbed_data[feature] = baseline
            perturbed_df = pd.DataFrame([perturbed_data])[FEATURE_NAMES]
            perturbed_scaled = scaler.transform(perturbed_df)
            
            # Get model's prediction probability on this perturbed vector
            perturbed_prob = model.predict_proba(perturbed_scaled)[0]
            perturbed_risk = 1.0 - float(perturbed_prob[low_idx])
            
            # Contribution represents the change in elevated risk probability
            # Positive value = this feature pushed the overall risk higher (risk contributor, red)
            # Negative value = this feature pushed the overall risk lower (protective factor, green)
            contrib = original_risk - perturbed_risk
            
            if abs(contrib) > 0.002:  # Only report meaningful shifts
                contributions.append({
                    'feature': feature,
                    'value': val,
                    'baseline': baseline,
                    'contribution': contrib
                })
                
        # Sort contributions by absolute magnitude (highest impact first)
        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)

        # Return results
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': prob_breakdown,
            'recommendations': recommendations,
            'contributions': contributions
        })
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({
            'success': False,
            'error': f"Prediction error: {str(e)}"
        }), 500

def de_space(s):
    if not s:
        return ""
    s = s.strip()
    # Count spaces that are single spaces between alphanumeric or word boundaries
    single_char_spaces = len(re.findall(r'\b\w\s\b', s)) + len(re.findall(r'\b\s\w\b', s))
    if len(s) > 3 and single_char_spaces > len(s) * 0.25:
        placeholder = "___WORD_SEP___"
        # Preserve word separation (usually 2 or more spaces in spaced-out text)
        temp = re.sub(r'\s{2,}', placeholder, s)
        # Remove remaining single spaces
        temp = temp.replace(" ", "")
        # Restore actual word separations as single spaces
        res = temp.replace(placeholder, " ")
        return res
    return s

def parse_pdf_text(text):
    data = {}
    
    def clean_for_matching(s):
        return re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        
    # 1. Match PatientName and PatientID in raw text (like footers where text is normal)
    title_match = re.search(r'\[(.*?)\]\s*([^\n\-]+)\s*-\s*Mady\s*PathLabs', text, re.IGNORECASE)
    if title_match:
        data['PatientID'] = de_space(title_match.group(1))
        data['PatientName'] = de_space(title_match.group(2))
    else:
        # Match spaced-out version of header: [ 1 2 3 ] M a d y - M a d y P a t h L a b s
        title_spaced = re.search(r'\[([^\]]+)\]([^\n\-]+)-\s*M\s*a\s*d\s*y\s*P\s*a\s*t\s*h\s*L\s*a\s*b\s*s', text, re.IGNORECASE)
        if title_spaced:
            data['PatientID'] = de_space(title_spaced.group(1))
            data['PatientName'] = de_space(title_spaced.group(2))

    # Fallback individual field matches on raw text
    if 'PatientName' not in data or not data['PatientName']:
        name_match = re.search(r'Patient\s*Name\s*:\s*([^\n]+)', text, re.IGNORECASE)
        if name_match:
            name_val = name_match.group(1)
            name_val = re.split(r'Patient\s*ID', name_val, flags=re.IGNORECASE)[0]
            data['PatientName'] = de_space(name_val)
            
    if 'PatientID' not in data or not data['PatientID']:
        id_match = re.search(r'Patient\s*ID\s*/\s*MRN\s*:\s*([^\n]+)', text, re.IGNORECASE)
        if id_match:
            id_val = id_match.group(1)
            id_val = re.split(r'Report\s*Date', id_val, flags=re.IGNORECASE)[0]
            data['PatientID'] = de_space(id_val)

    # 2. Normalize text by removing all whitespace characters to extract fields from body
    clean_lines = []
    for line in text.split('\n'):
        clean_line = re.sub(r'[\s\xa0\u2000-\u200a\u202f\u205f\u3000]', '', line)
        if clean_line:
            clean_lines.append(clean_line)
    clean_text = "\n".join(clean_lines)

    # If fallback Name/ID are still missing, try clean text
    if 'PatientName' not in data or not data['PatientName']:
        name_clean = re.search(r'PatientName:([^:\n]+?)(?:PatientID|$)', clean_text, re.IGNORECASE)
        if name_clean:
            data['PatientName'] = name_clean.group(1)
            
    if 'PatientID' not in data or not data['PatientID']:
        id_clean = re.search(r'PatientID/MRN:([^:\n]+?)(?:ReportDate|Age|Gender|$)', clean_text, re.IGNORECASE)
        if id_clean:
            data['PatientID'] = id_clean.group(1)

    # 3. Extract Age and Gender from clean text
    age_match = re.search(r'Age:(\d+)', clean_text, re.IGNORECASE)
    if age_match:
        data['Age'] = int(age_match.group(1))
    else:
        age_match2 = re.search(r'Age(\d+)', clean_text, re.IGNORECASE)
        if age_match2:
            data['Age'] = int(age_match2.group(1))
            
    gender_match = re.search(r'Gender:(Male|Female)', clean_text, re.IGNORECASE)
    if gender_match:
        gender_str = gender_match.group(1).lower()
        data['Gender'] = 1 if gender_str == 'male' else 2
    else:
        gender_match2 = re.search(r'Gender(Male|Female)', clean_text, re.IGNORECASE)
        if gender_match2:
            gender_str = gender_match2.group(1).lower()
            data['Gender'] = 1 if gender_str == 'male' else 2

    # 4. Extract all 21 sliders from clean text
    feature_display_map = {
        'Air Pollution': 'AirPollutionExposure',
        'Alcohol use': 'AlcoholConsumption',
        'Dust Allergy': 'DustAllergySeverity',
        'OccuPational Hazards': 'OccupationalHazards',
        'Genetic Risk': 'FamilyGeneticRisk',
        'chronic Lung Disease': 'ChronicLungDiseaseHistory',
        'Balanced Diet': 'BalancedDietQuality',
        'Obesity': 'ObesityLevel',
        'Smoking': 'ActiveSmoking',
        'Passive Smoker': 'PassiveSmokingExposure',
        'Chest Pain': 'ChestPainSeverity',
        'Coughing of Blood': 'CoughingofBlood',
        'Fatigue': 'FatigueLevel',
        'Weight Loss': 'WeightLossSeverity',
        'Shortness of Breath': 'ShortnessofBreath',
        'Wheezing': 'WheezingFrequency',
        'Swallowing Difficulty': 'DifficultySwallowing',
        'Clubbing of Finger Nails': 'ClubbingofFingerNails',
        'Frequent Cold': 'FrequentColds/Infections',
        'Dry Cough': 'DryCoughSeverity',
        'Snoring': 'HeavySnoring'
    }
    
    for feature_key, clean_label in feature_display_map.items():
        pattern = re.escape(clean_label) + r'(\d+)'
        match = re.search(pattern, clean_text, re.IGNORECASE)
        if match:
            data[feature_key] = int(match.group(1))
        else:
            pattern2 = re.escape(clean_label) + r':(\d+)'
            match2 = re.search(pattern2, clean_text, re.IGNORECASE)
            if match2:
                data[feature_key] = int(match2.group(1))

    # --- EXTRACTION OF DIAGNOSTIC REPORT OUTCOMES ---

    # 1. Action Recommendations (Threat Level, Confidence, Probability Distribution)
    pred_match = re.search(r'Threatlevel:(LOW|MEDIUM|HIGH)', clean_text, re.IGNORECASE)
    if pred_match:
        data['ThreatLevel'] = pred_match.group(1).capitalize()
    else:
        summary_pos = clean_text.find("DiagnosticSummary")
        if summary_pos != -1:
            sub_text = clean_text[summary_pos:summary_pos+300]
            threat_match = re.search(r'\b(LOW|MEDIUM|HIGH)\b', sub_text, re.IGNORECASE)
            if threat_match:
                data['ThreatLevel'] = threat_match.group(1).capitalize()
            else:
                data['ThreatLevel'] = "Low"
        else:
            data['ThreatLevel'] = "Low"

    # Confidence
    conf_match = re.search(r'Confidence:?(\d+)%', clean_text, re.IGNORECASE)
    if not conf_match:
        conf_match = re.search(r'(\d+)%\nCONFIDENCE', clean_text, re.IGNORECASE)
    
    if conf_match:
        data['Confidence'] = float(conf_match.group(1)) / 100.0
    else:
        data['Confidence'] = 0.50

    # Probability Distribution
    data['Probability_Low'] = 0.0
    data['Probability_Medium'] = 0.0
    data['Probability_High'] = 0.0
    
    low_match = re.search(r'LowThreatLevel(\d+)%', clean_text, re.IGNORECASE)
    if low_match:
        data['Probability_Low'] = float(low_match.group(1)) / 100.0
    medium_match = re.search(r'MediumThreatLevel(\d+)%', clean_text, re.IGNORECASE)
    if medium_match:
        data['Probability_Medium'] = float(medium_match.group(1)) / 100.0
    high_match = re.search(r'HighThreatLevel(\d+)%', clean_text, re.IGNORECASE)
    if high_match:
        data['Probability_High'] = float(high_match.group(1)) / 100.0

    # 2. Action Recommendations List (Page 0)
    recs = []
    parts = re.split(r'|\uf05a|•|\[BULLET\]', text)
    if len(parts) > 1:
        first_part = parts[0]
        conf_idx_match = re.search(r'C\s*O\s*N\s*F\s*I\s*D\s*E\s*N\s*C\s*E', first_part, re.IGNORECASE)
        if conf_idx_match:
            rec1_raw = first_part[conf_idx_match.end():]
            rec1 = de_space(rec1_raw).strip()
            rec1 = re.sub(r'^[^\w]+', '', rec1).strip()
            if rec1 and not any(stop in clean_for_matching(rec1) for stop in ["diagnosticsummary", "confidence"]):
                recs.append(rec1)
        
        for p in parts[1:]:
            cleaned_p = de_space(p).strip()
            cleaned_p = re.sub(r'^[^\w]+', '', cleaned_p).strip()
            if cleaned_p:
                if any(stop in clean_for_matching(cleaned_p) for stop in ["madypathlabs", "127001", "mypersonalactionplan", "laboratoryreport"]):
                    continue
                recs.append(cleaned_p)
                
    data['Recommendations'] = recs

    # 3. My Personal Action Plan (Page 1)
    action_plan_goals = []
    candidate_goals = [
        "Commit to active smoking cessation program to let lung tissue recover.",
        "Minimize exposure to secondhand smoke in household and social settings.",
        "Use N95 respirators on high-smog days and run a HEPA air filter in main rooms.",
        "Ensure workplace compliance: always wear protective respiratory gear on site.",
        "Reduce weekly alcohol consumption to boost overall immune response.",
        "Enrich daily diet with green vegetables and antioxidant-rich organic foods.",
        "Adopt structured portion control and light cardiovascular exercise to lower BMI.",
        "Schedule routine spirometry and pulmonary monitoring check-ups with your doctor.",
        "Schedule routine spirometry and pulmonary monitoring checkups with your doctor.",
        "Engage in 30 minutes of daily moderate activity to support pulmonary capacity.",
        "Maintain a regular sleep schedule to optimize daily energy and cellular repair.",
        "Schedule an annual routine health check-up for comprehensive respiratory evaluation."
    ]
    
    ap_idx = clean_text.find("MyPersonalActionPlan")
    if ap_idx != -1:
        ap_sub = clean_text[ap_idx:]
        cont_idx = ap_sub.find("KeyRiskContributions")
        if cont_idx != -1:
            ap_sub = ap_sub[:cont_idx]
        
        ap_sub_clean = clean_for_matching(ap_sub)
        
        for goal in candidate_goals:
            goal_clean = clean_for_matching(goal)
            if goal_clean in ap_sub_clean:
                final_goal = goal
                if "checkups" in goal:
                    final_goal = "Schedule routine spirometry and pulmonary monitoring check-ups with your doctor."
                if final_goal not in action_plan_goals:
                    action_plan_goals.append(final_goal)
                    
    data['ActionPlan'] = action_plan_goals

    # 4. Key Risk Contributions (Page 1)
    risk_contributions = []
    candidates_contributions = {
        'Smoking': ['activesmoking', 'smoking'],
        'Passive Smoker': ['passivesmokingexposure', 'passivesmoker'],
        'Air Pollution': ['airpollutionexposure', 'airpollution'],
        'Alcohol use': ['alcoholconsumption', 'alcoholuse'],
        'Dust Allergy': ['dustallergyseverity', 'dustallergy'],
        'OccuPational Hazards': ['occupationalhazards'],
        'Genetic Risk': ['familygeneticrisk', 'geneticrisk'],
        'chronic Lung Disease': ['chroniclungdiseasehistory', 'chroniclungdisease'],
        'Balanced Diet': ['balanceddietquality', 'balanceddiet'],
        'Obesity': ['obesitylevel', 'obesity'],
        'Chest Pain': ['chestpainseverity', 'chestpain'],
        'Coughing of Blood': ['coughingofblood', 'hemoptysis'],
        'Fatigue': ['fatiguelevel', 'fatigue'],
        'Weight Loss': ['weightlossseverity', 'weightloss'],
        'Shortness of Breath': ['shortnessofbreath', 'dyspnea'],
        'Wheezing': ['wheezingfrequency', 'wheezing'],
        'Swallowing Difficulty': ['difficultyswallowing', 'swallowingdifficulty'],
        'Clubbing of Finger Nails': ['clubbingoffingernails', 'nailclubbing'],
        'Frequent Cold': ['frequentcoldsinfections', 'frequentcold'],
        'Dry Cough': ['drycoughseverity', 'drycough'],
        'Snoring': ['heavysnoring', 'snoring']
    }

    krc_idx = clean_text.find("KeyRiskContributions")
    if krc_idx != -1:
        krc_sub = clean_text[krc_idx:]
        params_idx = krc_sub.find("SubmittedPatientParameters")
        if params_idx != -1:
            krc_sub = krc_sub[:params_idx]
            
        krc_sub_clean = clean_for_matching(krc_sub)
        
        for feat_key, variations in candidates_contributions.items():
            for var in variations:
                if var in krc_sub_clean:
                    if feat_key not in risk_contributions:
                        risk_contributions.append(feat_key)
                    break
    data['RiskContributions'] = risk_contributions

    # 5. Clinician's Remarks & Consultation Notes
    lines = text.split('\n')
    in_remarks = False
    remark_lines = []
    
    stop_words_clean = ["medicallabtechnologist", "madypathlabsstaff", "chiefpathologist", 
                        "draristhornemd", "madypathlabs", "verified", "page2"]
    
    for line in lines:
        line_clean = clean_for_matching(line)
        if not in_remarks:
            if "cliniciansremarks" in line_clean:
                in_remarks = True
                continue
        else:
            if any(stop in line_clean for stop in stop_words_clean) or "notice" in line_clean or "disclaimer" in line_clean:
                break
            if line.strip():
                remark_lines.append(de_space(line).strip())
                
    data['ClinicianRemarks'] = "\n".join(remark_lines).strip()
                
    return data


@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part in request'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400
        
    if file and file.filename.lower().endswith('.pdf'):
        try:
            import pypdf
            reader = pypdf.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
                
            parsed_data = parse_pdf_text(text)
            
            return jsonify({
                'success': True,
                'data': parsed_data
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f"Failed to parse PDF file: {str(e)}"
            }), 500
    else:
        return jsonify({'success': False, 'error': 'Invalid file format. Please upload a PDF.'}), 400



if __name__ == '__main__':
    # Load assets before starting the server
    try:
        load_assets()
    except Exception as e:
        print(f"Error reloading assets on main: {e}")
    app.run(debug=True, port=5000)
