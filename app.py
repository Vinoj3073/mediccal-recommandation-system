import streamlit as st
import pandas as pd
import numpy as np
import pickle


# ======================
# Load model and data
# ======================
svc = pickle.load(open(r"C:\Users\manoj\Desktop\MY PROJECTS\CP2\Medical Recomendation system\svc.pkl", "rb"))

description = pd.read_csv(r"C:\Users\manoj\Desktop\MY PROJECTS\CP2\Medical Recomendation system\description.csv")
precautions = pd.read_csv(r"C:\Users\manoj\Desktop\MY PROJECTS\CP2\Medical Recomendation system\precautions_df.csv")
workout = pd.read_csv(r"C:\Users\manoj\Desktop\MY PROJECTS\CP2\Medical Recomendation system\workout_df.csv")
diets = pd.read_csv(r"C:\Users\manoj\Desktop\MY PROJECTS\CP2\Medical Recomendation system\diets.csv")
medications = pd.read_csv(r"C:\Users\manoj\Desktop\MY PROJECTS\CP2\Medical Recomendation system\medications.csv")

# Symptoms dictionary (from jupyter notebook)
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'v': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26,
      'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 
      'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 
      'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94,
        'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 
        'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
        }


# Diseases mapping (from your notebook)
diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis',
    14: 'Drug Reaction', 33: 'Peptic ulcer disease', 1: 'AIDS', 12: 'Diabetes',
    17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension', 30: 'Migraine',
    7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice',
    29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid',
    40: 'Hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C',
    21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis',
    36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia',
    13: 'Piles', 18: 'Heart attack', 39: 'Varicose veins',
    26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia',
    31: 'Osteoarthritis', 5: 'Arthritis', 2: 'Acne',
    38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'
}

# ======================
# Helper Functions
# ======================
def predict_disease(symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    prediction = svc.predict([input_vector])[0]
    return diseases_list.get(prediction, "Unknown Disease")

def get_details(disease):
    desc = description[description['Disease'] == disease]['Description'].values
    pre = precautions[precautions['Disease'] == disease].values
    med = medications[medications['Disease'] == disease]['Medication'].values
    die = diets[diets['Disease'] == disease]['Diet'].values
    wrk = workout[workout['disease'] == disease]['workout'].values

    return desc, pre, med, die, wrk


# Page Config
# ======================
st.set_page_config(page_title="Medical Recommendation System", page_icon="💊", layout="wide")

# ======================
# Landing Page Layout
# ======================
# Big Title
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1; font-size: 50px;'>💊 Medical Recommendation System</h1>", 
    unsafe_allow_html=True
)
# Quote Line
st.markdown(
    "<h4 style='text-align: center; color: gray; font-style: italic;'>\"Your health is our priority.\"</h4>",
    unsafe_allow_html=True
)
st.image(
    r"C:\Users\manoj\Desktop\MY PROJECTS\CP2\Medical Recomendation system\future-trends-in-ai-in-healthcare25.jpg",
    width=400
)




# Extra spacing
st.write("")
st.write("")

# ======================
# Sidebar Navigation + Theme Toggle
# ======================
st.sidebar.image(r"C:\Users\manoj\Desktop\MY PROJECTS\CP2\Medical Recomendation system\LOGO.png",use_container_width=True)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "🔎 Predict", "ℹ️ About", "📞 Contact"])

dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=False)

# ======================
# Custom CSS for Light & Dark Mode
# ======================
if dark_mode:
    st.markdown("""
        <style>
        body {background-color: #0e1117; color: #fafafa;}
        .main {background-color: #0e1117; color: #fafafa;}
        .title {text-align: center; font-size: 40px; font-weight: bold; color: #1db954; margin-bottom: 5px;}
        .subtitle {text-align: center; font-size: 18px; color: #ddd; margin-bottom: 20px;}
        .section {padding: 15px; background-color: #161a23; border-radius: 12px; box-shadow: 0px 2px 10px rgba(0,0,0,0.4); margin-bottom: 20px;}
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body {background-color: #f9f9f9; color: #000;}
        .main {background-color: #f9f9f9;}
        .title {text-align: center; font-size: 40px; font-weight: bold; color: #0073e6; margin-bottom: 5px;}
        .subtitle {text-align: center; font-size: 18px; color: #555; margin-bottom: 20px;}
        .section {padding: 15px; background-color: #ffffff; border-radius: 12px; box-shadow: 0px 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px;}
        </style>
    """, unsafe_allow_html=True)

# ======================
# Pages
# ======================
if page == "🏠 Home":
    st.markdown('<p class="title">💊 Medical Recommendation System</p>', unsafe_allow_html=True)
    st.image(r"C:\\Users\\manoj\\Desktop\\MY PROJECTS\\CP2\\Medical Recomendation system\\ai-technologies24.jpg", width=300)

    st.markdown('<p class="subtitle">AI-powered system to predict diseases and recommend treatments</p>', unsafe_allow_html=True)

    st.markdown(
    """
    <div style='background-color:#F8F9F9; padding:20px; border-radius:12px;'>
        <h2 style='color:#2E86C1; text-align:center;'>✨ Features of Medical Recommendation System</h2>
        <ul style="font-size:18px; color:#34495E; line-height:1.8;">
            <li>🔹 <b>Symptom-based Disease Prediction</b> – Get instant predictions using trained ML models.</li>
            <li>🔹 <b>Interactive User Interface</b> – Simple, clean, and modern UI with Streamlit.</li>
            <li>🔹 <b>Multi-Model Support</b> – Uses SVM, Random Forest, and Gradient Boosting classifiers.</li>
            <li>🔹 <b>Accuracy Metrics</b> – Provides accuracy score and confusion matrix for model evaluation.</li>
            <li>🔹 <b>Personalized Health Recommendations</b> – Tailored suggestions based on selected symptoms.</li>
            <li>🔹 <b>Visual Insights</b> – Easy-to-read charts and graphs for better understanding.</li>
            <li>🔹 <b>Secure & Scalable</b> – Built with modular ML pipeline for easy extension.</li>
            <li>🔹 <b>Lightweight & Fast</b> – Runs smoothly even on low-resource systems.</li>
            <li>🔹 <b>Future Ready</b> – Can be extended with Deep Learning, NLP, and IoT healthcare devices.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

elif page == "🔎 Predict":
    st.markdown('<p class="title">🔎 Disease Prediction</p>', unsafe_allow_html=True)
    st.image(r"C:\Users\manoj\Desktop\MY PROJECTS\CP2\Medical Recomendation system\download.jpeg", width=400)
    st.markdown('<p class="subtitle">Select your symptoms to get predictions</p>', unsafe_allow_html=True)

    # Age input
    age = st.number_input("Enter your Age:", min_value=1, max_value=120, step=1)

    # Symptoms input
    selected_symptoms = st.multiselect("Choose your symptoms:", list(symptoms_dict.keys()))

    # Predict button
    if st.button("Predict"):
        if age < 18:
            st.error("⚠️ You must be 18 or older to use this system.")
        elif not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            disease = predict_disease(selected_symptoms)
            st.success(f"Predicted Disease: **{disease}**")

            desc, pre, med, die, wrk = get_details(disease)

            st.markdown("<div class='section'><h3>📝 Description</h3>", unsafe_allow_html=True)
            st.write(desc[0] if len(desc) else "No description available.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='section'><h3>💊 Medications</h3>", unsafe_allow_html=True)
            st.write(", ".join(med) if len(med) else "No medications found.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='section'><h3>🥗 Diet</h3>", unsafe_allow_html=True)
            st.write(", ".join(die) if len(die) else "No diet recommendations.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='section'><h3>🏋️ Workout</h3>", unsafe_allow_html=True)
            st.write(", ".join(wrk) if len(wrk) else "No workout suggestions.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='section'><h3>⚠️ Precautions</h3>", unsafe_allow_html=True)
            if len(pre):
                for p in pre[0]:
                    st.write("- ", p)
            else:
                st.write("No precautions available.")
            st.markdown("</div>", unsafe_allow_html=True)

elif page == "ℹ️ About":
    st.markdown('<p class="title">ℹ️ About This Project</p>', unsafe_allow_html=True)
    st.image(r"C:\Users\manoj\Desktop\MY PROJECTS\CP2\Medical Recomendation system\download (2).jpeg", width=400)
    st.markdown("ℹ️ About This Project")
    st.markdown("""
The **Medical Recommendation System** is an AI-powered healthcare application that helps patients 
and healthcare professionals by predicting possible diseases based on symptoms.  
It combines **Data Science, Machine Learning, and Healthcare Knowledge** to provide early insights 
and guide patients towards timely medical attention.  

---

### 🎯 Mission
Our mission is to make **healthcare more accessible and data-driven** by offering quick, 
AI-assisted recommendations that can support decision-making for both doctors and patients.

---

### 🌟 Key Benefits
- ✅ **Early Disease Detection** – Predicts possible illnesses at an early stage  
- ✅ **User-Friendly** – Simple and interactive interface for non-technical users  
- ✅ **Time-Saving** – Reduces guesswork by narrowing down potential conditions  
- ✅ **Educational** – Increases awareness of health symptoms and their importance  
- ✅ **Customizable** – Can be extended to support new symptoms and diseases  

---

### 🛠️ Technology Stack
- **Python** – Core programming language  
- **Scikit-learn** – Machine Learning models (SVM, Random Forest, Gradient Boosting)  
- **Pandas & NumPy** – Data handling and preprocessing  
- **Streamlit** – Interactive web application framework  

---

### 🚀 Future Scope
- 🔹 Integration with **real patient records & electronic health systems**  
- 🔹 Incorporating **Natural Language Processing (NLP)** for free-text symptom inputs  
- 🔹 Mobile app version for wider accessibility  
- 🔹 Integration with **IoT healthcare devices** for real-time monitoring  
- 🔹 Continuous improvement with **deep learning models** for higher accuracy  

---

This project is created as part of a **Data Science & Machine Learning initiative**  
with the vision of leveraging technology to **improve global healthcare accessibility**.
""", unsafe_allow_html=True)

elif page == "📞 Contact":
    st.markdown('<p class="title">📞 Contact</p>', unsafe_allow_html=True)
    st.image(r"C:\Users\manoj\Desktop\MY PROJECTS\CP2\Medical Recomendation system\download (1).jpeg",width=400)
    st.markdown("<div class='section'>"
                "<b>Developer:</b> VINOJ<br>"
                "<b>Email:</b> VINOJSUDHAVENI@GMAIL.com<br>"
                "<b>GitHub:</b> https://github.com/VINOJ3073<br>"
                "</div>", unsafe_allow_html=True)