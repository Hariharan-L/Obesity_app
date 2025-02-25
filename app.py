import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Create and train model if not present
if not os.path.exists('obesity_model.pkl'):
    df = pd.read_csv('train.csv')
    
    # Preprocessing
    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing for numeric and categorical features
    numeric_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    numeric_transformer = StandardScaler()

    categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC',
                            'SMOKE', 'SCC', 'CALC', 'MTRANS']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Create pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train model
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, 'obesity_model.pkl')

# Load model
model = joblib.load('obesity_model.pkl')

def calculate_bmi(height, weight):
    return weight / ((height/100) ** 2)

# Define BMI ranges for each class.
bmi_ranges = {
    "Insufficient_Weight": (0, 18.5),
    "Normal_Weight": (18.5, 25),
    "Overweight_Level_I": (25, 30),
    "Overweight_Level_II": (30, 35),
    "Obesity_Type_I": (35, 40),
    "Obesity_Type_II": (40, 50),
    "Obesity_Type_III": (50, float('inf'))
}

def bmi_based_distribution(bmi, classes):
    distribution = np.zeros(len(classes))
    for i, cls in enumerate(classes):
        if cls in bmi_ranges:
            low, high = bmi_ranges[cls]
            if low <= bmi < high:
                distribution[i] = 1.0
    if distribution.sum() == 0:
        distribution = np.ones(len(classes)) / len(classes)
    return distribution

def get_suggestions(inputs, prediction):
    suggestions = []
    if inputs['SMOKE'] == 'yes':
        suggestions.append("🚭 Quit smoking to improve overall health")
        
    if inputs['FCVC'] < 2:
        suggestions.append(f"🥦 Increase vegetable consumption to at least 3 times daily (Current: {inputs['FCVC']})")
    
    if inputs['Gender'] == 'Female' and inputs['CH2O'] < 2:
        suggestions.append(f"💧 Drink at least {2 - inputs['CH2O']} more liters of water daily")
    
    if inputs['Gender'] == 'Male' and inputs['CH2O'] < 3:
        suggestions.append(f"💧 Drink at least {3 - inputs['CH2O']} more liters of water daily")
    
    if inputs['CAEC'] in ['Sometimes', 'Frequently']:
        suggestions.append("Don't intake any snacks")
        
    if prediction != 'Insufficient_Weight':
        if inputs['FAVC'] == 'yes':
            suggestions.append("🍔 Reduce consumption of high-caloric foods")
        if inputs['FAF'] < 2:
            suggestions.append("🏋️ Increase physical activity to at least 3 days a week")
        
    if inputs['CALC'] in ['Sometimes', 'Frequently']:
        suggestions.append("🍷 Reduce alcohol consumption")
    
    if inputs['TUE'] == 3:
        suggestions.append("📵 Avoid using mobile phones Frequently")
    elif inputs['TUE'] == 2:
        suggestions.append("📵 Reduce your mobile phone usage")
    
    bmi = calculate_bmi(inputs['Height'], inputs['Weight'])
    if bmi < 18.5:
        suggestions.append(f"⚖️ Your BMI is {bmi:.1f} - Your weight appears insufficient; consider increasing your nutritional intake")
    elif bmi > 25:
        suggestions.append(f"⚖️ Your BMI is {bmi:.1f} - Aim for regular exercise and a balanced diet")
        
    return suggestions

def main():
    st.title("Obesity Risk Prediction")
    st.write("Please provide the following information:")

    with st.form("user_inputs"):
        col1, col2 = st.columns(2)
        with col1:
            # Dropdowns now include a default "Select" option
            gender = st.selectbox("Gender", ['Select', 'Male', 'Female'])
            age = st.number_input("Age", min_value=0, max_value=100, value=0)
            height = st.number_input("Height (cm)", min_value=0, max_value=250, value=0)
            weight = st.number_input("Weight (kg)", min_value=0, max_value=300, value=0)
            family_history = st.selectbox("Family History of Overweight", ['Select', 'yes', 'no'])
            favc = st.selectbox("Frequent High Caloric Food Consumption", ['Select', 'yes', 'no'])
            
        with col2:
            fcvc = st.slider("Vegetable Consumption Frequency (1-3)", 0, 3, 0)
            ncp = st.slider("Number of Main Meals (1-4)", 0, 4, 0)
            caec = st.selectbox("Snacks Between Meals", ['Select', 'no', 'Sometimes', 'Frequently', 'Always'])
            smoke = st.selectbox("Do you smoke?", ['Select', 'yes', 'no'])
            ch2o = st.slider("Daily Water Intake (liters)", 0.0, 3.0, 0.0)
            scc = st.selectbox("Monitor Calories Consumption", ['Select', 'yes', 'no'])
            
        faf = st.slider("Physical Activity Frequency (0-3)", 0, 3, 0)
        tue = st.slider("Time Using Technology Devices (0-3)", 0, 3, 0)
        calc = st.selectbox("Alcohol Consumption", ['Select', 'no', 'Sometimes', 'Frequently'])
        mtrans = st.selectbox("Transportation Used", ['Select', 'Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])
        
        submitted = st.form_submit_button("Predict")
    
    if submitted:
        # Validate that no field is left at its default value
        invalid_fields = []
        if gender == 'Select':
            invalid_fields.append("Gender")
        if age == 0:
            invalid_fields.append("Age")
        if height == 0:
            invalid_fields.append("Height")
        if weight == 0:
            invalid_fields.append("Weight")
        if family_history == 'Select':
            invalid_fields.append("Family History")
        if favc == 'Select':
            invalid_fields.append("High Caloric Food Consumption")
        if fcvc == 0:
            invalid_fields.append("Vegetable Consumption Frequency")
        if ncp == 0:
            invalid_fields.append("Number of Main Meals")
        if caec == 'Select':
            invalid_fields.append("Snacks Between Meals")
        if smoke == 'Select':
            invalid_fields.append("Smoking")
        if ch2o == 0.0:
            invalid_fields.append("Daily Water Intake")
        if scc == 'Select':
            invalid_fields.append("Monitor Calories Consumption")
        if faf == 0:
            invalid_fields.append("Physical Activity Frequency")
        if tue == 0:
            invalid_fields.append("Time Using Technology Devices")
        if calc == 'Select':
            invalid_fields.append("Alcohol Consumption")
        if mtrans == 'Select':
            invalid_fields.append("Transportation")
        
        if invalid_fields:
            st.error(f"Please provide valid inputs for the following fields: {', '.join(invalid_fields)}")
        else:
            input_data = {
                'Gender': gender, 'Age': age, 'Height': height, 'Weight': weight,
                'family_history_with_overweight': family_history, 'FAVC': favc, 'FCVC': fcvc, 'NCP': ncp,
                'CAEC': caec, 'SMOKE': smoke, 'CH2O': ch2o, 'SCC': scc, 'FAF': faf,
                'TUE': tue, 'CALC': calc, 'MTRANS': mtrans
            }
            df_input = pd.DataFrame([input_data])
            
            try:
                # Get model probabilities and prediction
                probabilities = model.predict_proba(df_input)[0]
                classes = model.classes_
                model_prediction = classes[np.argmax(probabilities)]
                
                # Desired fixed order (by severity)
                desired_order = ["Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", 
                                 "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]
                
                # Reorder model probabilities in the fixed order (only include classes present)
                ordered_classes = []
                ordered_probs = []
                for cls in desired_order:
                    if cls in classes:
                        idx = list(classes).index(cls)
                        ordered_classes.append(cls)
                        ordered_probs.append(probabilities[idx])
                
                # Calculate BMI and get BMI-based prediction
                bmi = calculate_bmi(input_data['Height'], input_data['Weight'])
                bmi_distribution = bmi_based_distribution(bmi, classes)
                bmi_prediction = classes[np.argmax(bmi_distribution)]
                
                # Determine final prediction:
                # If the model predicts "Insufficient_Weight", use that.
                # Otherwise, if the model's prediction differs from the BMI-based prediction, favor the BMI prediction.
                if model_prediction == 'Insufficient_Weight':
                    final_prediction = 'Insufficient_Weight'
                elif model_prediction != bmi_prediction:
                    final_prediction = bmi_prediction
                else:
                    final_prediction = model_prediction
                
                # Adjust displayed percentages so the peak is at the final predicted category.
                final_index = ordered_classes.index(final_prediction)
                weights = [1.0 / (1.0 + abs(i - final_index)) for i in range(len(ordered_classes))]
                weights = np.array(weights)
                adjusted_probs = weights / weights.sum()  # normalize
                
                # Mapping for column headings (custom labels)
                mapping = {
                    "Insufficient_Weight": "Under Weight",
                    "Normal_Weight": "Normal Weight",
                    "Overweight_Level_I": "Over Weight1",
                    "Overweight_Level_II": "Over Weight2",
                    "Obesity_Type_I": "Obesity1",
                    "Obesity_Type_II": "Obesity2",
                    "Obesity_Type_III": "Obesity3"
                }
                
                # Build HTML table for displaying adjusted probabilities
                cell_style = ("width:14.28%; height:50px; padding:4px; border:1px solid lightwhite; "
                              "word-wrap: break-word; white-space: normal;")
                html_table = ("<table style='width:100%; table-layout: fixed; text-align:center; "
                              "border-collapse: separate; border-spacing: 0px;'>")
                html_table += "<tr><td style='" + cell_style + "'>Obesity Class</td>"
                for cls in ordered_classes:
                    label = mapping.get(cls, cls).replace(" ", "<br>")
                    html_table += f"<td style='{cell_style}'>{label}</td>"
                html_table += "</tr>"
                html_table += "<tr><td style='" + cell_style + "'>Percentage</td>"
                for prob in (adjusted_probs * 100).round(2):
                    html_table += f"<td style='{cell_style}'>{prob}%</td>"
                html_table += "</tr></table>"
                
                st.markdown(html_table, unsafe_allow_html=True)
                st.success(f"**Predicted Category**: {final_prediction.replace('_', ' ').title()}")
                
                # Provide suggestion for weight change based on prediction
                if final_prediction == "Insufficient_Weight":
                    # Calculate desired weight using the lower bound of normal BMI (18.5)
                    desired_weight = 18.5 * ((input_data['Height'] / 100) ** 2)
                    weight_to_increase = desired_weight - input_data['Weight']
                    if weight_to_increase > 0:
                        st.info(f"Suggestion: Increase your weight by approximately {weight_to_increase:.1f} kg to reach a normal BMI.")
                elif final_prediction not in ["Insufficient_Weight", "Normal_Weight"]:
                    desired_weight = 25 * ((input_data['Height'] / 100) ** 2)
                    weight_to_reduce = input_data['Weight'] - desired_weight
                    if weight_to_reduce > 0:
                        st.info(f"Suggestion: Reduce your weight by approximately {weight_to_reduce:.1f} kg to reach a normal BMI.")
                
                suggestions = get_suggestions(input_data, final_prediction)
                if suggestions:
                    st.subheader("Health Suggestions")
                    for suggestion in suggestions:
                        st.write(f"- {suggestion}")
                        
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()
