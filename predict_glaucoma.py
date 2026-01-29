"""
GLAUCOMA PREDICTION SYSTEM
This script reads IOP sensor data, collects patient information,
and predicts glaucoma using a trained Random Forest model.
"""

import pandas as pd
import numpy as np
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

print("GLAUCOMA PREDICTION SYSTEM")
print("=" * 60)


def compute_final_iop():
    """Read sensor data and compute final IOP using trimmed mean"""
    print("\nStep 1: Computing Final IOP from sensor readings...")
    
    try:
        df = pd.read_csv('sensor_data.csv')
        print(f"   Found {len(df)} sensor readings")
        
        iop_values = df['IOP'].astype(float).tolist()
        
        print(f"\n   All IOP readings: {[f'{x:.2f}' for x in iop_values]}")
        
        iop_sorted = sorted(iop_values)
        trimmed_values = iop_sorted[2:-2] if len(iop_sorted) >= 6 else iop_sorted
        
        final_iop = np.mean(trimmed_values)
        median_iop = np.median(iop_values)
        mean_iop = np.mean(iop_values)
        
        print(f"\n   Statistics:")
        print(f"      Mean IOP: {mean_iop:.2f} mmHg")
        print(f"      Median IOP: {median_iop:.2f} mmHg")
        print(f"      Trimmed Mean IOP: {final_iop:.2f} mmHg")
        
        if final_iop <= 21:
            category = "Normal"
        elif final_iop <= 27:
            category = "Borderline/High Risk"
        elif final_iop <= 40:
            category = "Glaucoma Detected"
        else:
            category = "Critical/Emergency"
        
        print(f"\n   IOP Category: {category}")
        
        return final_iop
        
    except FileNotFoundError:
        print("   Error: sensor_data.csv not found!")
        print("   Please ensure the server has collected 10 readings first.")
        return None
    except Exception as e:
        print(f"   Error reading sensor data: {e}")
        return None


def get_user_input(final_iop):
    """Collect patient clinical data from user"""
    print("\nStep 2: Enter Patient Clinical Information")
    print("-" * 60)
    
    patient_data = {}
    
    patient_data['Intraocular Pressure (IOP)'] = final_iop
    print(f"IOP (from sensor): {final_iop:.2f} mmHg")
    
    while True:
        try:
            age = int(input("\nEnter Age (18-90): "))
            if 18 <= age <= 90:
                patient_data['Age'] = age
                break
            else:
                print("   Please enter age between 18 and 90")
        except ValueError:
            print("   Please enter a valid number")
    
    while True:
        gender = input("\nEnter Gender (Male/Female): ").strip().lower()
        if gender in ['male', 'female']:
            patient_data['Gender_Female'] = 1 if gender == 'female' else 0
            patient_data['Gender_Male'] = 1 if gender == 'male' else 0
            break
        else:
            print("   Please enter 'Male' or 'Female'")
    
    while True:
        try:
            cdr = float(input("\nEnter Cup-to-Disc Ratio (CDR) (0.0-1.0): "))
            if 0.0 <= cdr <= 1.0:
                patient_data['Cup-to-Disc Ratio (CDR)'] = cdr
                break
            else:
                print("   CDR should be between 0.0 and 1.0")
        except ValueError:
            print("   Please enter a valid number")
    
    while True:
        fh = input("\nFamily History of Glaucoma? (Yes/No): ").strip().lower()
        if fh in ['yes', 'no']:
            patient_data['Family History_No'] = 1 if fh == 'no' else 0
            patient_data['Family History_Yes'] = 1 if fh == 'yes' else 0
            break
        else:
            print("   Please enter 'Yes' or 'No'")
    
    print("\nMedical History:")
    print("   1. None")
    print("   2. Diabetes")
    print("   3. Hypertension")
    print("   4. Both Diabetes and Hypertension")
    while True:
        try:
            mh = int(input("Select option (1-4): "))
            if 1 <= mh <= 4:
                patient_data['Medical History_Both'] = 1 if mh == 4 else 0
                patient_data['Medical History_Diabetes'] = 1 if mh == 2 else 0
                patient_data['Medical History_Hypertension'] = 1 if mh == 3 else 0
                patient_data['Medical History_None'] = 1 if mh == 1 else 0
                break
            else:
                print("   Please select option 1-4")
        except ValueError:
            print("   Please enter a valid number")
    
    while True:
        try:
            va = float(input("\nEnter Visual Acuity (LogMAR) (0.0-2.0): "))
            if 0.0 <= va <= 2.0:
                patient_data['LogMAR VA'] = va
                break
            else:
                print("   LogMAR should be between 0.0 and 2.0")
        except ValueError:
            print("   Please enter a valid number")
    
    print("\nVisual Field Test (VFT) Results:")
    while True:
        try:
            vft_sens = float(input("   Sensitivity (0-100%): "))
            if 0 <= vft_sens <= 100:
                patient_data['VFT Sensitivity'] = vft_sens
                break
            else:
                print("   Please enter value between 0 and 100")
        except ValueError:
            print("   Please enter a valid number")
    
    while True:
        try:
            vft_spec = float(input("   Specificity (0-100%): "))
            if 0 <= vft_spec <= 100:
                patient_data['VFT Specificity'] = vft_spec
                break
            else:
                print("   Please enter value between 0 and 100")
        except ValueError:
            print("   Please enter a valid number")
    
    print("\nOCT (Optical Coherence Tomography) Results:")
    while True:
        try:
            rnfl = float(input("   RNFL Thickness (µm) (50-150): "))
            if 50 <= rnfl <= 150:
                patient_data['OCT RNFL Thickness (µm)'] = rnfl
                break
            else:
                print("   Please enter value between 50 and 150")
        except ValueError:
            print("   Please enter a valid number")
    
    while True:
        try:
            gcc = float(input("   GCC Thickness (µm) (50-150): "))
            if 50 <= gcc <= 150:
                patient_data['OCT GCC Thickness (µm)'] = gcc
                break
            else:
                print("   Please enter value between 50 and 150")
        except ValueError:
            print("   Please enter a valid number")
    
    while True:
        try:
            retinal = float(input("   Retinal Volume (mm³) (5-15): "))
            if 5 <= retinal <= 15:
                patient_data['OCT Retinal Volume (mm³)'] = retinal
                break
            else:
                print("   Please enter value between 5 and 15")
        except ValueError:
            print("   Please enter a valid number")
    
    while True:
        try:
            macular = float(input("   Macular Thickness (µm) (150-350): "))
            if 150 <= macular <= 350:
                patient_data['OCT Macular Thickness (µm)'] = macular
                break
            else:
                print("   Please enter value between 150 and 350")
        except ValueError:
            print("   Please enter a valid number")
    
    print("\nCataract Status:")
    print("   1. No Cataract")
    print("   2. Early Stage")
    print("   3. Advanced Stage")
    while True:
        try:
            cs = int(input("Select option (1-3): "))
            if 1 <= cs <= 3:
                patient_data['Cataract Status_Advanced Stage'] = 1 if cs == 3 else 0
                patient_data['Cataract Status_Early Stage'] = 1 if cs == 2 else 0
                patient_data['Cataract Status_No Cataract'] = 1 if cs == 1 else 0
                break
            else:
                print("   Please select option 1-3")
        except ValueError:
            print("   Please enter a valid number")
    
    print("\nAngle Closure Status:")
    print("   1. Open Angle")
    print("   2. Narrow Angle")
    print("   3. Angle Closure")
    while True:
        try:
            acs = int(input("Select option (1-3): "))
            if 1 <= acs <= 3:
                patient_data['Angle Closure Status_Angle Closure'] = 1 if acs == 3 else 0
                patient_data['Angle Closure Status_Narrow Angle'] = 1 if acs == 2 else 0
                patient_data['Angle Closure Status_Open Angle'] = 1 if acs == 1 else 0
                break
            else:
                print("   Please select option 1-3")
        except ValueError:
            print("   Please enter a valid number")
    
    print("\nMedications (enter comma-separated, or 'none'):")
    print("   Example: Timolol, Latanoprost")
    meds_input = input("Medications: ").strip()
    
    all_meds = ['Timolol', 'Latanoprost', 'Brimonidine', 'Dorzolamide', 'Bimatoprost']
    if meds_input.lower() != 'none':
        meds_list = [m.strip() for m in meds_input.split(',')]
        for med in all_meds:
            patient_data[med] = 1 if med in meds_list else 0
    else:
        for med in all_meds:
            patient_data[med] = 0
    
    print("\nVisual Symptoms (enter comma-separated, or 'none'):")
    print("   Example: Blurred Vision, Eye Pain")
    symptoms_input = input("Symptoms: ").strip()
    
    all_symptoms = ['Blurred Vision', 'Eye Pain', 'Halos', 'Reduced Peripheral Vision']
    if symptoms_input.lower() != 'none':
        symptoms_list = [s.strip() for s in symptoms_input.split(',')]
        for symptom in all_symptoms:
            patient_data[symptom] = 1 if symptom in symptoms_list else 0
    else:
        for symptom in all_symptoms:
            patient_data[symptom] = 0
    
    return patient_data


def predict_glaucoma(patient_data):
    """Load trained model and make prediction"""
    print("\nStep 3: Loading Model and Making Prediction...")
    print("-" * 60)
    
    try:
        with open('glaucoma_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")
        
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        print(f"Loaded {len(feature_names)} features")
        
        input_df = pd.DataFrame([patient_data])
        
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        input_df = input_df[feature_names]
        
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        
        if prediction == 1:
            print("\nDIAGNOSIS: GLAUCOMA DETECTED")
            print(f"   Confidence: {prediction_proba[1]*100:.1f}%")
            print("\nRECOMMENDATION:")
            print("   - Immediate consultation with an ophthalmologist required")
            print("   - Further diagnostic tests recommended")
            print("   - Start monitoring and treatment plan")
        else:
            print("\nDIAGNOSIS: NO GLAUCOMA")
            print(f"   Confidence: {prediction_proba[0]*100:.1f}%")
            print("\nRECOMMENDATION:")
            print("   - Continue regular eye check-ups")
            print("   - Monitor IOP levels periodically")
            print("   - Maintain healthy lifestyle")
        
        print("\nPrediction Probabilities:")
        print(f"   - No Glaucoma: {prediction_proba[0]*100:.1f}%")
        print(f"   - Glaucoma: {prediction_proba[1]*100:.1f}%")
        
        result_data = {
            'IOP': patient_data['Intraocular Pressure (IOP)'],
            'Age': patient_data['Age'],
            'Prediction': 'Glaucoma' if prediction == 1 else 'No Glaucoma',
            'Confidence': f"{max(prediction_proba)*100:.1f}%"
        }
        
        result_df = pd.DataFrame([result_data])
        result_df.to_csv('prediction_result.csv', index=False)
        print("\nResults saved to 'prediction_result.csv'")
        
        return prediction
        
    except FileNotFoundError:
        print("Error: Model files not found!")
        print("Please run 'testing.py' first to train and save the model.")
        return None
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None


def main():
    """Main execution flow"""
    
    final_iop = compute_final_iop()
    if final_iop is None:
        return
    
    try:
        patient_data = get_user_input(final_iop)
    except KeyboardInterrupt:
        print("\n\nInput cancelled by user")
        return
    
    prediction = predict_glaucoma(patient_data)
    
    print("\n" + "=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
