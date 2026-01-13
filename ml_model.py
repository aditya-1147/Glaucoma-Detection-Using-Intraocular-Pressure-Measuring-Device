import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class IOPGlaucomaModel:
    def __init__(self):
        self.iop_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.glaucoma_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the glaucoma dataset"""
        df = pd.read_csv(csv_path)
        
        # Handle categorical variables
        categorical_cols = ['Gender', 'Family History', 'Medical History', 'Cataract Status', 
                          'Angle Closure Status', 'Visual Symptoms', 'Diagnosis', 'Glaucoma Type']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Extract numeric features from complex columns
        if 'Visual Field Test Results' in df.columns:
            df['Sensitivity'] = df['Visual Field Test Results'].str.extract(r'Sensitivity: ([\d.]+)').astype(float)
            df['Specificity'] = df['Visual Field Test Results'].str.extract(r'Specificity: ([\d.]+)').astype(float)
        
        if 'Optical Coherence Tomography (OCT) Results' in df.columns:
            df['RNFL_Thickness'] = df['Optical Coherence Tomography (OCT) Results'].str.extract(r'RNFL Thickness: ([\d.]+)').astype(float)
            df['GCC_Thickness'] = df['Optical Coherence Tomography (OCT) Results'].str.extract(r'GCC Thickness: ([\d.]+)').astype(float)
            df['Retinal_Volume'] = df['Optical Coherence Tomography (OCT) Results'].str.extract(r'Retinal Volume: ([\d.]+)').astype(float)
            df['Macular_Thickness'] = df['Optical Coherence Tomography (OCT) Results'].str.extract(r'Macular Thickness: ([\d.]+)').astype(float)
        
        # Convert Visual Acuity to numeric
        if 'Visual Acuity Measurements' in df.columns:
            df['Visual_Acuity_Numeric'] = df['Visual Acuity Measurements'].str.extract(r'([\d.]+)').astype(float)
        
        return df
    
    def prepare_features(self, df):
        """Prepare feature matrix for training"""
        feature_cols = ['Age', 'Gender', 'Visual_Acuity_Numeric', 'Cup-to-Disc Ratio (CDR)',
                       'Family History', 'Medical History', 'Cataract Status', 'Angle Closure Status',
                       'Sensitivity', 'Specificity', 'RNFL_Thickness', 'GCC_Thickness',
                       'Retinal_Volume', 'Macular_Thickness', 'Pachymetry']
        
        # Select available features
        available_features = [col for col in feature_cols if col in df.columns]
        X = df[available_features].fillna(df[available_features].median())
        
        return X, available_features
    
    def train_models(self, csv_path):
        """Train both IOP prediction and glaucoma classification models"""
        print("Loading and preprocessing data...")
        df = self.load_and_preprocess_data(csv_path)
        
        X, feature_names = self.prepare_features(df)
        y_iop = df['Intraocular Pressure (IOP)']
        y_glaucoma = df['Diagnosis']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_iop_train, y_iop_test, y_glaucoma_train, y_glaucoma_test = train_test_split(
            X_scaled, y_iop, y_glaucoma, test_size=0.2, random_state=42
        )
        
        # Train IOP prediction model
        print("Training IOP prediction model...")
        self.iop_model.fit(X_train, y_iop_train)
        iop_pred = self.iop_model.predict(X_test)
        iop_mse = mean_squared_error(y_iop_test, iop_pred)
        print(f"IOP Model - MSE: {iop_mse:.2f}, RMSE: {np.sqrt(iop_mse):.2f}")
        
        # Train glaucoma classification model
        print("Training glaucoma classification model...")
        self.glaucoma_model.fit(X_train, y_glaucoma_train)
        glaucoma_pred = self.glaucoma_model.predict(X_test)
        glaucoma_acc = accuracy_score(y_glaucoma_test, glaucoma_pred)
        print(f"Glaucoma Model - Accuracy: {glaucoma_acc:.3f}")
        
        # Feature importance
        self.plot_feature_importance(feature_names)
        
        return {
            'iop_mse': iop_mse,
            'iop_rmse': np.sqrt(iop_mse),
            'glaucoma_accuracy': glaucoma_acc,
            'feature_names': feature_names
        }
    
    def plot_feature_importance(self, feature_names):
        """Plot feature importance for both models"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # IOP model feature importance
        iop_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.iop_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        sns.barplot(data=iop_importance.head(10), x='importance', y='feature', ax=ax1)
        ax1.set_title('IOP Prediction - Feature Importance')
        
        # Glaucoma model feature importance
        glaucoma_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.glaucoma_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        sns.barplot(data=glaucoma_importance.head(10), x='importance', y='feature', ax=ax2)
        ax2.set_title('Glaucoma Classification - Feature Importance')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_iop(self, patient_data):
        """Predict IOP for new patient data"""
        patient_scaled = self.scaler.transform([patient_data])
        return self.iop_model.predict(patient_scaled)[0]
    
    def predict_glaucoma(self, patient_data):
        """Predict glaucoma diagnosis for new patient data"""
        patient_scaled = self.scaler.transform([patient_data])
        prediction = self.glaucoma_model.predict(patient_scaled)[0]
        probability = self.glaucoma_model.predict_proba(patient_scaled)[0]
        return prediction, max(probability)
    
    def save_models(self):
        """Save trained models and preprocessors"""
        joblib.dump(self.iop_model, 'iop_model.pkl')
        joblib.dump(self.glaucoma_model, 'glaucoma_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.label_encoders, 'label_encoders.pkl')
        print("Models saved successfully!")
    
    def load_models(self):
        """Load pre-trained models"""
        self.iop_model = joblib.load('iop_model.pkl')
        self.glaucoma_model = joblib.load('glaucoma_model.pkl')
        self.scaler = joblib.load('scaler.pkl')
        self.label_encoders = joblib.load('label_encoders.pkl')
        print("Models loaded successfully!")

def main():
    """Main function to train and save the models"""
    model = IOPGlaucomaModel()
    
    # Train models
    results = model.train_models('glaucoma_dataset(1).csv')
    
    print("\nTraining Results:")
    print(f"IOP Prediction RMSE: {results['iop_rmse']:.2f} mmHg")
    print(f"Glaucoma Classification Accuracy: {results['glaucoma_accuracy']:.1%}")
    
    # Save models
    model.save_models()
    
    # Example prediction
    print("\nExample Prediction:")
    sample_data = [65, 1, 0.1, 0.5, 0, 1, 1, 1, 0.8, 0.9, 85, 60, 6.0, 270, 550]
    iop_pred = model.predict_iop(sample_data)
    glaucoma_pred, confidence = model.predict_glaucoma(sample_data)
    
    print(f"Predicted IOP: {iop_pred:.2f} mmHg")
    print(f"Glaucoma Risk: {'High' if glaucoma_pred == 1 else 'Low'} (Confidence: {confidence:.1%})")

if __name__ == "__main__":
    main()