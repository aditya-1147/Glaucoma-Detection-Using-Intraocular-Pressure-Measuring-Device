from flask import Flask, request, jsonify
import csv
import os
import pandas as pd
import numpy as np
from ml_model import IOPGlaucomaModel
import joblib
from datetime import datetime

app = Flask(__name__)

# Global variables
readings = []
counter = 0
model = None

def load_ml_model():
    """Load the trained ML model"""
    global model
    try:
        model = IOPGlaucomaModel()
        model.load_models()
        print("✅ ML Model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading ML model: {e}")
        return False

def predict_from_sensor_data(sensor_readings, patient_info=None):
    """Make predictions using sensor data"""
    if model is None:
        return None, "Model not loaded"
    
    try:
        # Default patient info if not provided
        if patient_info is None:
            patient_info = [65, 1, 0.1, 0.5, 0, 1, 1, 1, 0.8, 0.9, 85, 60, 6.0, 270, 550]
        
        # Convert sensor readings to IOP if needed
        if len(sensor_readings) > 0 and len(sensor_readings[0]) >= 3:
            # Use the IOP values from sensor data
            iop_values = [reading[2] for reading in sensor_readings]
            avg_iop = np.mean(iop_values)
            
            # Update patient info with actual IOP
            patient_info[2] = avg_iop  # Assuming IOP is at index 2
            
            # Make predictions
            iop_pred = model.predict_iop(patient_info)
            glaucoma_pred, confidence = model.predict_glaucoma(patient_info)
            
            return {
                'predicted_iop': round(iop_pred, 2),
                'glaucoma_risk': 'High' if glaucoma_pred == 1 else 'Low',
                'confidence': round(confidence * 100, 1),
                'sensor_iop_avg': round(avg_iop, 2),
                'num_readings': len(sensor_readings)
            }, None
        else:
            return None, "Insufficient sensor data"
            
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

@app.route('/iop', methods=['POST'])
def get_iop_data():
    global counter, readings

    try:
        if not request.is_json:
            return jsonify({"error": "Expected JSON data"}), 400

        data = request.get_json(force=True)
        piezo_value = data.get('piezo')
        fsr_value = data.get('fsr')
        iop_value = data.get('iop')

        if all(val is not None for val in [piezo_value, fsr_value, iop_value]):
            print(f"✅ Received: Piezo={piezo_value}, FSR={fsr_value}, IOP={iop_value}")
            readings.append([piezo_value, fsr_value, iop_value])
            counter += 1

            response_data = {"status": "success", "message": "Data received", "count": counter}

            # Make prediction every 5 readings for real-time feedback
            if counter % 5 == 0 and model is not None:
                prediction, error = predict_from_sensor_data(readings[-5:])
                if prediction:
                    response_data["prediction"] = prediction
                    print(f"🔮 Prediction: IOP={prediction['predicted_iop']}, Risk={prediction['glaucoma_risk']}")

            # Save data every 10 readings
            if counter >= 10:
                file_name = f'sensor_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                with open(file_name, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Piezo', 'FSR', 'IOP'])
                    writer.writerows(readings)
                
                # Final prediction on complete dataset
                if model is not None:
                    final_prediction, error = predict_from_sensor_data(readings)
                    if final_prediction:
                        response_data["final_prediction"] = final_prediction
                        print(f"📊 Final Analysis: {final_prediction}")
                
                print(f"📁 Data saved to {file_name}")
                readings = []
                counter = 0

            return jsonify(response_data), 200
        else:
            print("❌ Invalid JSON keys or values:", data)
            return jsonify({"error": "Missing or invalid data fields"}), 400

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def manual_prediction():
    """Manual prediction endpoint for uploaded data"""
    try:
        data = request.get_json()
        
        if 'sensor_data' in data:
            sensor_readings = data['sensor_data']
            patient_info = data.get('patient_info')
            
            prediction, error = predict_from_sensor_data(sensor_readings, patient_info)
            
            if error:
                return jsonify({"error": error}), 400
            
            return jsonify({
                "status": "success",
                "prediction": prediction
            }), 200
        else:
            return jsonify({"error": "No sensor_data provided"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Endpoint to retrain the model"""
    try:
        global model
        model = IOPGlaucomaModel()
        
        # Train on the dataset
        results = model.train_models('glaucoma_dataset(1).csv')
        model.save_models()
        
        return jsonify({
            "status": "success",
            "message": "Model trained successfully",
            "results": results
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get server and model status"""
    return jsonify({
        "server_status": "running",
        "model_loaded": model is not None,
        "current_readings": len(readings),
        "total_received": counter
    }), 200

@app.route('/iop', methods=['GET'])
def block_get():
    return jsonify({"error": "GET not allowed. Use POST with JSON."}), 405

if __name__ == '__main__':
    print("🚀 Starting IOP Estimation Server...")
    
    # Try to load the ML model
    if not load_ml_model():
        print("⚠️  Server starting without ML model. Use /train endpoint to train model.")
    
    print("📡 Server ready for sensor data on /iop")
    print("🔮 Manual predictions available on /predict")
    print("🎯 Model training available on /train")
    
    app.run(debug=False, host='0.0.0.0', port=5000)