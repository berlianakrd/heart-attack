"""
Indonesia Heart Attack Prediction - Flask Web Application
Main application file
"""

from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import joblib
import json
import os

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'indonesia-heart-attack-prediction-2024-secure-key'

# Load model and preprocessors
MODEL_PATH = 'models/best_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
ENCODERS_PATH = 'models/label_encoders.pkl'
METADATA_PATH = 'models/model_metadata.json'

# Global variables for model and preprocessors
model = None
scaler = None
label_encoders = None
model_metadata = None
feature_names = None

def load_model_and_preprocessors():
    """Load trained model and preprocessing objects"""
    global model, scaler, label_encoders, model_metadata, feature_names
    
    try:
        # Load model
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"✓ Model loaded from {MODEL_PATH}")
        else:
            print(f"⚠️  Model not found at {MODEL_PATH}")
            
        # Load scaler
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print(f"✓ Scaler loaded from {SCALER_PATH}")
        else:
            print(f"⚠️  Scaler not found at {SCALER_PATH}")
            
        # Load label encoders
        if os.path.exists(ENCODERS_PATH):
            label_encoders = joblib.load(ENCODERS_PATH)
            print(f"✓ Label encoders loaded from {ENCODERS_PATH}")
        else:
            print(f"⚠️  Label encoders not found at {ENCODERS_PATH}")
            
        # Load metadata
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                model_metadata = json.load(f)
            feature_names = model_metadata.get('features', [])
            print(f"✓ Model metadata loaded from {METADATA_PATH}")
        else:
            print(f"⚠️  Metadata not found at {METADATA_PATH}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading model/preprocessors: {str(e)}")
        return False


def preprocess_input(data):
    """
    Preprocess input data for prediction
    
    Parameters:
    -----------
    data : dict
        Input data from form
        
    Returns:
    --------
    numpy.array
        Preprocessed data ready for prediction
    """
    try:
        # Create DataFrame from input
        df = pd.DataFrame([data])
        
        # Encode categorical variables
        categorical_cols = ['gender', 'region', 'income_level', 'smoking_status', 
                          'alcohol_consumption', 'physical_activity', 'dietary_habits',
                          'air_pollution_exposure', 'stress_level', 'EKG_results']
        
        for col in categorical_cols:
            if col in df.columns and col in label_encoders:
                try:
                    df[col] = label_encoders[col].transform(df[col])
                except:
                    # If value not in encoder, use most common class
                    df[col] = 0
        
        # Ensure all required features are present
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select only required features in correct order
        df = df[feature_names]
        
        # Scale features
        if scaler is not None:
            df_scaled = scaler.transform(df)
        else:
            df_scaled = df.values
            
        return df_scaled
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise


# Routes
@app.route('/')
def index():
    """Homepage"""
    return render_template('index.html')


@app.route('/predict')
def predict_page():
    """Prediction form page"""
    return render_template('predict.html', 
                         model_name=model_metadata.get('model_name', 'Model') if model_metadata else 'Model',
                         accuracy=f"{model_metadata.get('accuracy', 0)*100:.2f}" if model_metadata else 'N/A')


@app.route('/analysis')
def analysis():
    """
    Data analysis page - Shows individual patient analysis
    If patient data exists in session, show individual analysis
    Otherwise, show dataset statistics
    """
    # Check if patient data exists in session
    patient_data = session.get('patient_data', None)
    prediction_result = session.get('prediction_result', None)
    
    if patient_data and prediction_result:
        # Individual patient analysis
        return render_template('analysis.html', 
                             patient_data=patient_data,
                             prediction_result=prediction_result,
                             is_individual=True)
    else:
        # Dataset statistics (fallback)
        try:
            # Try to load cleaned data first, fallback to original data
            data_paths = [
                'data/heart_attack_data_cleaned.csv',
                'data/heart_attack_data.csv'
            ]
            
            df = None
            for path in data_paths:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    print(f"✓ Loaded data from {path}")
                    break
            
            if df is None:
                raise FileNotFoundError("Dataset not found")
            
            # Calculate comprehensive statistics
            stats = {
                # Basic statistics
                'total_samples': int(len(df)),
                'heart_attack_cases': int(df['heart_attack'].sum()) if 'heart_attack' in df.columns else 0,
                'heart_attack_rate': f"{(df['heart_attack'].sum()/len(df))*100:.1f}" if 'heart_attack' in df.columns else "N/A",
                'mean_age': f"{df['age'].mean():.1f}" if 'age' in df.columns else "N/A",
                
                # Demographics
                'male_count': int((df['gender'] == 'Male').sum()) if 'gender' in df.columns else 0,
                'female_count': int((df['gender'] == 'Female').sum()) if 'gender' in df.columns else 0,
                'urban_count': int((df['region'] == 'Urban').sum()) if 'region' in df.columns else 0,
                'rural_count': int((df['region'] == 'Rural').sum()) if 'region' in df.columns else 0,
                
                # Risk factors prevalence
                'hypertension_rate': f"{(df['hypertension'].sum()/len(df))*100:.1f}" if 'hypertension' in df.columns else "N/A",
                'diabetes_rate': f"{(df['diabetes'].sum()/len(df))*100:.1f}" if 'diabetes' in df.columns else "N/A",
                'obesity_rate': f"{(df['obesity'].sum()/len(df))*100:.1f}" if 'obesity' in df.columns else "N/A",
            }
            
            print(f"✓ Statistics calculated successfully")
            
        except Exception as e:
            print(f"❌ Error loading statistics: {str(e)}")
            stats = {}
        
        return render_template('analysis.html', 
                             stats=stats,
                             is_individual=False)


@app.route('/about')
def about():
    """About page"""
    return render_template('about.html',
                         model_info=model_metadata if model_metadata else {})


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for heart attack prediction
    
    Expected JSON format:
    {
        "age": 55,
        "gender": "Male",
        "region": "Urban",
        ... (all required features)
    }
    
    Returns:
    {
        "success": true/false,
        "prediction": 0/1,
        "probability": 0.XX,
        "risk_level": "Low/Medium/High",
        "message": "..."
    }
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'success': False,
                'message': 'Model not loaded. Please contact administrator.'
            }), 500
        
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        # Save patient data to session for analysis page
        session['patient_data'] = data
        
        # Preprocess input
        X = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            probability = float(model.predict_proba(X)[0][1])
        
        # Determine risk level
        if probability is not None:
            if probability < 0.3:
                risk_level = "Low"
                risk_color = "success"
            elif probability < 0.6:
                risk_level = "Medium"
                risk_color = "warning"
            else:
                risk_level = "High"
                risk_color = "danger"
        else:
            risk_level = "High" if prediction == 1 else "Low"
            risk_color = "danger" if prediction == 1 else "success"
        
        # Prepare response
        response = {
            'success': True,
            'prediction': int(prediction),
            'prediction_label': 'Heart Attack Risk' if prediction == 1 else 'No Heart Attack Risk',
            'probability': round(probability * 100, 2) if probability is not None else None,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'message': get_recommendation_message(prediction, probability),
            'recommendations': get_recommendations(data, prediction)
        }
        
        # Save prediction result to session
        session['prediction_result'] = response
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Prediction error: {str(e)}'
        }), 500


def get_recommendation_message(prediction, probability):
    """Generate personalized recommendation message"""
    if prediction == 1:
        if probability is not None and probability > 0.7:
            return "⚠️ HIGH RISK: Immediate medical consultation recommended. Please see a cardiologist as soon as possible."
        else:
            return "⚠️ RISK DETECTED: Please consult with a healthcare professional for proper assessment."
    else:
        if probability is not None and probability < 0.2:
            return "✅ LOW RISK: Continue maintaining healthy lifestyle habits."
        else:
            return "✅ CURRENTLY LOW RISK: Maintain healthy habits and regular check-ups."


def get_recommendations(data, prediction):
    """Generate personalized recommendations based on risk factors"""
    recommendations = []
    
    # Check age
    if data.get('age', 0) > 50:
        recommendations.append("Regular cardiovascular check-ups recommended due to age")
    
    # Check hypertension
    if data.get('hypertension', 0) == 1:
        recommendations.append("Monitor and control blood pressure regularly")
    
    # Check diabetes
    if data.get('diabetes', 0) == 1:
        recommendations.append("Maintain good blood sugar control")
    
    # Check obesity
    if data.get('obesity', 0) == 1:
        recommendations.append("Weight management program recommended")
    
    # Check smoking
    if data.get('smoking_status', '') == 'Current':
        recommendations.append("Quit smoking - major risk factor for heart disease")
    
    # Check physical activity
    if data.get('physical_activity', '') == 'Low':
        recommendations.append("Increase physical activity to at least 150 minutes per week")
    
    # Check diet
    if data.get('dietary_habits', '') == 'Unhealthy':
        recommendations.append("Adopt a heart-healthy diet (low salt, low saturated fat)")
    
    # General recommendations
    if prediction == 1:
        recommendations.append("Schedule immediate appointment with cardiologist")
        recommendations.append("Consider cardiac screening tests (ECG, stress test)")
    else:
        recommendations.append("Continue annual health check-ups")
        recommendations.append("Maintain healthy lifestyle habits")
    
    return recommendations


@app.route('/api/statistics', methods=['GET'])
def api_statistics():
    """API endpoint to get dataset statistics"""
    try:
        # Try multiple data paths
        data_paths = [
            'data/heart_attack_data_cleaned.csv',
            'data/heart_attack_data.csv'
        ]
        
        df = None
        for path in data_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                break
        
        if df is None:
            raise FileNotFoundError("Dataset not found")
        
        stats = {
            'total_samples': int(len(df)),
            'heart_attack_cases': int(df['heart_attack'].sum()),
            'heart_attack_rate': float((df['heart_attack'].sum()/len(df))*100),
            'demographics': {
                'mean_age': float(df['age'].mean()),
                'age_range': [int(df['age'].min()), int(df['age'].max())],
                'gender': {
                    'male': int((df['gender'] == 'Male').sum()),
                    'female': int((df['gender'] == 'Female').sum())
                },
                'region': {
                    'urban': int((df['region'] == 'Urban').sum()),
                    'rural': int((df['region'] == 'Rural').sum())
                }
            },
            'risk_factors': {
                'hypertension': float((df['hypertension'].sum()/len(df))*100),
                'diabetes': float((df['diabetes'].sum()/len(df))*100),
                'obesity': float((df['obesity'].sum()/len(df))*100),
                'family_history': float((df['family_history'].sum()/len(df))*100)
            }
        }
        
        return jsonify({
            'success': True,
            'data': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/model-info', methods=['GET'])
def api_model_info():
    """API endpoint to get model information"""
    if model_metadata:
        return jsonify({
            'success': True,
            'data': model_metadata
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Model metadata not available'
        }), 404


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500


# Initialize
if __name__ == '__main__':
    print("\n" + "="*60)
    print("INDONESIA HEART ATTACK PREDICTION - WEB APPLICATION")
    print("="*60)
    
    # Load model and preprocessors
    print("\nLoading model and preprocessors...")
    success = load_model_and_preprocessors()
    
    if success:
        print("\n✅ All components loaded successfully!")
        print("\n" + "="*60)
        print("Starting Flask application...")
        print("Access the application at: http://localhost:5000")
        print("="*60 + "\n")
        
        # Run Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load components. Please check if model files exist.")
        print("Run the notebooks first to generate model files.")