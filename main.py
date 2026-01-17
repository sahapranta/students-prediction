import gradio as gr
import pandas as pd
import pickle
import numpy as np
from pathlib import Path


class HSCPredictor:
    """HSC GPA prediction model wrapper"""
    
    def __init__(self, model_path="student_rf_pipeline.pkl"):
        self.model = self._load_model(model_path)
        self.feature_names = [
            'gender', 'age', 'address', 'famsize', 'Pstatus', 
            'M_Edu', 'F_Edu', 'M_Job', 'F_Job', 'relationship', 
            'smoker', 'tuition_fee', 'time_friends', 'ssc_result'
        ]
    
    def _load_model(self, model_path):
        """Load the trained model with error handling"""
        try:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print(f"✓ Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def predict(self, *args):
        """Make prediction from input arguments"""
        try:
            # Validate inputs
            if len(args) != len(self.feature_names):
                return "Error: Invalid number of inputs"
            
            # Create DataFrame
            input_df = pd.DataFrame([args], columns=self.feature_names)
            
            # Make prediction
            prediction = self.model.predict(input_df)[0]
            clipped_prediction = np.clip(prediction, 0, 5)
            
            # Return formatted result with confidence indicator
            confidence = "High" if 1.0 <= clipped_prediction <= 4.5 else "Moderate"
            return (f"**Predicted HSC Result:** {clipped_prediction:.2f} / 5.00\n\n"
                    f"*Confidence: {confidence}*")
            
        except Exception as e:
            return f"Error making prediction: {str(e)}"


def create_interface():
    """Create and configure the Gradio interface"""
    
    predictor = HSCPredictor()
    
    # Define inputs with better defaults and validation
    inputs = [
        gr.Radio(["M", "F"], label="Gender", value="M"),
        gr.Number(label="Age", value=18, minimum=15, maximum=25),
        gr.Radio(["Urban", "Rural"], label="Address", value="Urban"),
        gr.Radio(["GT3", "LE3"], label="Family Size (GT3 = >3, LE3 = ≤3)", value="GT3"),
        gr.Radio(["Together", "Apart"], label="Parent Status", value="Together"),
        gr.Slider(0, 4, step=1, value=2, label="Mother's Education (0=None, 4=Higher)"),
        gr.Slider(0, 4, step=1, value=2, label="Father's Education (0=None, 4=Higher)"),
        gr.Dropdown(
            ["At_home", "Health", "Other", "Services", "Teacher"], 
            label="Mother's Job", 
            value="Other"
        ),
        gr.Dropdown(
            ["Teacher", "Other", "Services", "Health", "Business", "Farmer"], 
            label="Father's Job",
            value="Other"
        ),
        gr.Radio(["Yes", "No"], label="In a Relationship", value="No"),
        gr.Radio(["Yes", "No"], label="Smoker", value="No"),
        gr.Number(label="Tuition Fee (Annual)", value=5000, minimum=0),
        gr.Slider(1, 5, step=1, value=3, label="Time with Friends (1=Very Low, 5=Very High)"),
        gr.Number(label="SSC Result (GPA)", value=4.0, minimum=0, maximum=5)
    ]
    
    # Create interface with better styling
    interface = gr.Interface(
        fn=predictor.predict,
        inputs=inputs,
        outputs=gr.Markdown(label="Prediction Result"),
        title="🎓 HSC Result Predictor",
        description=(
            "Predict Higher Secondary Certificate (HSC) exam results based on "
            "student demographics, family background, and academic history."
        ),
        examples=[
            ["M", 18, "Urban", "GT3", "Together", 3, 3, "Teacher", "Services", 
             "No", "No", 6000, 3, 4.5],
            ["F", 17, "Rural", "LE3", "Apart", 2, 2, "At_home", "Farmer", 
             "Yes", "No", 3000, 4, 3.8],
        ],        
    )
    
    return interface


def main():
    """Main entry point"""
    app = create_interface()
    app.launch(share=True)


if __name__ == "__main__":
    main()