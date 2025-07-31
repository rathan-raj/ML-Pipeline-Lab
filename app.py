
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import gradio as gr
import os

# Initialize H2O cluster if not already running
# This is crucial for deployment on Hugging Face Spaces where it's a fresh environment
try:
    h2o.init()
except Exception as e:
    print(f"H2O init failed: {e}. Assuming already running or trying again.")
    h2o.cluster().shutdown() # try to shutdown if stuck
    h2o.init()

# Load the trained H2O model
# The model will be in the same directory as app.py on Hugging Face Spaces
# Adjust model_filename if you saved with a specific H2O path that is not root
try:
    loaded_model = h2o.load("DeepLearning_grid_2_AutoML_1_20250731_184152_model_1")
    print(f"Model {loaded_model.model_id} loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    loaded_model = None # Handle case where model might not load


def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    if loaded_model is None:
        return "Model not loaded. Please check deployment logs."

    # Create an H2OFrame from the input values
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                               columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    h2o_input = h2o.H2OFrame(input_data)

    # Make a prediction
    prediction = loaded_model.predict(h2o_input)

    # Get the predicted class (species)
    predicted_class = prediction['predict'].as_data_frame().iloc

    # Optionally get class probabilities
    # probabilities = prediction.as_data_frame().iloc[0, 1:].values
    # prob_str = ", ".join([f"{col}: {prob:.2f}" for col, prob in zip(loaded_model.target_names, probabilities)])

    return f"Predicted Species: {predicted_class}" # (Probabilities: {prob_str})"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_species,
    inputs=[
        gr.Slider(minimum=4.0, maximum=8.0, value=5.8, label="Sepal Length (cm)"),
        gr.Slider(minimum=2.0, maximum=4.5, value=3.0, label="Sepal Width (cm)"),
        gr.Slider(minimum=1.0, maximum=7.0, value=4.0, label="Petal Length (cm)"),
        gr.Slider(minimum=0.1, maximum=2.5, value=1.2, label="Petal Width (cm)")
    ],
    outputs="text",
    title="Iris Species Predictor (H2O.ai AutoML)",
    description="Predicts Iris species based on flower measurements using a model trained with H2O.ai AutoML."
)

# Launch the Gradio app (only for local testing, Hugging Face handles launch for deployed spaces)
if __name__ == "__main__":
    iface.launch(share=True) # share=True generates a temporary public link for local testing
