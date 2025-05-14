# bias_classifier.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import torch
import os
from bias_definitions import id2label, CLASSIFIER_LABELS # Make sure CLASSIFIER_LABELS is imported

# NEW IMPORT
from transformers_interpret import SequenceClassificationExplainer # If you installed transformers-interpret

# --- Configuration ---
MODEL_DIR = "./fine_tuned_bert_model"

# --- Global variables ---
classifier_pipeline = None
raw_model = None # To store the raw model for the explainer
raw_tokenizer = None # To store the raw tokenizer for the explainer

def load_classifier_and_model():
    """
    Loads the fine-tuned BERT model, tokenizer, and creates a pipeline.
    Also stores raw model and tokenizer for interpretability.
    """
    global classifier_pipeline, raw_model, raw_tokenizer
    if classifier_pipeline is not None:
        return True # Already loaded

    print("Loading fine-tuned BERT bias classifier model...")
    if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
        print(f"Error: Model directory '{MODEL_DIR}' not found or empty.")
        print("Please run 'python train_transformer.py' first.")
        return False

    try:
        device_id = 0 if torch.cuda.is_available() else -1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device} (id: {device_id}) for classifier")

        raw_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        raw_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        raw_model.to(device)
        raw_model.eval() # Ensure model is in eval mode for explanations

        classifier_pipeline = TextClassificationPipeline(
            model=raw_model,
            tokenizer=raw_tokenizer,
            device=device_id
        )
        print("BERT Classifier model and raw components loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading BERT classifier model: {e}")
        classifier_pipeline = None # Ensure it's None on failure
        raw_model = None
        raw_tokenizer = None
        return False

def get_word_attributions(text, predicted_class_name):
    """
    Uses transformers-interpret to get word attributions for the given text and predicted class.
    Returns a list of (word, attribution_score) tuples.
    """
    global raw_model, raw_tokenizer
    if raw_model is None or raw_tokenizer is None:
        print("Error: Raw model/tokenizer not loaded for attributions.")
        return []

    try:
        # Ensure the predicted_class_name is one of the actual labels the model was trained on
        # If predicted_class_name is e.g. "Classifier Error", we can't get attributions for it.
        if predicted_class_name not in CLASSIFIER_LABELS:
             print(f"Cannot get attributions for class '{predicted_class_name}'. It's not a trained label.")
             return []


        # Initialize the explainer for the specific model type (BERT in this case)
        # For BERT, it's often 'bert'
        # Check transformers-interpret documentation for other model types if you switch.
        # The explainer needs the raw model and tokenizer, not the pipeline.
        cls_explainer = SequenceClassificationExplainer(
            raw_model,
            raw_tokenizer
        )
        # Get attributions. The target class can be specified by its string name.
        word_attributions = cls_explainer(text, class_name=predicted_class_name)
        # cls_explainer.visualize("attribution_output.html") # Optional: creates an HTML visualization

        # word_attributions is a list of tuples (word, score)
        return word_attributions
    except Exception as e:
        print(f"Error getting word attributions: {e}")
        return []


def predict_bias_with_classifier(text, get_attributions=False): # Added get_attributions flag
    """
    Predicts bias and optionally returns word attributions.
    Returns: (predicted_bias, confidence, word_attributions_list)
    """
    global classifier_pipeline
    attributions = []

    if classifier_pipeline is None:
        print("BERT Classifier pipeline not loaded. Attempting to load...")
        if not load_classifier_and_model(): # Use the new loading function
             return "Classifier Error", 0.0, attributions # Return specific error label

    if not text or not isinstance(text, str):
         print("Warning: Invalid input text for classifier.")
         return "Neutral", 0.0, attributions

    try:
        results = classifier_pipeline(text, return_all_scores=False)
        if not results:
             print("Warning: Classifier returned no results.")
             return "Neutral", 0.0, attributions

        top_result = results[0]
        predicted_label_id_str = top_result['label']
        confidence = top_result['score']
        predicted_bias = "Unknown Label" # Default

        print(f"DEBUG: Raw label from pipeline: {predicted_label_id_str}")

        if predicted_label_id_str.startswith("LABEL_"):
            try:
                predicted_label_id = int(predicted_label_id_str.split('_')[-1])
                predicted_bias = id2label.get(predicted_label_id, "Unknown Label")
            except ValueError:
                print(f"Error: Could not parse integer ID from label: {predicted_label_id_str}")
                return "Classifier Error", 0.0, attributions
        elif predicted_label_id_str in id2label.values(): # If pipeline returns string name
            predicted_bias = predicted_label_id_str
        else:
            print(f"Error: Unexpected label format from pipeline: {predicted_label_id_str}")
            return "Classifier Error", 0.0, attributions

        # Get attributions if requested and if a valid bias was predicted
        if get_attributions and predicted_bias not in ["Unknown Label", "Classifier Error", "Neutral"]:
            # We generally want attributions for *why* a specific bias was chosen,
            # not for why it was 'Neutral' (though technically possible).
            # Only get attributions if the predicted_bias is a valid, non-error, non-neutral class.
            if predicted_bias in CLASSIFIER_LABELS and predicted_bias != "Neutral":
                 attributions = get_word_attributions(text, predicted_bias)
            else:
                 print(f"Skipping attributions for predicted class: {predicted_bias}")


        return predicted_bias, confidence, attributions

    except Exception as e:
        print(f"An error occurred during classifier prediction: {e}")
        return "Classifier Error", 0.0, attributions

# Call the loader when the module is imported (or app starts)
# load_classifier_and_model() # Call this to ensure model/tokenizer are loaded for explainer

if __name__ == "__main__":
    if load_classifier_and_model(): # Ensure model is loaded
        test_text_confirm = "This data clearly confirms what I suspected about user retention."
        test_text_neutral = "The meeting is scheduled for 3 PM."

        bias, score, attrs = predict_bias_with_classifier(test_text_confirm, get_attributions=True)
        print(f"\nText: '{test_text_confirm}'\nPredicted Bias: {bias}, Confidence: {score:.4f}")
        if attrs:
            print("Attributions:", attrs)

        bias, score, attrs = predict_bias_with_classifier(test_text_neutral, get_attributions=True)
        print(f"\nText: '{test_text_neutral}'\nPredicted Bias: {bias}, Confidence: {score:.4f}")
        if attrs:
            print("Attributions:", attrs) # Likely empty for "Neutral" based on current logic
    else:
        print("\nCould not load BERT classifier. Run training script first.")
