# app.py
from flask import Flask, request, render_template, flash
from markupsafe import Markup
from bias_classifier import predict_bias_with_classifier, load_classifier_and_model # Updated import
from bias_definitions import BIAS_INFO
import os

# Attempt to load the classifier model and raw components on startup
print("Attempting to load BERT classifier model and components...")
classifier_available = load_classifier_and_model() # Use the new loader
if not classifier_available:
    print("WARN: BERT Classifier model/components failed to load. Analysis will be unavailable.")

app = Flask(__name__)
app.secret_key = os.urandom(24)

def format_text_with_attributions(text, attributions, threshold=0.1):
    """
    Formats the input text as HTML, highlighting words based on attribution scores.
    Positive scores are greenish, negative are reddish (conceptual).
    Only highlights if score magnitude is above threshold.
    """
    if not attributions:
        return text # Return plain text if no attributions

    highlighted_parts = []
    current_text_pos = 0
    processed_tokens_text = ""

    # Split the text into words and keep track of original casing and spaces
    import re
    words_and_separators = re.findall(r"[\w'-]+|[^\w\s']+|\s+", text, re.UNICODE)

    # Create a map of attributed tokens (lowercase) to their scores
    attr_map = {word.lower(): score for word, score in attributions}
    
    output_html = ""
    for part in words_and_separators:
        if part.strip() == "": # if it's just whitespace
            output_html += part
            continue

        word_lower = part.lower()
        score = attr_map.get(word_lower, 0) # Get score for the lowercase version

        if abs(score) > threshold:
            # More intense color for higher scores
            alpha = min(1, abs(score) * 2) # Scale alpha, cap at 1
            if score > 0: # Contributes positively to this class
                # Green for positive contributions
                color_intensity = int(min(200, 50 + abs(score) * 300)) # Cap intensity
                background_color = f"rgba({200-color_intensity//2}, {150+color_intensity//2}, {200-color_intensity//2}, 0.7)"

            else: # Contributes negatively (or towards other classes)
                # Red for negative contributions
                color_intensity = int(min(200, 50 + abs(score) * 300))
                background_color = f"rgba({150+color_intensity//2}, {200-color_intensity//2}, {200-color_intensity//2}, 0.7)"

            output_html += f'<span style="background-color: {background_color}; padding: 1px; border-radius: 3px;" title="Score: {score:.3f}">{part}</span>'
        else:
            output_html += part
            
    return Markup(output_html)


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    input_text = ""
    highlighted_text = "" # New variable

    if request.method == 'POST':
        input_text = request.form.get('text_input', '')

        if not input_text.strip():
            flash("Please enter some text to analyze.", "warning")
            return render_template('index.html', result=None, input_text=input_text, highlighted_text=input_text)

        print(f"Received request: Text='{input_text[:50]}...'")

        predicted_bias = "Classifier Error"
        confidence = 0.0
        attributions = [] # Store attributions
        explanation = "Analysis could not be performed."

        if not classifier_available:
             explanation = "The fine-tuned BERT classifier model is not loaded. Please check server logs."
             flash("Classifier model is unavailable. Please run training or check logs.", "danger")
             highlighted_text = input_text
        else:
             # Request attributions from the classifier function
             predicted_bias, confidence, attributions = predict_bias_with_classifier(input_text, get_attributions=True)

             if predicted_bias == "Classifier Error":
                  flash("An error occurred during classifier prediction. Check logs.", "danger")
                  explanation = BIAS_INFO.get(predicted_bias, {}).get("definition", explanation)
                  highlighted_text = input_text
             else:
                  explanation = BIAS_INFO.get(predicted_bias, {}).get("definition", "No definition available.")
                  # Format text with highlights if attributions are available
                  if attributions:
                      highlighted_text = format_text_with_attributions(input_text, attributions)
                  else:
                      highlighted_text = input_text


        bias_info = BIAS_INFO.get(predicted_bias, {})
        result = {
            "text": input_text, # Original text
            "highlighted_text": highlighted_text, # Text with HTML highlights
            "bias": predicted_bias,
            "confidence": f"{confidence:.2f}",
            "definition": bias_info.get("definition", "Definition not available."),
            "recommendations": bias_info.get("recommendations", []),
            "attributions_raw": attributions # For debugging or alternative display
        }
        if not bias_info and predicted_bias != "Classifier Error":
             flash(f"Analysis resulted in an unexpected state: {predicted_bias}", "warning")
             result["recommendations"] = ["Review the input text and consider if it matches known bias patterns."]

    return render_template('index.html', result=result, input_text=input_text, highlighted_text=highlighted_text)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
