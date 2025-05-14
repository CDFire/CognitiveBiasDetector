BIAS_INFO = {
    "Confirmation Bias": {
        "definition": "The tendency to search for, interpret, favor, and recall information in a way that confirms or supports one's prior beliefs or hypotheses.",
        "recommendations": [
            "Actively seek out disconfirming evidence or alternative viewpoints.",
            "Consider the opposite: Ask 'What are reasons this might be wrong?'",
            "Surround yourself with diverse perspectives.",
            "Be aware of filtering information sources (echo chambers)."
        ],
         "keywords": ["confirm", "suspected", "already thought", "evidence supports", "knew it"]
    },
    "Anchoring Bias": {
        "definition": "The tendency to rely too heavily on the first piece of information offered (the 'anchor') when making decisions.",
        "recommendations": [
            "Be aware of the first number or fact presented. Question its relevance.",
            "Generate your own estimate or opinion *before* seeing others'.",
            "Seek multiple data points or opinions before deciding.",
            "If negotiating, make the first offer if you are well-informed."
        ],
         "keywords": ["initial offer", "first quote", "starting point", "compared to", "baseline"]
    },
    "Availability Heuristic": {
        "definition": "Overestimating the importance or likelihood of events that are more easily recalled in memory, often because they are recent or emotionally charged.",
        "recommendations": [
            "Don't rely solely on memory. Look for actual data or statistics.",
            "Consider less vivid but potentially more relevant information.",
            "Ask: 'Is this easily recalled just because it was recent/dramatic?'",
            "Systematically list pros and cons or use a decision matrix."
        ],
         "keywords": ["remember that time", "just read", "vivid example", "fresh in my mind", "media coverage"]
    },
    "Sunk Cost Fallacy": {
        "definition": "Continuing a behavior or endeavor as a result of previously invested resources (time, money, or effort), even when it's clear that further investment is not rational.",
        "recommendations": [
            "Evaluate decisions based on *future* costs and benefits, not past investments.",
            "Ask: 'If I hadn't invested anything yet, would I still do this now?'",
            "Set predefined exit points or criteria for projects.",
            "Get an outside perspective from someone not invested in the past costs."
        ],
         "keywords": ["already invested", "too much time", "can't stop now", "see it through", "waste effort"]
    },
     "Neutral": {
        "definition": "The text does not strongly exhibit patterns associated with the specific cognitive biases the model is trained to detect.",
        "recommendations": [
            "Continue to be mindful of potential biases.",
            "Ensure clear and objective communication.",
            "Consider the broader context of the decision or discussion."
        ],
         "keywords": []
    },
     "Classifier Error": { # Placeholder for Classifier errors
        "definition": "An error occurred while running the bias classification model.",
        "recommendations": [
             "Ensure the model files exist and are accessible.",
             "Check the application logs for specific error messages."
         ]
    }
    # Add more biases
}

# List of biases the classifier will be trained on
CLASSIFIER_LABELS = ["Confirmation Bias", "Anchoring Bias", "Availability Heuristic", "Sunk Cost Fallacy", "Neutral"]

# Create mappings for the classifier model
label2id = {label: i for i, label in enumerate(CLASSIFIER_LABELS)}
id2label = {i: label for label, i in label2id.items()}
