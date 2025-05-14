import csv
import random
from bias_definitions import CLASSIFIER_LABELS # Use labels defined for classifier

PLACEHOLDERS = {
    "topic": ["market share", "customer retention", "product viability", "AI ethics", "team performance", "investment strategy"],
    "number": ["$75,000", "20%", "500 users", "Q3 targets", "1.5 million", "six figures"],
    "event": ["the recent merger", "that viral campaign", "the budget cut", "the server outage", "last year's conference", "the negative press"],
    "product": ["our new SaaS platform", "the competitor's offering", "this open-source library", "the legacy system", "marketing automation tool"],
    "project": ["Project Phoenix", "the website redesign", "the international expansion", "the research phase", "our main initiative"],
    "amount": ["$20k and three months", "a huge amount of resources", "years of effort", "significant political capital", "the initial seed funding"],
    "effort": ["all this planning", "the team's hard work", "the overtime hours", "this complex integration", "the stakeholder alignment"],
    "person": ["Alice", "Bob", "the CEO", "our client", "the new hire", "the consultant"],
    "opinion": ["clearly the best option", "likely to succeed", "a risky move", "underestimated", "overrated"],
    "justification": ["based on the initial report", "because everyone agrees", "as the expert suggested", "due to the urgency", "following standard procedure"],
    "emotion": ["concerned", "excited", "skeptical", "confident", "disappointed"],
    "action": ["proceed with the plan", "delay the decision", "gather more data", "consult an expert", "pivot our strategy"]
}

TEMPLATES = {
    "Confirmation Bias": [
        "This new data perfectly aligns with my hypothesis about {topic}.",
        "I specifically looked for information proving {topic} and found plenty.",
        "Ignoring the contradictory results, the overall picture confirms my view on {topic}.",
        "As {person} and I suspected, {topic} is {opinion}. The evidence we gathered supports this.",
        "Let's focus on the data points that reinforce our belief in {product}'s success.",
        "I'm {emotion} that the analysis validated what I already believed regarding {project}.",
    ],
    "Anchoring Bias": [
        "They mentioned {number} first, so anything around that feels acceptable for {project}.",
        "The initial projection was {number}. Even though things changed, it's hard to ignore that figure.",
        "Compared to the list price of {number}, the discount makes {product} seem like a bargain.",
        "Let's use {number} as the starting point for our discussion on {topic}.",
        "Since the competitor is valued at {number}, our own valuation should be in that ballpark.",
        "My first impression, based on the {number} figure, strongly influences my view of {topic}.",
    ],
    "Availability Heuristic": [
        "After hearing about {event} from {person}, I'm really {emotion} it could happen to us.",
        "That one vivid example of {product} failing makes me hesitant, despite the stats.",
        "I keep thinking about {event}; it must be a bigger risk than the data suggests for {project}.",
        "News about {topic} is everywhere lately; it's definitely the most important trend.",
        "My gut feeling, likely influenced by the recent {event}, tells me to {action}.",
        "Remembering {person}'s story about {topic} makes this seem more likely than it probably is.",
    ],
    "Sunk Cost Fallacy": [
        "We've poured {amount} into {project}, we have to {action} now or it's all wasted.",
        "Given the {effort} already expended on {product}, abandoning it isn't an option.",
        "I know {project} is struggling, but think of the {amount} invested. Let's add more resources.",
        "It feels wrong to stop {project} after all the {effort}. We need to justify the initial decision.",
        "We're pot-committed to {topic} after spending {amount}.",
        "{Person} feels we must continue {project} because of the past {effort}, despite the poor outlook.",
    ],
    "Neutral": [
        "The quarterly report shows a {number} change in {topic}.",
        "We need to schedule a meeting to discuss the next steps for {project}.",
        "Let's analyze the feedback received regarding {product}.",
        "The agenda includes reviewing {topic} and planning for {event}.",
        "Data collection for {project} is ongoing as per the timeline.",
        "{Person} presented the findings on {topic}.",
        "The team evaluated several options before recommending {action}.",
        "Market research indicates potential challenges in {topic}.",
        "Please review the document outlining the strategy for {product}.",
        "The current budget allocation for {project} is {number}.",
    ]
}

def fill_template(template):
    """Fills placeholders in a template string using expanded placeholders."""
    text = template
    placeholders_in_template = set(ph[1:-1] for ph in text.split() if ph.startswith('{') and ph.endswith('}'))
    replacements = {}
    for ph_key in placeholders_in_template:
        if ph_key in PLACEHOLDERS:
            replacements[ph_key] = random.choice(PLACEHOLDERS[ph_key])
        else:
            replacements[ph_key] = ph_key # Keep placeholder if key not found
    for key, value in replacements.items():
         text = text.replace(f"{{{key}}}", value)
    text = text.replace(" a $", " $").replace(" a Q", " Q").replace(" a 1", " 1").replace(" a 7", " 7")
    return text

def generate_data(num_samples_per_label=200, output_file="generated_bias_data.csv"):
    """Generates synthetic data and saves to CSV."""
    data = []
    print(f"Generating synthetic data for labels: {CLASSIFIER_LABELS}")
    for label in CLASSIFIER_LABELS:
        if label in TEMPLATES:
            templates = TEMPLATES[label]
            if not templates:
                 print(f"Warning: No templates found for label: {label}")
                 continue
            count = 0
            indices = list(range(len(templates)))
            while count < num_samples_per_label:
                if not indices:
                    indices = list(range(len(templates)))
                    random.shuffle(indices)
                idx = indices.pop()
                template = templates[idx]
                prefix = random.choice(["", "Thinking about it, ", "Based on the discussion, ", "My assessment is that "])
                suffix = random.choice(["", " What do you think?", " Let's discuss this further.", " This needs careful consideration."])
                text = fill_template(template)
                if label != "Neutral" and random.random() < 0.3:
                     text = random.choice([prefix + text, text + suffix, prefix + text + suffix])
                data.append({"text": text.strip(), "label": label})
                count += 1
        else:
            print(f"Warning: Label '{label}' not found in TEMPLATES dictionary.")
    if not data:
        print("Error: No data generated. Check templates and labels.")
        return
    random.shuffle(data)
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['text', 'label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Enhanced synthetic data generated ({len(data)} samples) and saved to {output_file}")
    except IOError as e:
        print(f"Error writing to CSV file {output_file}: {e}")

if __name__ == "__main__":
    generate_data(num_samples_per_label=300) # Generate data for BERT
