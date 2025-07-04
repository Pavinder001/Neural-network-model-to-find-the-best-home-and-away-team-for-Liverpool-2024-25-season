import itertools
import json
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import random
from tqdm import tqdm
import pandas as pd
import os

# Show current working directory
print("📂 Current working directory:", os.getcwd())

# Optional: Consistent random sampling
random.seed(42)

# Load model and preprocessing assets
model = load_model("lfc_match_predictor_softmax.keras")
scaler = joblib.load("lfc_input_scaler.pkl")
with open("bool_cols.json", "r") as f:
    bool_cols = json.load(f)

labels = ['Loss', 'Draw', 'Win']

# Predict probabilities for a given lineup
def predict_lineup(lineup, is_home, model, scaler, bool_cols):
    vector = np.zeros(len(bool_cols) + 1, dtype='float32')
    for i, player in enumerate(bool_cols):
        if player in lineup:
            vector[i] = 1.0
    vector[-1] = 1.0 if is_home else 0.0
    scaled = scaler.transform([vector])
    probs = model.predict(scaled, verbose=0)[0]
    return labels[np.argmax(probs)], dict(zip(labels, np.round(probs, 3)))

# Find top 5 lineups with specific GK constraint
def find_best_lineups(model, scaler, bool_cols, is_home=True, sample_size=3000):
    print(f"\n🔍 Finding best XI for {'Home' if is_home else 'Away'} matches...")

    all_players = bool_cols.copy()
    if len(all_players) < 11:
        raise ValueError("You need at least 11 players in bool_cols.")

    #  Include Alisson,  Exclude Kelleher
    valid_combos = [
        combo for combo in itertools.combinations(all_players, 11)
        if "Alisson" in combo and "Kelleher" not in combo
    ]

    if not valid_combos:
        raise ValueError("No valid lineups found with Alisson and without Kelleher.")

    sample_size = min(sample_size, len(valid_combos))
    sampled_lineups = random.sample(valid_combos, sample_size)

    results = []
    for lineup in tqdm(sampled_lineups, desc=f"Scoring {'Home' if is_home else 'Away'} lineups"):
        prediction, probs = predict_lineup(lineup, is_home, model, scaler, bool_cols)
        win_prob = probs['Win']
        results.append((win_prob, lineup, probs))

    results.sort(reverse=True, key=lambda x: x[0])
    return results[:5]

# Save to Excel
def save_results_to_excel(results, match_type):
    rows = []
    for i, (win_prob, lineup, probs) in enumerate(results, 1):
        row = {
            "XI #": i,
            "Win Prob": win_prob,
            "Loss Prob": probs['Loss'],
            "Draw Prob": probs['Draw'],
            "Players": ", ".join(lineup)
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    filename = f"{match_type}_lineups.xlsx"
    df.to_excel(filename, index=False)
    print(f" Excel file saved: {os.path.abspath(filename)}")

# Run simulations
top_home = find_best_lineups(model, scaler, bool_cols, is_home=True)
top_away = find_best_lineups(model, scaler, bool_cols, is_home=False)

# Print results
print("\n Top 5 Home Lineups (by Win Probability):")
for i, (prob, lineup, probs) in enumerate(top_home, 1):
    print(f"\nXI #{i} — Win: {prob:.3f}")
    print("Players:", lineup)
    print("Probabilities:", probs)

print("\n Top 5 Away Lineups (by Win Probability):")
for i, (prob, lineup, probs) in enumerate(top_away, 1):
    print(f"\nXI #{i} — Win: {prob:.3f}")
    print("Players:", lineup)
    print("Probabilities:", probs)

# Save to Excel
save_results_to_excel(top_home, "home")
save_results_to_excel(top_away, "away")