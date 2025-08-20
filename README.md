# Politics of Emotions (Congressional Tweets, Jan 6, 2021)

This project analyzes how emotional language is used strategically in U.S. congressional tweets around the Capitol Riot (Jan 6, 2021). We combine a RoBERTa model fine‑tuned on GoEmotions with explainability (attention maps + SHAP) to study macro trends (4‑day windows) and micro shifts (24‑hour windows), and to interpret rhetoric in case studies (Pelosi, McCarthy).

## Key Features
- **Emotion classification** with `SamLowe/roberta-base-go_emotions`
- **Label grouping** into interpretable categories (anger, sadness, joy, pride, fear, moral/empathic, surprise, disgust, neutral)
- **Time windows**: Macro (4 days before/after), Micro (24h before/during/after)
- **Explainability**: attention heatmaps + SHAP token attributions
- **Visualization**: party‑level distributions and deltas, figure-ready plots
