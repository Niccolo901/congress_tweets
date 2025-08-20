from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.pipelines import pipeline
import torch
from tqdm import tqdm
import os
import shap
import matplotlib.pyplot as plt
import seaborn as sns


class EmotionClassifier:
    def __init__(self, model_name="SamLowe/roberta-base-go_emotions", emotion_map=None):
        """
        Initialize the EmotionClassifier with a pre-trained model and tokenizer.

        Parameters
        ----------
        model_name : str
            Name of the pre-trained model to use.
        emotion_map : dict or None
            Mapping of raw emotion labels to grouped categories.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = 0 if torch.cuda.is_available() else -1

        self.pipeline = pipeline(
            task="text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            top_k=None  # Retrieve all emotion scores
        )

        self.emotion_map = emotion_map if emotion_map else {}

    def classify(self, text):
        """
        Classify the emotion of a single text and return the grouped emotion and score.

        Parameters
        ----------
        text : str
            The input text to classify.

        Returns
        -------
        tuple
            A tuple containing the top grouped emotion and its score.
        """
        preds = self.pipeline(text)[0]
        grouped_scores = {}
        for pred in preds:
            label = pred["label"].lower()
            score = pred["score"]
            group = self.emotion_map.get(label, "other")
            grouped_scores[group] = grouped_scores.get(group, 0) + score
        if not grouped_scores:
            return "other", 0.0
        top_group = max(grouped_scores, key=lambda x: float(grouped_scores[x]))
        return top_group, grouped_scores[top_group]

    def get_top_k_emotions(self, preds, k=3):
        """
        Retrieve the top-k raw emotions from prediction output.

        Parameters
        ----------
        preds : list of dict
            List of emotion predictions with scores.
        k : int
            Number of top emotions to return.

        Returns
        -------
        list of dict
            Top-k emotions sorted by score.
        """
        return sorted(preds, key=lambda x: x["score"], reverse=True)[:k]

    def classify_top(self, texts, k=1, show_progress=True):
        """
        Classify a list of texts and return the top emotion for each.

        Parameters
        ----------
        texts : list of str
            List of input texts to classify.
        k : int
            Number of top emotions to consider (default is 1).
        show_progress : bool
            Whether to display a progress bar.

        Returns
        -------
        list of dict
            List of classification results with raw label, score, and group.
        """
        results = []
        iterator = tqdm(texts, desc="Classifying emotions") if show_progress else texts

        for text in iterator:
            try:
                preds = self.pipeline(text)[0]
                top = self.get_top_k_emotions(preds, k=k)[0]
                label = top["label"].lower()
                group = self.emotion_map.get(label, "other")
                results.append({
                    "raw_label": label,
                    "score": top["score"],
                    "group": group
                })
            except Exception as e:
                print("Error:", e)
                results.append({
                    "raw_label": None,
                    "score": None,
                    "group": None
                })

        return results

    def classify_and_save(self, df, save_path, k=1, show_progress=True):
        """
        Classify emotions in a DataFrame and save the results to a CSV file.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing a 'text' column to classify.
        save_path : str
            Path to save the resulting CSV file.
        k : int
            Number of top emotions to consider (default is 1).
        show_progress : bool
            Whether to display a progress bar.

        Returns
        -------
        pd.DataFrame
            Updated DataFrame with emotion classification results.
        """
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        results = self.classify_top(df["text"], k=k, show_progress=show_progress)
        df = df.copy()
        df["emotion_label"] = [r["raw_label"] for r in results]
        df["emotion_score"] = [r["score"] for r in results]
        df["emotion_group"] = [r["group"] for r in results]
        df.to_csv(save_path, index=False)
        print(f"Saved classified tweets to {save_path}")
        return df

    def analyze_with_attention(self, sentence):
        """
        Analyze a sentence using attention weights and SHAP explainability.

        Parameters
        ----------
        sentence : str
            Input sentence to analyze.

        Returns
        -------
        None
        """
        print(f"\nAnalyzing: {sentence}\n")

        # Tokenize input
        inputs = self.tokenizer(sentence, return_tensors="pt")
        device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass with attention
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attentions = outputs.attentions
            logits = outputs.logits

        # Emotion prediction
        id2label = self.model.config.id2label
        probs = torch.softmax(logits, dim=1)[0]
        topk = torch.topk(probs, k=5)

        print("Top predicted emotions:")
        for score, idx in zip(topk.values, topk.indices):
            raw_emotion = id2label[idx.item()]
            mapped_emotion = self.emotion_map.get(raw_emotion.lower(), raw_emotion) if self.emotion_map else raw_emotion
            print(f"  {raw_emotion} → {mapped_emotion}: {score.item():.3f}")

        # Attention heatmap
        last_layer_attention = attentions[-1][0]  # (num_heads, seq_len, seq_len)
        attention_weights = last_layer_attention[0].detach().cpu().numpy()  # head 0
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labeled_tokens = [f"{i}:{tok}" for i, tok in enumerate(tokens)]

        plt.figure(figsize=(16, 12))
        sns.heatmap(attention_weights,
                    xticklabels=labeled_tokens,
                    yticklabels=labeled_tokens,
                    cmap="viridis")
        plt.title("Attention Heatmap (last layer, head 0)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # SHAP analysis
        print("\nRunning SHAP explainability...")
        explainer = shap.Explainer(self.pipeline)
        shap_values = explainer([sentence])
        shap.plots.text(shap_values[0])


