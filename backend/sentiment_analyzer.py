import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import numpy as np
from typing import Dict, List, Tuple

class SentimentAnalyzer:
    """
    Advanced sentiment analysis using BERT-based models
    Supports multi-language sentiment detection with emotion classification
    """
    
    def __init__(self, model_name='bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name).to(self.device)
        
        # Sentiment classification head
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # Positive, Negative, Neutral
        ).to(self.device)
        
        # Emotion classification head
        self.emotion_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 8)  # Joy, Sadness, Anger, Fear, Surprise, Disgust, Trust, Anticipation
        ).to(self.device)
        
        self.sentiment_labels = ['negative', 'neutral', 'positive']
        self.emotion_labels = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'trust', 'anticipation']
        
        # Load pre-trained weights if available
        self._load_weights()
    
    def analyze(self, text: str, language: str = 'en') -> Dict:
        """
        Analyze sentiment and emotions in text
        
        Args:
            text: Input text to analyze
            language: Language code (en, es, fr, etc.)
        
        Returns:
            Dictionary with sentiment and emotion analysis
        """
        # Preprocess text
        text = self._preprocess_text(text)
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Sentiment prediction
        sentiment_logits = self.sentiment_classifier(embeddings)
        sentiment_probs = torch.softmax(sentiment_logits, dim=1)
        sentiment_idx = torch.argmax(sentiment_probs, dim=1).item()
        sentiment_score = sentiment_probs[0][sentiment_idx].item()
        
        # Emotion prediction
        emotion_logits = self.emotion_classifier(embeddings)
        emotion_probs = torch.softmax(emotion_logits, dim=1)
        
        # Get top 3 emotions
        top_emotions = torch.topk(emotion_probs, k=3, dim=1)
        emotions = [
            {
                'emotion': self.emotion_labels[idx],
                'score': score.item()
            }
            for idx, score in zip(top_emotions.indices[0], top_emotions.values[0])
        ]
        
        # Detect sarcasm
        sarcasm_score = self._detect_sarcasm(text, sentiment_probs)
        
        return {
            'text': text,
            'language': language,
            'sentiment': {
                'label': self.sentiment_labels[sentiment_idx],
                'score': sentiment_score,
                'confidence': sentiment_score,
                'distribution': {
                    label: prob.item()
                    for label, prob in zip(self.sentiment_labels, sentiment_probs[0])
                }
            },
            'emotions': emotions,
            'sarcasm': {
                'detected': sarcasm_score > 0.7,
                'score': sarcasm_score
            },
            'intensity': self._calculate_intensity(sentiment_probs, emotion_probs)
        }
    
    def analyze_batch(self, texts: List[str], language: str = 'en') -> List[Dict]:
        """Analyze multiple texts in batch for efficiency"""
        results = []
        
        # Process in batches of 32
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]
                
                # Predictions
                sentiment_logits = self.sentiment_classifier(embeddings)
                sentiment_probs = torch.softmax(sentiment_logits, dim=1)
                
                emotion_logits = self.emotion_classifier(embeddings)
                emotion_probs = torch.softmax(emotion_logits, dim=1)
            
            # Process each result
            for j, text in enumerate(batch):
                sentiment_idx = torch.argmax(sentiment_probs[j]).item()
                
                results.append({
                    'text': text,
                    'language': language,
                    'sentiment': {
                        'label': self.sentiment_labels[sentiment_idx],
                        'score': sentiment_probs[j][sentiment_idx].item()
                    },
                    'emotions': self._get_top_emotions(emotion_probs[j])
                })
        
        return results
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        # Remove URLs
        import re
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags (but keep the text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _detect_sarcasm(self, text: str, sentiment_probs: torch.Tensor) -> float:
        """
        Detect sarcasm using linguistic patterns and sentiment contradiction
        """
        sarcasm_indicators = [
            'yeah right', 'sure', 'totally', 'obviously',
            'great job', 'well done', 'fantastic', 'brilliant'
        ]
        
        text_lower = text.lower()
        
        # Check for sarcasm indicators
        indicator_count = sum(1 for indicator in sarcasm_indicators if indicator in text_lower)
        
        # Check for sentiment contradiction (e.g., positive words with negative context)
        # This is a simplified heuristic
        has_exclamation = '!' in text
        has_ellipsis = '...' in text
        
        # Calculate sarcasm score
        sarcasm_score = 0.0
        
        if indicator_count > 0:
            sarcasm_score += 0.3 * indicator_count
        
        if has_exclamation and sentiment_probs[0][0] > 0.3:  # Negative sentiment with exclamation
            sarcasm_score += 0.4
        
        if has_ellipsis:
            sarcasm_score += 0.2
        
        return min(sarcasm_score, 1.0)
    
    def _calculate_intensity(self, sentiment_probs: torch.Tensor, emotion_probs: torch.Tensor) -> float:
        """Calculate overall emotional intensity"""
        # Use entropy to measure intensity
        sentiment_entropy = -torch.sum(sentiment_probs * torch.log(sentiment_probs + 1e-10))
        emotion_entropy = -torch.sum(emotion_probs * torch.log(emotion_probs + 1e-10))
        
        # Lower entropy = higher intensity (more confident)
        max_sentiment_entropy = np.log(3)  # 3 sentiment classes
        max_emotion_entropy = np.log(8)    # 8 emotion classes
        
        sentiment_intensity = 1 - (sentiment_entropy / max_sentiment_entropy)
        emotion_intensity = 1 - (emotion_entropy / max_emotion_entropy)
        
        return ((sentiment_intensity + emotion_intensity) / 2).item()
    
    def _get_top_emotions(self, emotion_probs: torch.Tensor, k: int = 3) -> List[Dict]:
        """Get top k emotions with scores"""
        top_k = torch.topk(emotion_probs, k=k)
        
        return [
            {
                'emotion': self.emotion_labels[idx.item()],
                'score': score.item()
            }
            for idx, score in zip(top_k.indices, top_k.values)
        ]
    
    def _load_weights(self):
        """Load pre-trained weights if available"""
        try:
            checkpoint = torch.load('models/sentiment_model.pth', map_location=self.device)
            self.sentiment_classifier.load_state_dict(checkpoint['sentiment_classifier'])
            self.emotion_classifier.load_state_dict(checkpoint['emotion_classifier'])
            print("✅ Loaded pre-trained sentiment model")
        except FileNotFoundError:
            print("⚠️ No pre-trained weights found. Using base model.")
    
    def save_model(self, path: str = 'models/sentiment_model.pth'):
        """Save model weights"""
        torch.save({
            'sentiment_classifier': self.sentiment_classifier.state_dict(),
            'emotion_classifier': self.emotion_classifier.state_dict()
        }, path)
        print(f"✅ Model saved to {path}")
    
    def train(self, train_data: List[Tuple[str, str, List[str]]], epochs: int = 10):
        """
        Train the sentiment and emotion classifiers
        
        Args:
            train_data: List of (text, sentiment_label, emotion_labels)
            epochs: Number of training epochs
        """
        optimizer = torch.optim.Adam(
            list(self.sentiment_classifier.parameters()) + 
            list(self.emotion_classifier.parameters()),
            lr=2e-5
        )
        
        sentiment_criterion = nn.CrossEntropyLoss()
        emotion_criterion = nn.BCEWithLogitsLoss()
        
        self.sentiment_classifier.train()
        self.emotion_classifier.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for text, sentiment_label, emotion_labels in train_data:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=512,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :]
                
                # Forward pass
                sentiment_logits = self.sentiment_classifier(embeddings)
                emotion_logits = self.emotion_classifier(embeddings)
                
                # Calculate loss
                sentiment_target = torch.tensor([self.sentiment_labels.index(sentiment_label)]).to(self.device)
                emotion_target = torch.zeros(8).to(self.device)
                for emotion in emotion_labels:
                    emotion_target[self.emotion_labels.index(emotion)] = 1.0
                
                loss = sentiment_criterion(sentiment_logits, sentiment_target) + \
                       emotion_criterion(emotion_logits.squeeze(), emotion_target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_data):.4f}")