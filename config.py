"""
Configuration file for LyriSent_Bert project.
Centralizes all configuration settings for consistency across modules.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_DIR = PROJECT_ROOT / 'trained_bert_model'
RESULTS_DIR = PROJECT_ROOT / 'comparison_results'

# Data files
EMOTION_WHEEL_CSV = DATA_DIR / 'EmotionWheelFinal (1).csv'
RADA_ANNOTATION_CSV = DATA_DIR / 'RadaNewAnnotation (1).csv'
SENTIMENT_COMPARISON_CSV = DATA_DIR / 'sentiment_comparison.csv'
NRC_LEXICON_TXT = DATA_DIR / 'NRC-Hashtag-Emotion-Lexicon-v0.2.txt'

# Emotion labels (Plutchik's 8 core emotions)
# IMPORTANT: This order must match across all modules!
EMOTION_LABELS = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger', 'Anticipation']

# BERT Configuration
BERT_MODEL_PATH = MODEL_DIR
PRETRAINED_MODEL_NAME = 'bert-base-uncased'
MAX_SEQ_LENGTH = 512  # Use same length for training and inference
BATCH_SIZE = 16
NUM_EPOCHS = 25
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# API Configuration
GENIUS_TOKEN = os.getenv('GENIUS_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = 'gpt-5.4'
OPENAI_PROMPTING_MODES = ('zero_shot', 'few_shot')

# CSV Column Configuration
TEXT_COLUMN = 'Lyrics'
EMOTION_COLUMNS_SLICE = slice(4, -1)
EXCLUDE_COLUMN = 'Unnamed: 11'