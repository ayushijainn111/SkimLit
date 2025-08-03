# Model configuration
MODEL_PATH = "models/skimlit_hybrid_model.keras"
MAX_CHARS = 1000
MAX_LINE_NUMBERS = 15
MAX_TOTAL_LINES = 20

# Request configuration
REQUEST_TIMEOUT = 10
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

# Class labels (adjust based on your model)
CLASS_LABELS = {
     0: 'Background',
    1: 'Objective',
    2: 'Methods',
    3: 'Results',
    4: 'Conclusions'
    # Add your actual class labels
}