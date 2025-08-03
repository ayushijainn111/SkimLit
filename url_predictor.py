import tensorflow as tf
import numpy as np
import requests
from bs4 import BeautifulSoup
from typing import Dict
from collections import defaultdict
import config
import re
import os
import warnings

# Suppress protobuf warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

def simple_sent_tokenize(text):
    """Basic fallback sentence tokenizer (no NLTK dependency)"""
    return re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())

class TrulyLazyPredictor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.model_loaded = False
        print(f"✓ Predictor initialized (model will load on first prediction)")
        print(f"Model path: {model_path}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        file_size = os.path.getsize(model_path)
        print(f"Model file size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")

    def _load_model_when_needed(self):
        if not self.model_loaded:
            print("🔀 Loading model for first prediction...")
            print(f"TensorFlow version: {tf.__version__}")

            try:
                # Import tensorflow_hub with error handling
                import tensorflow_hub as hub
                print("✅ TensorFlow Hub imported")
                
                # Define custom layer with better error handling
                class USEWrapperLayer(tf.keras.layers.Layer):
                    def __init__(self, **kwargs):
                        super(USEWrapperLayer, self).__init__(**kwargs)
                        self.use = None
                        self._use_loaded = False

                    def call(self, inputs):
                        if not self._use_loaded:
                            print("📱 Loading Universal Sentence Encoder...")
                            try:
                                print("⏳ Downloading Universal Sentence Encoder (this might take a while)...")
                                self.use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
                                self._use_loaded = True
                                print("✅ Universal Sentence Encoder loaded!")
                            except Exception as e:
                                print(f"❌ Failed to load USE: {e}")
                                # Fallback: return zeros with correct shape
                                batch_size = tf.shape(inputs)[0]
                                return tf.zeros((batch_size, 512), dtype=tf.float32)
                        
                        try:
                            print("📅 Running USE embedding...")
                            return self.use(inputs)
                        except Exception as e:
                            print(f"❌ USE embedding failed: {e}")
                            # Fallback: return zeros
                            batch_size = tf.shape(inputs)[0]
                            return tf.zeros((batch_size, 512), dtype=tf.float32)

                # Load model with custom objects and error handling
                print("📂 Loading Keras model...")
                self.model = tf.keras.models.load_model(
                    self.model_path,
                    custom_objects={'USEWrapperLayer': USEWrapperLayer},
                    compile=False
                )
                print("✅ Model loaded successfully!")
                self.model_loaded = True
                
            except Exception as e:
                print(f"❌ Model loading failed: {e}")
                raise Exception(f"Failed to load model: {str(e)}")

    def fetch_content(self, url: str) -> str:
        print(f"🌐 Fetching: {url}")
        headers = {'User-Agent': config.USER_AGENT}
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for element in soup(["script", "style", "nav", "footer", "aside", "header"]):
                element.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines() if line.strip())
            text = ' '.join(lines)
            text = re.sub(r'\s+', ' ', text).strip()
            print(f"✅ Content fetched: {len(text)} characters")
            return text
        except Exception as e:
            raise Exception(f"Failed to fetch URL: {str(e)}")

    def preprocess_sentence(self, sentence: str) -> Dict[str, np.ndarray]:
        # Ensure TensorFlow operations are safe
        try:
            token_input = tf.convert_to_tensor([sentence], dtype=tf.string)
            char_input = tf.convert_to_tensor([sentence[:config.MAX_CHARS]], dtype=tf.string)
        except Exception as e:
            print(f"Warning: TensorFlow tensor conversion failed: {e}")
            # Fallback to numpy/list
            token_input = [sentence]
            char_input = [sentence[:config.MAX_CHARS]]

        line_features = np.zeros(config.MAX_LINE_NUMBERS, dtype=np.float32)
        line_features[0] = min(len(sentence) / 100.0, 1.0)

        total_features = np.zeros(config.MAX_TOTAL_LINES, dtype=np.float32)
        total_features[0] = 1.0
        total_features[1] = min(len(sentence) / 10000.0, 1.0)
        total_features[5] = 1.0
        total_features[6] = min(len(sentence.split()) / 5000.0, 1.0)

        return {
            'token_input': token_input,
            'char_inputs': char_input,
            'line_number_inputs': np.array([line_features], dtype=np.float32),
            'total_lines_inputs': np.array([total_features], dtype=np.float32)
        }

    def predict_from_url(self, url: str) -> Dict:
        try:
            self._load_model_when_needed()
            content = self.fetch_content(url)
            sentences = simple_sent_tokenize(content)[:100]  # Limit for speed

            all_inputs = [self.preprocess_sentence(s) for s in sentences]

            # Batch inputs with error handling
            try:
                batch_inputs = {
                    'line_number_inputs': np.vstack([x['line_number_inputs'] for x in all_inputs]),
                    'total_lines_inputs': np.vstack([x['total_lines_inputs'] for x in all_inputs]),
                    'token_input': tf.concat([x['token_input'] for x in all_inputs], axis=0) if isinstance(all_inputs[0]['token_input'], tf.Tensor) else [x['token_input'][0] for x in all_inputs],
                    'char_inputs': tf.concat([x['char_inputs'] for x in all_inputs], axis=0) if isinstance(all_inputs[0]['char_inputs'], tf.Tensor) else [x['char_inputs'][0] for x in all_inputs]
                }
            except Exception as e:
                print(f"Warning: Batch preparation failed: {e}")
                # Fallback processing
                raise Exception(f"Input preprocessing failed: {str(e)}")

            print("🤖 Running sentence-level prediction...")
            
            # Model prediction with error handling
            try:
                predictions = self.model.predict([
                    batch_inputs['line_number_inputs'],
                    batch_inputs['total_lines_inputs'],
                    batch_inputs['token_input'],
                    batch_inputs['char_inputs']
                ], verbose=0)
            except Exception as e:
                print(f"❌ Model prediction failed: {e}")
                raise Exception(f"Model prediction failed: {str(e)}")

            section_map = defaultdict(list)
            for i, pred in enumerate(predictions):
                label = np.argmax(pred)
                section = config.CLASS_LABELS.get(label, f'Class_{label}')
                section_map[section].append(sentences[i])

            return {
                'url': url,
                'success': True,
                'structured_sections': section_map,
                'total_sentences': len(sentences),
                'sentences': sentences
            }

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return {
                'url': url,
                'error': str(e),
                'success': False
            }

def predict_sentences_from_url(url: str):
    predictor = TrulyLazyPredictor(config.MODEL_PATH)
    return predictor.predict_from_url(url)

def main():
    try:
        print("🚀 Starting Truly Lazy URL Predictor")
        print("=" * 50)
        predictor = TrulyLazyPredictor(config.MODEL_PATH)

        while True:
            print("\n" + "🔍 URL Text Classification")
            print("-" * 30)
            url = input("Enter URL to analyze (or 'quit'): ").strip()

            if url.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if url and not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                print(f"🔗 Using: {url}")

            if not url:
                print("❌ Please enter a valid URL")
                continue

            result = predictor.predict_from_url(url)
            predictor.print_results(result)

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"💥 Fatal error: {e}")

if __name__ == "__main__":
    main()