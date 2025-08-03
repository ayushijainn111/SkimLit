import tensorflow as tf
import numpy as np
import requests
from bs4 import BeautifulSoup
from typing import Dict
from collections import defaultdict
import config
import re


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

    def _load_model_when_needed(self):
        if not self.model_loaded:
            print("🔀 Loading model for first prediction...")

            import tensorflow_hub as hub
            from tensorflow.keras import layers

            class USEWrapperLayer(layers.Layer):
                def __init__(self, **kwargs):
                    super(USEWrapperLayer, self).__init__(**kwargs)
                    self.use = None
                    self._use_loaded = False

                def call(self, inputs):
                    if not self._use_loaded:
                        print("📱 Loading Universal Sentence Encoder...")
                        print("⏳ Downloading Universal Sentence Encoder (this might take a while)...")
                        self.use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
                        self._use_loaded = True
                        print("✅ Universal Sentence Encoder loaded!")
                    print("📅 Running USE embedding...")
                    return self.use(inputs)

            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects={'USEWrapperLayer': USEWrapperLayer},
                compile=False
            )
            print("✅ Model loaded successfully!")
            self.model_loaded = True

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
        token_input = tf.convert_to_tensor([sentence], dtype=tf.string)
        char_input = tf.convert_to_tensor([sentence[:config.MAX_CHARS]], dtype=tf.string)

        line_features = np.zeros(config.MAX_LINE_NUMBERS)
        line_features[0] = min(len(sentence) / 100.0, 1.0)

        total_features = np.zeros(config.MAX_TOTAL_LINES)
        total_features[0] = 1.0
        total_features[1] = min(len(sentence) / 10000.0, 1.0)
        total_features[5] = 1.0
        total_features[6] = min(len(sentence.split()) / 5000.0, 1.0)

        return {
            'token_input': token_input,
            'char_inputs': char_input,
            'line_number_inputs': np.array([line_features]),
            'total_lines_inputs': np.array([total_features])
        }

    def predict_from_url(self, url: str) -> Dict:
        try:
            self._load_model_when_needed()
            content = self.fetch_content(url)
            sentences = simple_sent_tokenize(content)[:100]  # Limit for speed

            all_inputs = [self.preprocess_sentence(s) for s in sentences]

            batch_inputs = {
                'line_number_inputs': np.vstack([x['line_number_inputs'] for x in all_inputs]),
                'total_lines_inputs': np.vstack([x['total_lines_inputs'] for x in all_inputs]),
                'token_input': tf.concat([x['token_input'] for x in all_inputs], axis=0),
                'char_inputs': tf.concat([x['char_inputs'] for x in all_inputs], axis=0)
            }

            print("🤖 Running sentence-level prediction...")
            predictions = self.model.predict([
                batch_inputs['line_number_inputs'],
                batch_inputs['total_lines_inputs'],
                batch_inputs['token_input'],
                batch_inputs['char_inputs']
            ], verbose=0)

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

    def print_results(self, result: Dict):
        print("\n" + "=" * 70)
        if result['success']:
            print("🎯 STRUCTURED SENTENCE CLASSIFICATION")
            print("=" * 70)
            print(f"🔗 URL: {result['url']}")
            print(f"✂ Total Sentences: {result['total_sentences']}")
            print("\n🗂️ Section-wise Output:")

            for section, sentences in result['structured_sections'].items():
                print(f"\n📘 {section}")
                print("-" * (len(section) + 3))
                for sent in sentences:
                    print(f"- {sent}")
        else:
            print("❌ PREDICTION FAILED")
            print("=" * 70)
            print(f"🔗 URL: {result['url']}")
            print(f"💥 Error: {result['error']}")


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
