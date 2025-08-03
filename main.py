from url_predictor import TrulyLazyPredictor
import config

def main():
    predictor = TrulyLazyPredictor(config.MODEL_PATH)
    
    url = input("Enter URL to analyze: ")
    result = predictor.predict_from_url(url)
    
    if result.get("success", False):
        print(f"\nResults for: {result['url']}")
        print(f"Total Sentences: {result['total_sentences']}")
        print(f"Sections Found: {list(result['structured_sections'].keys())}")
    else:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()
