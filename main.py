from url_predictor import URLModelPredictor
import config

def main():
    # Initialize predictor
    predictor = URLModelPredictor(config.MODEL_PATH)
    
    # Test with a single URL
    url = input("Enter URL to analyze: ")
    result = predictor.predict_from_url(url)
    
    if 'error' not in result:
        class_name = config.CLASS_LABELS.get(result['predicted_class'], 'Unknown')
        print(f"\nResults for: {result['url']}")
        print(f"Predicted Class: {class_name} (ID: {result['predicted_class']})")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Content Length: {result['content_length']} characters")
    else:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()