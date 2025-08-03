import requests
from bs4 import BeautifulSoup
import numpy as np
from typing import Dict
import config
import re

def fetch_url_content(url: str) -> str:
    """
    Fetch and extract text content from URL
    
    Args:
        url: URL to fetch content from
        
    Returns:
        Extracted text content
    """
    try:
        headers = {
            'User-Agent': config.USER_AGENT
        }
        response = requests.get(url, headers=headers, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        
        # Parse HTML and extract text
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
        
    except Exception as e:
        raise Exception(f"Error fetching URL content: {str(e)}")

def clean_text(text: str) -> str:
    """
    Clean and normalize text content
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:]', ' ', text)
    
    # Remove extra spaces
    text = text.strip()
    
    return text

def preprocess_text_for_model(text: str, max_chars: int, max_line_numbers: int, max_total_lines: int) -> Dict[str, np.ndarray]:
    """
    Preprocess text to match model input requirements
    
    Args:
        text: Raw text content
        max_chars: Maximum character length
        max_line_numbers: Maximum number of line features
        max_total_lines: Maximum number of total line features
        
    Returns:
        Dictionary with preprocessed inputs for the model
    """
    # Clean the text
    text = clean_text(text)
    
    # Split text into lines
    lines = text.split('\n')
    # Filter out empty lines
    lines = [line.strip() for line in lines if line.strip()]
    total_lines = len(lines)
    
    processed_inputs = {}
    
    # 1. Token inputs (for Universal Sentence Encoder)
    # Take first few sentences for token embedding
    sentences = text.split('.')[:config.MAX_SENTENCES_FOR_USE]
    sentences = [s.strip() for s in sentences if s.strip()]
    token_input = '. '.join(sentences)
    
    # Ensure we have some content
    if not token_input.strip():
        token_input = text[:500] if text else "empty content"
    
    processed_inputs['token_input'] = np.array([token_input])
    
    # 2. Character inputs
    char_text = text[:max_chars] if len(text) > max_chars else text
    if not char_text.strip():
        char_text = "empty"
    processed_inputs['char_inputs'] = np.array([char_text])
    
    # 3. Line number inputs (positional features)
    line_features = np.zeros(max_line_numbers)
    
    for i, line in enumerate(lines[:max_line_numbers]):
        if i < max_line_numbers:
            # Various line-level features
            line_features[i] = min(len(line) / 100.0, 1.0)  # Normalized line length
    
    # Add some aggregate features if we have remaining slots
    if max_line_numbers > len(lines):
        remaining_slots = max_line_numbers - len(lines)
        if remaining_slots > 0:
            # Average line length
            avg_line_length = np.mean([len(line) for line in lines]) if lines else 0
            line_features[len(lines)] = min(avg_line_length / 100.0, 1.0)
        if remaining_slots > 1:
            # Standard deviation of line lengths
            std_line_length = np.std([len(line) for line in lines]) if lines else 0
            line_features[len(lines) + 1] = min(std_line_length / 100.0, 1.0)
    
    processed_inputs['line_number_inputs'] = np.array([line_features])
    
    # 4. Total lines inputs (document-level features)
    total_line_features = np.zeros(max_total_lines)
    
    # Feature 0: Normalized total lines
    total_line_features[0] = min(total_lines / 100.0, 1.0)
    
    # Feature 1: Normalized text length
    total_line_features[1] = min(len(text) / 10000.0, 1.0)
    
    # Feature 2: Average words per line
    if total_lines > 0:
        avg_words_per_line = sum(len(line.split()) for line in lines) / total_lines
        total_line_features[2] = min(avg_words_per_line / 50.0, 1.0)
    
    # Feature 3: Ratio of short lines (< 50 chars)
    if total_lines > 0:
        short_lines = sum(1 for line in lines if len(line) < 50)
        total_line_features[3] = short_lines / total_lines
    
    # Feature 4: Ratio of long lines (> 200 chars)
    if total_lines > 0:
        long_lines = sum(1 for line in lines if len(line) > 200)
        total_line_features[4] = long_lines / total_lines
    
    # Feature 5: Number of sentences (approximate)
    sentence_count = len([s for s in text.split('.') if s.strip()])
    total_line_features[5] = min(sentence_count / 100.0, 1.0)
    
    # Feature 6: Number of words
    word_count = len(text.split())
    total_line_features[6] = min(word_count / 5000.0, 1.0)
    
    # Features 7-19: Can add more document-level features as needed
    # For now, leave them as zeros
    
    processed_inputs['total_lines_inputs'] = np.array([total_line_features])
    
    return processed_inputs

def print_prediction_results(result: Dict):
    """
    Pretty print prediction results
    
    Args:
        result: Result dictionary from prediction
    """
    if result['success']:
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        
        if 'url' in result:
            print(f"URL: {result['url']}")
        
        print(f"Predicted Class: {result['predicted_class_name']} (ID: {result['predicted_class']})")
        print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print(f"Content Length: {result['content_length']} characters")
        
        # Show all class probabilities
        print(f"\nAll Class Probabilities:")
        for i, prob in enumerate(result['all_probabilities']):
            class_name = config.CLASS_LABELS.get(i, f'Class_{i}')
            print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")
        
        if 'content_preview' in result:
            print(f"\nContent Preview:")
            print("-" * 30)
            print(result['content_preview'])
        
    else:
        print(f"\nERROR: {result['error']}")

def validate_url(url: str) -> bool:
    """
    Basic URL validation
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL appears valid
    """
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return url_pattern.match(url) is not None