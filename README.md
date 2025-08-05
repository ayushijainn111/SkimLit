# SkimLit – Scientific Abstract Sentence Classifier

SkimLit is a machine learning application that automatically processes biomedical/scientific abstracts and classifies each sentence into structured categories such as Background, Objective, Methods, Results, and Conclusions.

It is inspired by the [PubMed 200k RCT dataset](https://github.com/Franck-Dernoncourt/pubmed-rct) and research on automated abstract structuring.  
This implementation extends the original idea to work with live URLs from PubMed, arXiv, and similar sources.

## Features
- Fetch from URL – Provide a PubMed or arXiv link, and the app extracts the main text.
- Sentence Splitting – Breaks content into sentences using a custom tokenizer.
- Hybrid Model – Combines:
  - Text embeddings from the Universal Sentence Encoder (TensorFlow Hub)
  - Positional features (line number, total lines)
- Section Classification – Assigns each sentence to a structured section:
  - Background
  - Objective
  - Methods
  - Results
  - Conclusions
- Streamlit UI – User-friendly interface.
- Real-time predictions – Results grouped by section for easy reading.

## Tech Stack
- Language: Python 3.11  
- Libraries: TensorFlow 2.11, TensorFlow Hub, BeautifulSoup, NumPy, Pandas, Streamlit  
- Model: Trained hybrid deep learning model (`skimlit_hybrid_model.keras`)
- Deployment: Streamlit Cloud

## Project Structure
```
├── app.py # Streamlit app entry point
├── url_predictor.py # URL fetching, preprocessing, sentence classification
├── config.py # Configuration settings (paths, constants, labels)
├── utils.py # Helper functions
├── models/
│ └── skimlit_hybrid_model.keras # Trained model
├── requirements.txt # Python dependencies
├── runtime.txt # Python version for deployment
└── README.md
```

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/ayushijainn111/SkimLit.git
cd SkimLit
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run locally
```bash
streamlit run app.py
```

## References
- Dernoncourt, F., & Lee, J. Y. (2017). PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts. *arXiv preprint* [arXiv:1710.06071](https://arxiv.org/abs/1710.06071)
- [TensorFlow Hub: Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/4)

## License
This project is open-source and available under the MIT License.

Created by [Ayushi Jain](https://github.com/ayushijainn111)


