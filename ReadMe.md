# 📚 summarAIzer: Search & Summarize Academic Research

AI-powered research assistant that searches and summarizes academic papers using RAG (Retrieval Augmented Generation). Fetches real-time data from arXiv, cross-verifies information, and delivers accurate summaries to keep researchers informed of latest scientific developments.

## 🌟 Features

- 🔍 Searches arXiv for recent papers based on keywords
- 📥 Downloads and extracts text from PDF papers
- 🤖 Uses Google's Gemini AI to generate concise summaries
- 🤖 Uses AI as a judge to check inaccuracies in the summary.
- 📊 Supports multiple Gemini models for flexibility
- 🎨 Presents results in a clean, readable format

## 🚀 Quick Start

1. Install the required dependencies:
   ```
   pip install arxiv requests pypdf google-generativeai faiss-cpu sentence_transformers

   ```

2. Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

3. Run the script:
   ```python
   import summarAIzer as sa

   sa.main(
       base_query='exoplanet',
       keywords=['demographics'],
       max_results=5,
       used_model='gemini-1.5-flash',
       API_KEY='YOUR_API_KEY_HERE'
   )
   ```

## Example Output

>### Title: Reading Between the Rainbows: Comparative Exoplanet Characterisation through Molecule Agnostic Spectral Clustering
>Authors: Ilyana A. Guez, Mark Claire
>
>### Summary Analysis: 
>
>**Reasonably supported bullet points:**
>
>* **The paper presents a new method for characterizing exoplanet atmospheres called "Molecule Agnostic Spectral Clustering" (MASC).** [Accuracy: 5/5]
>* **The study uses 42 synthetic transit transmission spectra from the Virtual Planetary Laboratory Spectral Explorer (VPLSE).** [Accuracy: 5/5]
>* **The spectra are divided into bands, and the enclosed areas within these bands are compared to find patterns.** [Accuracy: 5/5]
>* **The HDBSCAN clustering algorithm is used to identify clusters in the data based on density.** [Accuracy: 5/5]
>* **The method was tested with different resolutions, and it was found that a resolving power of R∼similar-to\sim∼300 is necessary for useful results.** [Accuracy: 5/5]
>* **MASC successfully identifies clusters that correlate with atmospheric composition, such as the presence of CO2 and O2.** [Accuracy: 5/5]
>* **The method can also be used to estimate the ratio of O2 to CO2 in an exoplanet's atmosphere.** [Accuracy: 5/5]
>* **The authors test MASC by analyzing an unknown spectrum, accurately identifying it as a dry O2 atmosphere similar to Earth's.** [Accuracy: 5/5]
>* **The study concludes that MASC shows promise as a new method for characterizing exoplanet atmospheres, especially with future advancements in telescope technology.** [Accuracy: 5/5] 
>
>**Doubtful bullet points:**
>
>* **MASC focuses on identifying patterns in spectral features without relying on specific molecular signatures.** [Accuracy: 3/5]  The paper states it's "agnostic" but does analyze specific molecules like CO2 and O2 in its correlations. 
>* **The method was tested with different resolutions, and it was found that a resolving power of R∼similar-to\sim∼300 is necessary for useful results.** [Accuracy: 3/5] While the paper mentions testing at different resolutions, it doesn't explicitly state that R∼similar-to\sim∼300 is the "necessary" resolution. 
>
>
>**Confidence score:** 85%
>
>**Reason:** The summary accurately reflects the main points of the paper, including the method, its application, and the results. However, the two statements concerning the method's focus and the "necessary" resolution are somewhat inaccurate, implying a stronger statement than the paper provides. 


## 🛠️ Configuration

- `base_query`: Field to search (eg: `'exoplanets'`)
- `keywords`: List of keywords to filter papers (default: `['demographics']`)
- `max_results`: Maximum number of papers to process (default: 5)
- `used_model`: Gemini model to use (default: 'gemini-1.5-flash')
- `API_KEY`: Your Google AI API key (can also be set as an environment variable `GEMINI_API_KEY`)

## 📝 Output

For each paper, summarAIzer will print:
- Title
- Authors
- A 10-point summary of the paper's key findings

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](link-to-issues).

## 📄 License

This project is [MIT](link-to-license) licensed.

---

Happy summarizing! 🚀🪐

