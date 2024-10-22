# 📚 summarAIzer: AI-Powered Research Paper Summarizer

summarAIzer is a powerful tool that leverages AI to fetch, process, and summarize recent research papers from arXiv. It's designed to help researchers and enthusiasts stay up-to-date with the latest developments in science.

## 🌟 Features

- 🔍 Searches arXiv for recent papers based on keywords
- 📥 Downloads and extracts text from PDF papers
- 🤖 Uses Google's Gemini AI to generate concise summaries
- 📊 Supports multiple Gemini models for flexibility
- 🎨 Presents results in a clean, readable format

## 🚀 Quick Start

1. Install the required dependencies:
   ```
   pip install arxiv requests pypdf google-generativeai
   ```

2. Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

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

## 🛠️ Configuration

-`base_query`: Field to search (eg: `'exoplanets'`)
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

