# Get research papers from arXiv with specific keywords
import arxiv
import requests
import pypdf
import google.generativeai as genai
import os
import textwrap
import logging
from IPython.display import display, Markdown
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_filtered_research_papers(base_query,max_results=10,keywords=["demographics"]):

    # Create a combined query string
    keyword_query = " OR ".join(keywords)
    search_query = f"{base_query} AND ({keyword_query})"

    # Initialize the arXiv client
    client = arxiv.Client()

    # Search for papers in the arXiv database
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    return client.results(search)
    
# Function to download and extract text from a PDF
def extract_text_from_pdf(pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()  # Ensure the request was successful
    except requests.RequestException as e:
        logging.error(f"Failed to download PDF: {e}")
        return None
    
    # Save the PDF to a temporary file
    temp_file = 'temp.pdf'
    try:
        with open(temp_file, 'wb') as f:
            f.write(response.content)

        # Extract text from the PDF
        with open(temp_file, 'rb') as f:
            reader = pypdf.PdfReader(f)
            text = ''.join(page.extract_text() for page in reader.pages)
    except Exception as e:
        logging.error(f"Failed to extract text from PDF: {e}")
        return None
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    return text

def configure_genai(api_key):
    genai.configure(api_key=api_key)

def summarize_paper(model, prompt, custom_instructions, paper_text):
    try:
        response = model.generate_content(prompt + custom_instructions + paper_text)
        return textwrap.fill(response.text, width=80)
    except Exception as e:
        logging.error(f"Failed to generate summary: {e}")
        return None
    
def main(base_query, keywords, max_results=5, used_model='gemini-1.5-flash', API_KEY=None):
    if keywords is None:
        keywords = ["demographics"]

    max_results=max_results
    if used_model is None:
        models=['gemini-1.5-flash','gemini-1.5-flash-8b','gemini-1.5-pro','gemini-1.0-pro','gemini-pro','text-embedding-004','AQA',]
        used_model=models[0]


    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Call the function to retrieve filtered research papers
    paper_collection=get_filtered_research_papers(base_query,max_results=max_results,keywords=keywords)

    if API_KEY is None:
        API_KEY = os.environ.get("GEMINI_API_KEY")
    configure_genai(API_KEY)

    logging.info(f"Using model {used_model}")
    model = genai.GenerativeModel(used_model)

    # Custom instructions for the model, idea from Complexity discord.
    with open('prompt.txt', 'r') as f:
        prompt=f.read()
    custom_instructions = "You are an expert in "+ base_query+". Summarise this paper in 10 bullet points. Do not hallucinate."


    for p in paper_collection:
        logging.info(f"Processing paper: {p.title}")
        print("Title:", p.title)
        print("Authors:", textwrap.fill(", ".join(author.name for author in p.authors),width=80))
        print("-" * 80)
        try:
            html_url=p.pdf_url[:17]+'html'+p.pdf_url[20:]
            url_response = requests.get(html_url)
            pdf_text=url_response.text
            extra_instructions="The paper is provided as html. Check the formatting carefully.  The contents of the paper are:\n\n"
        except:
            pdf_text = extract_text_from_pdf(p.pdf_url)
            extra_instructions="The paper is provided as text extracted from pdf file. Check the formatting carefully.  The contents of the paper are:\n\n"
        

        summary = summarize_paper(model, prompt, custom_instructions, extra_instructions + pdf_text)
            
        # Add a horizontal rule to separate summaries
        print(summary)
        print("=" * 80)

if __name__ == "__main__":
    main()