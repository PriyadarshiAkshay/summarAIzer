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
import re
from bs4 import BeautifulSoup


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
    
    return list(client.results(search))

def clean_html(soup):
    # Remove unwanted sections
    sections_to_remove = [
        'references', 'bibliography', 'acknowledgement', 'acknowledgements',
        'citations', 'reference', 'acknowledgment', 'acknowledgments'
    ]

    for section in sections_to_remove:
        # Find and remove sections by headers
        for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            if section in header.get_text().lower():
                # Remove the header and all following siblings until the next header
                current = header
                while current and not current.find_next_sibling(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    next_elem = current.find_next_sibling()
                    if next_elem:
                        next_elem.decompose()
                    current = next_elem
                header.decompose()

    _html_text = soup.body.get_text()
    _html_text = _html_text.replace('  ', ' ')
    _html_text = _html_text.replace('\n\n\n', '\n')
    
    return _html_text

def get_html_text(url):
    url_response = requests.get(url)
    soup = BeautifulSoup(url_response.text, 'html.parser')
    return clean_html(soup)

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
        return response.text#textwrap.fill(response.text, width=80)
    except Exception as e:
        logging.error(f"Failed to generate summary: {e}")
        return None
    
def crosscheck_summary(model, summary, paper_text):
    """
    Validates the generated summary against the original paper text.
    Returns a confidence score and potential inconsistencies.
    """
    verification_prompt = """
    Compare the following summary against the original paper text and:
    1. Verify and add each bullet point's accuracy (scale 1-5)
    2. Identify any statements not supported by the original text. 
    3. Provide a confidence score (0-100%)
    4. Return all bullet points as it is if the summary is accurate. 
    5. The output should have 3 sections: Reasonably supported bullet points, Doubtful bullet points, and Confidence score on the summary.
    
    Summary to verify:
    {summary}
    
    Original text:
    {paper_text}
    """
    
    try:
        verification = model.generate_content(verification_prompt.format(
            summary=summary,
            paper_text=paper_text#[:10000]  # Limit text length to avoid token limits
        ))
        
        return verification.text#textwrap.fill(verification.text, width=80)
    except Exception as e:
        logging.error(f"Failed to crosscheck summary: {e}")
        return None
def summarize_from_url(url,base_query='exoplanets', used_model='gemini-1.5-flash', API_KEY=None):
    
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    if API_KEY is None:
        API_KEY = os.environ.get("GEMINI_API_KEY")
    configure_genai(API_KEY)

    logging.info(f"Using model {used_model}")
    model = genai.GenerativeModel(used_model)

    # Custom instructions for the model, idea from Complexity discord.
    with open('prompt.txt', 'r') as f:
        prompt=f.read()
    custom_instructions = "You are an expert in "+ base_query+". Summarise this paper in 10 bullet points. Do not hallucinate."


    
    if 'html' in url:
        html_url=url#[:17]+'html'+p.pdf_url[20:]
        pdf_text = get_html_text(html_url)

        extra_instructions="The paper is provided as text extracted from html. Check the formatting carefully.  The contents of the paper are:\n\n"
    else:
        pdf_text = extract_text_from_pdf(url)
        extra_instructions="The paper is provided as text extracted from pdf file. Check the formatting carefully.  The contents of the paper are:\n\n"
        

        summary = summarize_paper(model, prompt, custom_instructions, extra_instructions + pdf_text)
        
        if summary:
            verification_result = crosscheck_summary(model, summary, pdf_text)
            #print(summary)
            #print("\nVerification Results:")
            #display(Markdown("### Summary\n" + summary))
            display(Markdown(verification_result))
            print("=" * 80)

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

    print(f"Found {len(paper_collection)} papers.")

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
            pdf_text = get_html_text(html_url)

            extra_instructions="The paper is provided as text extracted from html. Check the formatting carefully.  The contents of the paper are:\n\n"
        except:
            pdf_text = extract_text_from_pdf(p.pdf_url)
            extra_instructions="The paper is provided as text extracted from pdf file. Check the formatting carefully.  The contents of the paper are:\n\n"
        

        summary = summarize_paper(model, prompt, custom_instructions, extra_instructions + pdf_text)
        
        if summary:
            verification_result = crosscheck_summary(model, summary, pdf_text)
            #print(summary)
            #print("\nVerification Results:")
            #display(Markdown("### Summary\n" + summary))
            display(Markdown(verification_result))
            print("=" * 80)

if __name__ == "__main__":
    main(base_query='exoplanets', keywords=["demographics"], max_results=5, used_model='gemini-1.5-flash', API_KEY=None)