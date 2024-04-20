import PyPDF2
import spacy

class Document:
    def __init__(self, page_content='', metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}


def is_table_of_contents(text):
    # Define patterns commonly found in table of contents
    toc_patterns = ["Table of Contents", "Contents", "Table of Content", "Index", "Chapter"]

    # Check if the text matches any of the patterns
    for pattern in toc_patterns:
        if pattern.lower() in text.lower():
            return True
    return False


def extract_topics_and_related_text_from_pdf(pdf_file):
    # Initialize spaCy NLP pipeline
    nlp = spacy.load("en_core_web_sm")

    # Initialize list to store page content and metadata
    page_contents = []

    # Open the PDF file
    with open(pdf_file, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)

        # Iterate through each page in the PDF
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()

            # Skip text if it resembles a table of contents
            if is_table_of_contents(text):
                continue

            # Process the text using spaCy
            doc = nlp(text)

            # Initialize variables for topic extraction
            current_topic = ""
            related_text = ""

            # Identify headings and extract topics
            for token in doc:
                if token.pos_ == "NOUN" and token.dep_ == "compound":
                    if current_topic != "":
                        # Store related text with the previous topic
                        page_contents.append({'page_content': related_text.strip(), 'metadata': {'source': pdf_file, 'page': page_num}})
                        related_text = ""
                    current_topic = token.text
                elif token.pos_ != "PUNCT":
                    # Collect related text until a new topic is found
                    related_text += token.text + " "

            # Add the last topic and related text pair
            page_contents.append({'page_content': related_text.strip(), 'metadata': {'source': pdf_file, 'page': page_num}})

    return page_contents


def create_document_from_dict(doc_dict):
    page_content = doc_dict.get('page_content', '')
    metadata = doc_dict.get('metadata', {})
    return Document(page_content=page_content, metadata=metadata)


# Example usage
pdf_file = "data/Machine Learning- Step-by-Step Guide To Implement Machine Learning Algorithms with Python ( PDFDrive ).pdf"
page_contents = extract_topics_and_related_text_from_pdf(pdf_file)


for i in range(len(page_contents)):
    page_contents[i]['page_content'] = ' '.join(page_contents[i]['page_content'].split())


docs_list = []
for doc in page_contents:
    docs_list.append(create_document_from_dict(doc))
