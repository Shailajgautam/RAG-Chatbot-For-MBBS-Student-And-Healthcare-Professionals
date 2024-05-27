import os
import torch
import pdfplumber
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

BOOKS_PATH = 'data2/'
DB_FAISS_PATH = 'data2_vec/'


# ----------------------------------------------------------------------------
# Function to recursively find PDF files in a directory and its subdirectories
# ----------------------------------------------------------------------------
def find_pdf_files(directory):
    pdf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files


# -----------------------------------------------------------------
# Function to extract text from a PDF file
# -----------------------------------------------------------------
def extract_text_from_pdf(pdf_file):
    texts = []
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                try:
                    text = page.extract_text()
                    if text:
                        texts.append(text)
                except Exception as e:
                    print(f"Error extracting text from page in {pdf_file}: {e}")
    except Exception as e:
        print(f"Error loading {pdf_file}: {e}")
    return texts


# -----------------------------------------------------------------
# Create vector database
# -----------------------------------------------------------------
def create_vector_db():
    # Find all PDF files in the 'books' directory and its subdirectories
    pdf_files = find_pdf_files(BOOKS_PATH)

    print(f"Found PDF files: {pdf_files}")

    all_texts = []
    for pdf_file in pdf_files:
        print(f"Loading PDF file: {pdf_file}")  # Print the name of the PDF being loaded
        texts = extract_text_from_pdf(pdf_file)
        all_texts.extend(texts)

    if not all_texts:
        print("No documents were loaded. Exiting.")
        return

    concatenated_text = " ".join(all_texts)
    print(f"Concatenated text length: {len(concatenated_text)} characters")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(concatenated_text)

    print(f"Number of text chunks generated: {len(texts)}")

    if len(texts) == 0:
        print("No text chunks were generated. Exiting.")
        return

    # Check if GPU is available and use it
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': device})

    # Generate embeddings for the text chunks
    try:
        db = FAISS.from_texts(texts, embeddings)
        db.save_local(DB_FAISS_PATH)
        print("Vector database created and saved successfully.")
    except IndexError as e:
        print("Error generating embeddings or creating FAISS index:", e)
        print(f"Number of texts: {len(texts)}")
        if len(texts) > 0:
            print(f"Sample text: {texts[0]}")


if __name__ == "__main__":
    create_vector_db()
