from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chainlit as cl
import torch

DB_FAISS_PATH = 'vector_db/'

# Custom Prompt Template for QA Retrieval
custom_prompt_template = """You are an expert assistant. Answer the following question based on the provided context:
Context: {context}
Question: {question}

Provide the most relevant answer below:
Helpful answer:
"""


# -----------------------------------------------------------------
# Function to Load the LLM (Large Language Model) on GPU
# -----------------------------------------------------------------
def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.7,  # Adjust temperature for better response quality
        local=True
    )
    return llm


# -----------------------------------------------------------------
# Function to Set Up the Custom Prompt for QA Retrieval
# -----------------------------------------------------------------
def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt


# -----------------------------------------------------------------
# Function to Create the QA Retrieval Chain
# -----------------------------------------------------------------
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain


# -----------------------------------------------------------------
# Function to Get Embeddings, Using GPU If Available
# -----------------------------------------------------------------
def get_embeddings():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device}
    )
    return embeddings


# -----------------------------------------------------------------
# Function to Set Up the QA Bot
# -----------------------------------------------------------------
def qa_bot():
    embeddings = get_embeddings()
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa


# -----------------------------------------------------------------
# Function to Process Uploaded PDF Files
# -----------------------------------------------------------------
async def process_pdfs(files):
    documents = []
    for file in files:
        file_path = file.path
        loader = PyMuPDFLoader(file_path)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()
    pdf_db = FAISS.from_documents(chunks, embeddings)

    return pdf_db


# -----------------------------------------------------------------
# Function to Process Multiple URLs
# -----------------------------------------------------------------
async def process_urls(urls):
    documents = []
    for url in urls:
        loader = WebBaseLoader(url)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()
    url_db = FAISS.from_documents(chunks, embeddings)

    return url_db


# -----------------------------------------------------------------
# Function to Merge FAISS Indexes
# -----------------------------------------------------------------
def merge_faiss_indexes(original_index, new_index):
    if original_index is None:
        return new_index
    else:
        original_index.merge_from(new_index)
        return original_index


# -----------------------------------------------------------------
# Event Handler for Bot Startup
# -----------------------------------------------------------------
@cl.on_chat_start
async def start():
    cl.user_session.set("pdf_db", None)
    cl.user_session.set("url_db", None)
    cl.user_session.set("user_choice", None)
    await cl.Message(
        content="Welcome! Please choose an option:\n\n"
                "1. **Chat with the chatbot**\n"
                "2. **Upload PDF files**\n"
                "3. **Input URLs**\n"
                "4. **Use all functionalities (chat, PDFs, and URLs)**\n\n"
                "Type '1', '2', '3', or '4' to proceed. You can switch modes anytime."
    ).send()


# -----------------------------------------------------------------
# Main Event Handler for Incoming Messages
# -----------------------------------------------------------------
@cl.on_message
async def main(message: cl.Message):
    user_choice = cl.user_session.get("user_choice")
    choice = message.content.strip().lower()

    # -----------------------------------------------------------------
    # Mode 1: Initialize the QA Bot for Chatting
    # -----------------------------------------------------------------
    if choice == "1":
        chain = qa_bot()
        cl.user_session.set("chain", chain)
        cl.user_session.set("user_choice", "chat")
        await cl.Message(
            content="Hello! I'm here to assist you. What would you like to know today?"
        ).send()

    # -----------------------------------------------------------------
    # Mode 2: Handle PDF Upload
    # -----------------------------------------------------------------
    elif choice == "2":
        pdf_db = cl.user_session.get("pdf_db")
        if pdf_db:
            cl.user_session.set("user_choice", "pdf_select")
            await cl.Message(
                content="Would you like to use the previously uploaded PDFs or upload new ones?\n"
                        "Type 'previous' to use previously uploaded PDFs or 'new' to upload new PDFs."
            ).send()
        else:
            cl.user_session.set("user_choice", "pdf_upload")
            files = await cl.AskFileMessage(
                content="Please upload PDF files (up to 20).",
                accept=["application/pdf"],
                max_size_mb=5,
                max_files=20
            ).send()
            if files:
                pdf_db = await process_pdfs(files)
                cl.user_session.set("pdf_db", pdf_db)
                chain = retrieval_qa_chain(load_llm(), set_custom_prompt(), pdf_db)
                cl.user_session.set("chain", chain)
                cl.user_session.set("user_choice", "pdf_chat")
                await cl.Message(
                    content="The PDFs have been processed. You can now ask questions based on the uploaded PDFs."
                ).send()
            else:
                await cl.Message(content="No files uploaded. Please upload PDF files.").send()

    elif choice == "previous" and (user_choice == "pdf_select" or user_choice == "combined_chat"):
        pdf_db = cl.user_session.get("pdf_db")
        chain = retrieval_qa_chain(load_llm(), set_custom_prompt(), pdf_db)
        cl.user_session.set("chain", chain)
        cl.user_session.set("user_choice", "pdf_chat")
        await cl.Message(
            content="Using previously uploaded PDFs. You can now ask questions based on the uploaded PDFs."
        ).send()

    elif choice == "new" and (user_choice == "pdf_select" or user_choice == "pdf_chat"):
        cl.user_session.set("user_choice", "pdf_upload")
        files = await cl.AskFileMessage(
            content="Please upload PDF files (up to 20).",
            accept=["application/pdf"],
            max_size_mb=5,
            max_files=20
        ).send()
        if files:
            pdf_db = await process_pdfs(files)
            cl.user_session.set("pdf_db", pdf_db)
            chain = retrieval_qa_chain(load_llm(), set_custom_prompt(), pdf_db)
            cl.user_session.set("chain", chain)
            cl.user_session.set("user_choice", "pdf_chat")
            await cl.Message(
                content="The PDFs have been processed. You can now ask questions based on the uploaded PDFs."
            ).send()
        else:
            await cl.Message(content="No files uploaded. Please upload PDF files.").send()

    # -----------------------------------------------------------------
    # Mode 3: Handle URL Input
    # -----------------------------------------------------------------
    elif choice == "3":
        url_db = cl.user_session.get("url_db")
        if url_db:
            cl.user_session.set("user_choice", "url_select")
            await cl.Message(
                content="Would you like to use the previously input URLs or input new ones?\n"
                        "Type 'previous' to use previously input URLs or 'new' to input new URLs."
            ).send()
        else:
            cl.user_session.set("user_choice", "url_input")
            await cl.Message(
                content="Please input the URLs you would like to process (one per line, up to 20)."
            ).send()

    elif choice == "previous" and (user_choice == "url_select" or user_choice == "combined_chat"):
        url_db = cl.user_session.get("url_db")
        chain = retrieval_qa_chain(load_llm(), set_custom_prompt(), url_db)
        cl.user_session.set("chain", chain)
        cl.user_session.set("user_choice", "url_chat")
        await cl.Message(
            content="Using previously input URLs. You can now ask questions based on the content from the URLs."
        ).send()

    elif choice == "new" and (user_choice == "url_select" or user_choice == "url_chat"):
        cl.user_session.set("user_choice", "url_input")
        await cl.Message(
            content="Please input the URLs you would like to process (one per line, up to 20)."
        ).send()

    elif user_choice == "url_input":
        urls = message.content.strip().split()
        if all(url.startswith("http://") or url.startswith("https://") for url in urls):
            if len(urls) > 20:
                await cl.Message(content="Please provide up to 20 URLs.").send()
            else:
                url_db = await process_urls(urls)
                cl.user_session.set("url_db", url_db)
                chain = retrieval_qa_chain(load_llm(), set_custom_prompt(), url_db)
                cl.user_session.set("chain", chain)
                cl.user_session.set("user_choice", "url_chat")
                await cl.Message(
                    content="The URLs have been processed. You can now ask questions based on the content from the URLs."
                ).send()
        else:
            await cl.Message(content="Invalid URLs. Please input valid URLs starting with http:// or https://").send()

    # -----------------------------------------------------------------
    # Mode 4: Handle Chat with Chatbot, PDFs, and URLs
    # -----------------------------------------------------------------
    elif choice == "4":
        pdf_db = cl.user_session.get("pdf_db")
        url_db = cl.user_session.get("url_db")

        if not pdf_db and not url_db:
            await cl.Message(
                content="Please make sure you have uploaded PDFs and input URLs before switching to this mode."
            ).send()
        else:
            # Load the original FAISS index
            combined_db = FAISS.load_local(DB_FAISS_PATH, get_embeddings(), allow_dangerous_deserialization=True)

            # Merge PDF index with the combined index
            if pdf_db:
                combined_db = merge_faiss_indexes(combined_db, pdf_db)

            # Merge URL index with the combined index
            if url_db:
                combined_db = merge_faiss_indexes(combined_db, url_db)

            combined_chain = retrieval_qa_chain(load_llm(), set_custom_prompt(), combined_db)
            cl.user_session.set("chain", combined_chain)
            cl.user_session.set("user_choice", "combined_chat")

            # Set pdf_db and url_db in the user session for future use
            cl.user_session.set("pdf_db", pdf_db)
            cl.user_session.set("url_db", url_db)

            await cl.Message(
                content="You can now chat with the assistant and ask questions based on the uploaded PDFs, URLs, and the chatbot dataset."
            ).send()

    # -----------------------------------------------------------------
    # Handle Incoming Chat Messages for the Current Mode
    # -----------------------------------------------------------------
    elif user_choice in ["chat", "pdf_chat", "url_chat", "combined_chat"]:
        chain = cl.user_session.get("chain")
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        res = await chain.ainvoke(message.content, callbacks=[cb])
        answer = res["result"]
        sources = res["source_documents"]

        # Filter out duplicate documents
        unique_sources = set()
        filtered_sources = []
        for doc in sources:
            doc_id = doc.metadata.get("file_path")
            if doc_id not in unique_sources:
                unique_sources.add(doc_id)
                filtered_sources.append(doc)

        if filtered_sources:
            source_texts = "\n\nSources:"
            for doc in filtered_sources:
                source = doc.metadata.get('source', 'Dataset')
                source_texts += f"\n- {source}"
            answer = f"{answer}\n{source_texts}"
        else:
            answer += "\n\nNo sources found"

        await cl.Message(content=answer).send()

    # -----------------------------------------------------------------
    # Invalid Choice Handling
    # -----------------------------------------------------------------
    else:
        await cl.Message(
            content="Invalid choice. Please type '1' for chat, '2' for PDF upload, '3' for URL input, or '4' to use all at once."
        ).send()
