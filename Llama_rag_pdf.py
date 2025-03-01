import os
import tempfile

import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

# This is the system prompt.
system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

cyberpunk_css = """
<style>
body {
    background-color: #1A1A1A;
    font-family: 'Courier New', monospace;
}
.stApp {
    background-color: #1A1A1A;
    color: #00FFFF;
}
h1, h2, h3 {
    color: #FF00FF;
    text-shadow: 0 0 5px #FF00FF, 0 0 10px #FF00FF;
}
button {
    background-color: #00FFFF;
    color: #1A1A1A;
    border: 2px solid #FF00FF;
    border-radius: 5px;
    padding: 8px 16px;
    font-weight: bold;
    text-shadow: 0 0 3px #00FFFF;
    transition: all 0.3s ease;
}
button:hover {
    background-color: #FF00FF;
    color: #00FFFF;
    box-shadow: 0 0 10px #FF00FF;
}
.stTextArea textarea {
    background-color: #2A2A2A;
    color: #00FFFF;
    border: 2px solid #FF00FF;
    border-radius: 5px;
    box-shadow: 0 0 5px #FF00FF;
}
.stFileUploader {
    background-color: #2A2A2A;
    border: 2px dashed #00FFFF;
    border-radius: 5px;
    padding: 10px;
}
.stExpander {
    background-color: #2A2A2A;
    border: 1px solid #FF00FF;
    border-radius: 5px;
    color: #00FFFF;
}
.stSuccess {
    color: #00FFFF;
    text-shadow: 0 0 5px #00FFFF;
}
.sidebar .sidebar-content {
    background-color: #2A2A2A;
    border-right: 2px solid #FF00FF;
}
</style>
"""
# Converts the uploaded file into -> List of Documents which is 'langchain abstraction' using 'uploaded_file' from streamlit.
def process_document(uploaded_file: UploadedFile) -> list[Document]:

    # Store uploaded file as a temp file.
    # We have permission to write in bytes ->'wb'.
    # We are deleting the file mannualy -> 'False'.
    # Here we are writing the contents of the file into a temp file.
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())

    # Loaded the file to temp location.
    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    # After loading the file we are gonna delete the temp file, because we don't need it and the file is in the memory of the program.
    os.unlink(temp_file.name)  # Delete temp file

    # For creating chunks of data so that it can be easily used with embedding model, LLM, and for reranker model.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, 
        # For semantic meaning between the chunks.
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)

# For creating Vector Database.
# To store data in a Vector Database we need to convert into data in vector embeddings, using -> OllamaEmbeddingFunction.
def get_vector_collection() -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        # The server for ollama runs on this port -> 11434, api,embeddings are the endpoint.
        url="http://localhost:11434/api/embeddings",
        # We are using this model becuase it has very large context window:8192.It has high dimensions:768, this means the semantic meaning created by this embedding model is preety good.
        model_name="nomic-embed-text:latest",
    )

    #We are persiting data in our local disk instead of memory.This is the path 'rag-chroma' for the local disk.It stores the vector embeddings.It uses SQLite by default.
    chroma_client = chromadb.PersistentClient(path="./rag-chroma")
    return chroma_client.get_or_create_collection(
        # Here we are creating tables on relational databases.
        # We are creating a table called -> 'llama_rag_app'.
        name="llama_rag_app",
        embedding_function=ollama_ef,
        # We are using 'cosine' becuase we are concerned about semantic meaning that the vector embeddings provide than the magitude of it.
        metadata={"hnsw:space": "cosine"},
    )

# Here we are storing the embeddings into vector databse based on the splits done by the ->  'text spliter' in the process_document.
def add_to_vector_collection(all_splits: list[Document], file_name: str):
    
    # Then it is stored in this collection.
    collection = get_vector_collection()
    # Here we create some empty lists, we will fill in later.
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        # page_content -> refers to text from the pdf.
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        # ids are stored with file_name+ the iteration.This is done because we need unique ids.
        ids.append(f"{file_name}_{idx}")

    # Now we will add this data to the empty lists which we created.
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        #Bases on the 'ids' its gonna populate the Vector Database.
        ids=ids,
    )
    st.success("Data added to the vector store!")

# Queries the vector collection with a given user prompt to retrieve relevant chunks.
# The search query text is used to find relevant chunks, It can return maximum of 10 chunks.
def query_collection(prompt: str, n_results: int = 10):
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results

# This is used to call the Large Language Model with context and prompt to generate a response.
def call_llm(context: str, prompt: str):
    response = ollama.chat(
        # This is the foundation model which we are using.
        model="llama3.2:3b",
        stream=True,
        # Here we are creating 2 roles on being the 'system' the other one is the 'user'.
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}",
            },
        ],
    )
    # Here if the 'chunk["done"]' is -> true indicates the LLm model has generated the answer.If it is false it is still generating the answers.
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break


def re_rank_cross_encoders(prompt: str, documents: list[str]) -> tuple[str, list[int]]:
    if not documents:
        return "", []

    # This is going to be the text corpus.Here all the related text chunks/documents are combined into a single string.
    relevant_text = ""
    # Here we will be taking relevant id's based on the ranking.This intitally empty.
    relevant_text_ids = []
    
    # This the CrossEncoder this has the context window of 512 tokens.
    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Here we are going to rank all the chunks and we are going to take the top 3 results based on the ranking.
    ranks = encoder_model.rank(prompt, documents, top_k=3)
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        # Here we are appending related text id's to empty list which we created early.
        relevant_text_ids.append(rank["corpus_id"])

    return relevant_text, relevant_text_ids


if __name__ == "__main__":

    st.set_page_config(page_title="Llama Rag for Pdf Q&A")

    # Then inject custom CSS
    st.markdown(cyberpunk_css, unsafe_allow_html=True)

    # Document Upload Area
    # This is for Sidebar
    with st.sidebar:
        # This is the file upload section where we can drag and drop the files.
        # The type of files which we allow is only pdf, and allow only one file not multiple files.
        uploaded_file = st.file_uploader(
            "**Upload PDF files for QnA**", type=["pdf"], accept_multiple_files=False
        )

        # This process button will convert the pdf text into chunks and store it in Vector Database.
        process = st.button(
            "Process",
        )
        if uploaded_file and process:
            normalize_uploaded_file_name = uploaded_file.name.translate(
                # The fila name can have these symbols, which we will convert into -> '_'.
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            # Here we are going to split the document, and then we gonna create the embeddings and add it to the vector database.
            all_splits = process_document(uploaded_file)
            add_to_vector_collection(all_splits, normalize_uploaded_file_name)

    # Question and Answer Area
    st.header("Llama Rag for Pdf Q&A")
    # This section is for user prompt.
    prompt = st.text_area("**Ask a question related to your document:**")
    ask = st.button("Ask")

    if ask and prompt:
        results = query_collection(prompt)
        context = results.get("documents", [[]])[0]
        # This will give us relvant chunks/documents.
        relevant_text, relevant_text_ids = re_rank_cross_encoders(prompt, context)
        # After having the relevant chunks/documents we will be calling the LLM.
        response = call_llm(context=relevant_text, prompt=prompt)
        st.write_stream(response)

        with st.expander("See retrieved documents"):
            st.write(results)

        with st.expander("See most relevant document ids"):
            st.write(relevant_text_ids)
            st.write(relevant_text)