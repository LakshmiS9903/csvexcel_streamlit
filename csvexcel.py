import streamlit as st
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key not found. Please set it in the .env file.")
genai.configure(api_key=api_key)

# Function to extract text from CSV/Excel files
def get_file_text(file_docs):
    text = ""
    for file in file_docs:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file)
            else:
                st.warning(f"Unsupported file type: {file.name}")
                continue

            # Convert DataFrame to string and append to text
            text += df.to_string(index=False) + "\n"
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")
    return text

# Function to split the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate and save the vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # Ensure directory exists before saving
    os.makedirs("faiss_index", exist_ok=True)
    vector_store.save_local("faiss_index")

# Function to create the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say "answer is not available in the context"; do not provide the wrong answer.\n\n
    Context:\n {context}\n
    Question:\n {question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Use load_qa_chain from langchain
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except FileNotFoundError:
        st.error("FAISS index file not found. Please process files first.")
        return

    docs = new_db.similarity_search(user_question)
    if not docs:
        st.warning("No relevant information found for your question.")
        return

    # Create the conversational chain
    chain = get_conversational_chain()

    # Generate a response using the chain
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response.get("output_text", "No response generated."))

# Main function to set up Streamlit app
def main():
    st.set_page_config(page_title="Chat with Excel/CSV")
    st.header("Chat with Excel/CSV Files using Gemini 💁")

    user_question = st.text_input("Ask a Question from the Excel/CSV Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        file_docs = st.file_uploader("Upload your Excel or CSV Files and click 'Submit & Process'", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if file_docs:
                with st.spinner("Processing..."):
                    raw_text = get_file_text(file_docs)
                    if raw_text.strip():  # Ensure there is content to process
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Processing completed successfully!")
                    else:
                        st.warning("No valid content found in the uploaded files.")
            else:
                st.warning("Please upload at least one file.")

if __name__ == "__main__":
    main()
