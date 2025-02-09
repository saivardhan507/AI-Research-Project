import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
from io import BytesIO
import time

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

question_answer_history = []

# Added function to translate text
def translate_text(text, target_language):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(f"Translate the following text to {target_language}:\n\n{text}")
    return response.text

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_url_pdf_text(urls):
    text = ""
    for url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            pdf_reader = PdfReader(BytesIO(response.content))
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, provide the correct answer from the google resources\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def search_google(query, retries=5):
    """Searches Google and returns a simplified summary of the top result."""
    model = genai.GenerativeModel('gemini-pro')
    for attempt in range(retries):
        try:
            response = model.generate_content(f"Give me a summary of : {query}")
            return response.text
        except google.api_core.exceptions.ResourceExhausted:
            if attempt < retries - 1:
                # Exponential backoff
                time.sleep((2 ** attempt) + 1)
            else:
                raise
        except Exception as e:
            print(f"Error during Google search: {e}")
            return "Could not find information on Google."

    return "Could not find information on Google."

def user_input(user_question, target_language):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    try:
        new_db = FAISS.load_local(os.path.join(os.getcwd(), "faiss_index"), embeddings,
                                  allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    if "cannot be answered from the given context" in response["output_text"].lower() or "not found in the provided context" in response["output_text"].lower():
        google_answer = search_google(user_question)
        response["output_text"] = f"This question cannot be answered from the given context because the provided context does not mention anything about it. But I can provide you with an answer from other sources:\n\n{google_answer}"

    translated_answer = translate_text(response["output_text"], target_language)
    question_answer_history.append({"question": user_question, "answer": translated_answer})
    print(translated_answer)
    # st.write("Reply:\n ", translated_answer)

def main():
    st.set_page_config("VNRVJIET AI Research Assistant", page_icon=":scroll:")
    st.header("📚 Personal AI Research Assistant 🤖 ")

    # Language selection
    language = st.selectbox("Select Language", (
        "English", 
        "Spanish", 
        "French", 
        "German", 
        "Chinese", 
        "Hindi" ,
        "Telugu"
    ))

    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝")

    if user_question:
        user_input(user_question, language)

        for i, pair in enumerate(question_answer_history, start=1):
            st.write(f"Question: {pair['question']}")
            st.write(f"Answer: {pair['answer']}")
            st.write("---")

    with st.sidebar:    
        st.image("img/Robot.jpg")
        st.write("---")
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ",
                                    accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):  # user friendly message.
                raw_text = get_pdf_text(pdf_docs)  # get the pdf text
                text_chunks = get_text_chunks(raw_text)  # get the text chunks
                get_vector_store(text_chunks)  # create vector store
                st.success("Done")

        st.write("---")
        
        st.title("🌐 URL File's Section")
        urls_input = st.text_area("Enter URLs (separated by commas)", "")
        urls = [url.strip() for url in urls_input.split(",") if url.strip()]

        if st.button("Submit & Process URLs"):
            with st.spinner("Processing..."):  # user friendly message.
                raw_text = get_url_pdf_text(urls)  # get the pdf text from URLs
                text_chunks = get_text_chunks(raw_text)  # get the text chunks
                get_vector_store(text_chunks)  # create vector store
                st.success("Done")

        st.write("---")
        st.image("img/gkj.jpg")
        st.write("CREATED BY @ VNRJIET STUDENTS")

    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
             © <a href="https://wa.me/qr/FIEHCYZKAOBED1" target="_blank">For more info Connect Here </a> | Mentor: Dr.V.Radhika
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
