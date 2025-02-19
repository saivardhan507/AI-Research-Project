import streamlit as st
st.set_page_config(
    page_title="Personal AI Research Assistant",
    page_icon=":scroll:",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

# Load environment variables and configure API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize conversation history in session_state if not present
if "question_answer_history" not in st.session_state:
    st.session_state.question_answer_history = []

# ------------------ Core Processing Functions ------------------

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
    You are provided with the following context and a question. Please craft a thorough, well-organized answer that covers every relevant detail. Your response should be at least 300 words long and include tables, bullet points, diagrams, or other structured formats as needed to enhance clarity. If the provided context does not cover some aspects of the question, please supplement your answer with credible information fetched from Google resources. 

Your answer must:
- Explain the topic comprehensively, covering all pertinent aspects.
- Include tables or structured data to summarize key information, where applicable.
- Use clear headings, subheadings, bullet points, or numbered lists to organize your explanation.
- Elaborate with in-depth details, examples, and analysis.
- Ensure that the explanation is logically structured and easy to follow.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def search_google(query, retries=5):
    model = genai.GenerativeModel('gemini-pro')
    for attempt in range(retries):
        try:
            response = model.generate_content(f"Give me a summary of : {query}")
            return response.text
        except google.api_core.exceptions.ResourceExhausted:
            if attempt < retries - 1:
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
    
    if ("cannot be answered from the given context" in response["output_text"].lower() or 
        "not found in the provided context" in response["output_text"].lower()):
        google_answer = search_google(user_question)
        response["output_text"] = (
            "This question cannot be answered from the given context because the provided context does not mention anything about it. "
            "But here is an answer sourced from Google:\n\n" + google_answer
        )

    translated_answer = translate_text(response["output_text"], target_language)
    
    # Save the conversation in session state
    st.session_state.question_answer_history.append({
        "question": user_question,
        "answer": translated_answer
    })
    
    st.write("**Reply:**")
    st.write(translated_answer)

# ------------------ Main App with Enhanced UI ------------------

def main():
    # Inject custom CSS for modern visuals and conversation history styling with updated colors
    st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #ece9e6, #ffffff);
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 1em;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .header {
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar-section {
        margin-bottom: 2rem;
    }
    .footer {
        text-align: center;
        font-size: 0.8em;
        padding: 15px;
        background-color: #0E1117;
        color: white;
        position: fixed;
        bottom: 0;
        width: 100%;
    }
    /* Updated conversation container styling for better visibility */
    .conversation-container {
        background: #2c3e50;
        color: #ecf0f1;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .conversation-box {
        background-color: #34495e;
        border-left: 4px solid #e74c3c;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="header">📚 Personal AI Research Assistant 🤖</div>', unsafe_allow_html=True)

    # Main Section: Language selection and question input
    language = st.selectbox("Select Language", (
        "English", "Spanish", "French", "German", "Chinese", "Hindi", "Telugu", "Tamil", "Gujarati", "Marathi", 
        "Punjabi", "Kannada", "Malayalam", "Odia", "Assamese", "Urdu", "Konkani", "Maithili", "Santali", "Sindhi", 
        "Nepali", "Bodo", "Dogri", "Kashmiri", "Manipuri", "Sanskrit", "Japanese", "Korean", "Italian", "Portuguese", 
        "Russian", "Arabic", "Dutch", "Greek", "Swedish", "Turkish", "Vietnamese", "Polish", "Bengali", "Afrikaans", 
        "Albanian", "Armenian", "Azerbaijani", "Basque", "Belarusian", "Bosnian", "Bulgarian", "Catalan", "Croatian", 
        "Czech", "Danish", "Estonian", "Finnish", "Georgian", "Hebrew", "Hungarian", "Icelandic", "Indonesian", "Irish", 
        "Latvian", "Lithuanian", "Macedonian", "Malagasy", "Maltese", "Mongolian", "Montenegrin", "Norwegian", 
        "Pashto", "Persian", "Romanian", "Serbian", "Slovak", "Slovenian", "Somali", "Swahili", "Tajik", "Tatar", 
        "Thai", "Tibetan", "Turkmen", "Ukrainian", "Uzbek", "Welsh", "Yoruba", "Zulu", "Amharic", "Chechen", 
        "Chichewa", "Esperanto", "Fijian", "Fula", "Galician", "Hausa", "Hmong", "Igbo", "Inuktitut", "Javanese", 
        "Kinyarwanda", "Kirundi", "Kurdish", "Lao", "Luganda", "Luxembourgish", "Maldivian", "Marshallese", "Nauru", 
        "Navajo", "Oriya", "Palauan", "Quechua", "Samoan", "Sango", "Serer", "Shona", "Sotho", "Tagalog", "Tahitian", 
        "Tigrinya", "Tonga", "Tswana", "Tuvaluan", "Wallisian", "Xhosa", "Akan", "Bambara", "Bashkir", "Bislama", 
        "Chuvash", "Divehi", "Dzongkha", "Ewe", "Faroese", "Gaelic", "Greenlandic", "Haitian", "Herero", "Kashubian", 
        "Kikuyu", "Lingala", "Lozi", "Makonde", "Mandingo", "Ndonga", "Nuosu", "Nyanja", "Nyamwezi", "Oromo", 
        "Rohingya", "Saraiki", "Shan", "Silesian", "Sinhala", "Sorani", "Tsonga", "Wolof", "Zaza"
    ))

    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝")
    if st.button("Submit Question"):
        if user_question:
            user_input(user_question, language)
            st.write("---")

    # Display conversation history in a creative modern box with updated colors
    if st.session_state.question_answer_history:
        
        st.markdown('<h2 style="text-align: center; margin-bottom: 20px;">Conversation History📜</h2>', unsafe_allow_html=True)
        for pair in st.session_state.question_answer_history:
            conversation_html = f'''
            <div class="conversation-box">
              <p><strong>Question:</strong> {pair['question']}</p>
              <p><strong>Answer:</strong> {pair['answer']}</p>
            </div>
            '''
            st.markdown(conversation_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ------------------ Sidebar Layout ------------------
    with st.sidebar:
        st.image("img/Robot.jpg")
        st.markdown("---")
        st.markdown('<div class="sidebar-section"><h3>📁 PDF Files Section</h3></div>', unsafe_allow_html=True)
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
        if st.button("Submit & Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDFs processed successfully!")
            else:
                st.error("Please upload at least one PDF file.")

        st.markdown("---")
        st.markdown('<div class="sidebar-section"><h3>🌐 URL Files Section</h3></div>', unsafe_allow_html=True)
        num_urls = st.number_input("How many URLs do you want to enter?", min_value=1, value=1, step=1)
        url_list = []
        for i in range(int(num_urls)):
            url = st.text_input(f"Enter URL {i+1}:")
            url_list.append(url)
        if st.button("Submit & Process URLs"):
            urls = [url.strip() for url in url_list if url.strip()]
            if urls:
                with st.spinner("Processing URLs..."):
                    raw_text = get_url_pdf_text(urls)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("URLs processed successfully!")
            else:
                st.error("Please enter at least one valid URL.")

        st.markdown("---")
        st.image("img/gkj.jpg")
        st.markdown('<a href="https://vnrvjiet.ac.in/" target="_blank">CREATED BY @ VNRJIET STUDENTS</a>', unsafe_allow_html=True)

    # ------------------ Footer ------------------
        st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
             © <a href="https://wa.me/qr/FIEHCYZKAOBED1" target="_blank">For more info Connect Here </a> | Project Guide: Dr.V.Radhika
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
