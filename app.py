import streamlit as st
st.set_page_config(
    page_title="Personal AI Research Assistant",
    page_icon=":scroll:",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import shutil
import time
import requests
import docx
import pytesseract
import google.generativeai as genai
from PyPDF2 import PdfReader
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_bytes
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS
import base64

# ------------------ Environment Setup ------------------
load_dotenv()
if "TESSERACT_PATH" in os.environ:
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_PATH"]
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize conversation history in session_state if not present
if "question_answer_history" not in st.session_state:
    st.session_state.question_answer_history = []
# Initialize transcribed text storage
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""

# ------------------ Helper Functions for Speech ------------------

# Language mapping for speech recognition and TTS (add more if needed)
language_map = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Spanish": "es",
    "French": "fr"
}

def speech_to_text(language):
    """Captures speech input and returns transcribed text in the chosen language."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Speak now...")
        try:
            audio = r.listen(source, timeout=10)
            lang_code = language_map.get(language, "en-IN")
            # Attempt to use Google's free recognition service
            text = r.recognize_google(audio, language=lang_code)
            return text
        except sr.WaitTimeoutError:
            st.error("No speech detected. Please try again.")
            return ""
        except sr.RequestError as req_err:
            st.error(f"Network error: Could not request results from Google Speech Recognition service; {req_err}")
            return ""
        except Exception as e:
            st.error(f"Speech recognition error: {e}")
            return ""

def text_to_speech(text, language):
    """Converts text to audio and plays it in the browser."""
    try:
        lang_code = language_map.get(language, "en")
        tts = gTTS(text=text, lang=lang_code, slow=False)
        audio_file = BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        # Encode audio for HTML playback
        audio_base64 = base64.b64encode(audio_file.read()).decode()
        audio_html = f'''
            <audio controls autoplay style="width: 100%;">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
        '''
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Text-to-speech failed: {e}")

# ------------------ Helper Functions for Document & URL Processing ------------------

def extract_text_from_image(image):
    if shutil.which("tesseract") is None:
        st.error("Tesseract is not installed or not in your PATH. Please install Tesseract OCR and ensure it's accessible.")
        return ""
    try:
        ocr_text = pytesseract.image_to_string(image)
        return ocr_text
    except Exception as e:
        st.error(f"OCR failed: {e}")
        return ""

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_bytes = pdf.read()
            pdf.seek(0)
            pdf_reader = PdfReader(BytesIO(pdf_bytes))
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text += page_text
                else:
                    try:
                        images = convert_from_bytes(pdf_bytes, first_page=page_num+1, last_page=page_num+1)
                        for image in images:
                            ocr_text = extract_text_from_image(image)
                            if ocr_text:
                                st.info(f"OCR performed on PDF page {page_num+1}.")
                            text += ocr_text
                    except Exception as e:
                        st.error(f"OCR failed for PDF page {page_num+1}: {e}")
        except Exception as e:
            st.error(f"Failed to process a PDF: {e}")
    return text

def get_docx_text(docx_docs):
    text = ""
    for docx_file in docx_docs:
        try:
            document = docx.Document(docx_file)
            for para in document.paragraphs:
                text += para.text + "\n"
            for rel in document.part._rels:
                rel_part = document.part._rels[rel]
                if "image" in rel_part.target_ref:
                    try:
                        image_bytes = rel_part.target_part.blob
                        image = Image.open(BytesIO(image_bytes))
                        ocr_text = extract_text_from_image(image)
                        if ocr_text:
                            st.info("OCR performed on an image in a Word document.")
                        text += "\n" + ocr_text
                    except Exception as e:
                        st.error(f"OCR failed for an image in a Word document: {e}")
        except Exception as e:
            st.error(f"Failed to process a Word document: {e}")
    return text

def get_url_text_selenium(url):
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        time.sleep(5)
        html_content = driver.page_source
        driver.quit()
        soup = BeautifulSoup(html_content, "html.parser")
        for script in soup(["script", "style"]):
            script.extract()
        visible_text = soup.get_text(separator=" ", strip=True)
        return visible_text
    except Exception as e:
        st.error(f"Selenium failed for {url}: {e}")
        return ""

def get_url_text(urls):
    text = ""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:104.0) Gecko/20100101 Firefox/104.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Referer": "https://www.google.com/"
    }
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                st.error(f"Standard request failed for {url} with status code {response.status_code}. Trying Selenium fallback.")
                selenium_text = get_url_text_selenium(url)
                if selenium_text:
                    text += selenium_text
                else:
                    st.error(f"Failed to extract text from {url} using Selenium.")
                continue
            if url.lower().endswith(".pdf"):
                pdf_bytes = response.content
                try:
                    pdf_reader = PdfReader(BytesIO(pdf_bytes))
                    for page_num, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text += page_text
                        else:
                            try:
                                images = convert_from_bytes(pdf_bytes, first_page=page_num+1, last_page=page_num+1)
                                for image in images:
                                    ocr_text = extract_text_from_image(image)
                                    if ocr_text:
                                        st.info(f"OCR performed on URL PDF page {page_num+1}.")
                                    text += ocr_text
                            except Exception as e:
                                st.error(f"OCR failed for URL PDF page {page_num+1}: {e}")
                except Exception as e:
                    st.error(f"Failed to process PDF from {url}: {e}")
            elif url.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    image_bytes = response.content
                    image = Image.open(BytesIO(image_bytes))
                    ocr_text = extract_text_from_image(image)
                    if ocr_text:
                        st.info("OCR performed on image URL.")
                    text += ocr_text
                except Exception as e:
                    st.error(f"Failed to process image URL {url}: {e}")
            else:
                try:
                    html_content = response.text
                    soup = BeautifulSoup(html_content, "html.parser")
                    for script in soup(["script", "style"]):
                        script.extract()
                    visible_text = soup.get_text(separator=" ", strip=True)
                    text += visible_text
                except Exception as e:
                    st.error(f"Failed to process webpage URL {url}: {e}")
        except Exception as e:
            st.error(f"Failed to process URL {url}: {e}")
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
        except Exception as e:
            if attempt < retries - 1:
                time.sleep((2 ** attempt) + 1)
            else:
                st.error(f"Error during Google search: {e}")
                return "Could not find information on Google."
    return "Could not find information on Google."

def translate_text(text, target_language):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(f"Translate the following text to {target_language}:\n\n{text}")
    return response.text

def user_input(user_question, target_language):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local(os.path.join(os.getcwd(), "faiss_index"), embeddings,
                                  allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
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
    
    st.session_state.question_answer_history.append({
        "question": user_question,
        "answer": translated_answer
    })
    
    st.write("**Reply:**")
    st.write(translated_answer)

# ------------------ Main App with Enhanced UI ------------------

def main():
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

    st.markdown('<div class="header">📚 Personal AI Research Assistant 🤖</div>', unsafe_allow_html=True)

    # Main Section: Input Mode Selection, Language, and Question Input
    input_mode = st.radio("Input Mode:", ["Type", "Speak"], horizontal=True)
    
    if input_mode == "Speak":
        col1, col2 = st.columns([3, 1])
        with col1:
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
        with col2:
            if st.button("🎤 Record Question"):
                # Use the speech_recognition based function
                recorded_text = speech_to_text(language)
                st.session_state.transcribed_text = recorded_text
        # Display the transcribed text in an editable text box
        user_question = st.text_input("Recorded Question (editable)", value=st.session_state.get("transcribed_text", ""))
    else:
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
        user_question = st.text_input("Ask a Question from the uploaded documents .. ✍️📝")
    
    audio_output = st.checkbox("🔊 Enable Audio Output")
    
    if st.button("Submit Question"):
        if user_question:
            user_input(user_question, language)
            if audio_output and st.session_state.question_answer_history:
                latest_answer = st.session_state.question_answer_history[-1]["answer"]
                text_to_speech(latest_answer, language)
        else:
            st.error("Please enter or record a question before submitting.")
    
    st.markdown("---")
    
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
    
    with st.sidebar:
        st.image("img/Robot.jpg")
        st.markdown("---")
        st.markdown('<div class="sidebar-section"><h3>📁 Documents Section</h3></div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Upload your PDF or Word documents", accept_multiple_files=True, type=["pdf", "docx"])
        if st.button("Submit & Process Documents"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    raw_text = ""
                    pdf_files = [file for file in uploaded_files if file.type == "application/pdf"]
                    docx_files = [file for file in uploaded_files if file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
                    if pdf_files:
                        raw_text += get_pdf_text(pdf_files)
                    if docx_files:
                        raw_text += get_docx_text(docx_files)
                    if raw_text.strip():
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Documents processed successfully!")
                    else:
                        st.error("No text could be extracted from the uploaded documents.")
            else:
                st.error("Please upload at least one document.")
    
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
                    raw_text = get_url_text(urls)
                    if raw_text.strip():
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("URLs processed successfully!")
                    else:
                        st.error("No text could be extracted from the provided URLs.")
            else:
                st.error("Please enter at least one valid URL.")
    
        st.markdown("---")
        st.image("img/gkj.jpg")
        st.markdown('<a href="https://vnrvjiet.ac.in/" target="_blank">CREATED BY @ VNRJIET STUDENTS</a>', unsafe_allow_html=True)
    
    with st.sidebar:
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
