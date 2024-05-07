
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import LlamaCpp
from googletrans import Translator
from langdetect import detect
from bert_score import score

translator = Translator()
def translate_to_english(text):
    translation = translator.translate(text, src='vi', dest='en')
    return translation.text

def translate_to_vietnamese(text):
    translation = translator.translate(text, src='en', dest='vi')
    return translation.text

def translate_to_language(src, dest, text):
    translation = translator.translate(text, src=src, dest=dest)
    return translation.text
def detect_language(text):
    return detect(text)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = LlamaCpp(
        model_path="models/llama-2-7b-chat.Q4_K_M.gguf", n_ctx=1024, n_batch=512)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
        memory=memory
    )
    return conversation_chain


def evaluate_chatbot_performance(chatbot_answers, ground_truth_answers):
    # Initialize variables for TP, FP, FN
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for chatbot_answer, ground_truth_answer in zip(chatbot_answers, ground_truth_answers):
        # Compare chatbot answer with ground truth answer
        if chatbot_answer == ground_truth_answer:
            true_positives += 1
        else:
            false_positives += 1
            false_negatives += 1
    # Calculate Precision, Recall, and F1-score
    if true_positives + false_positives == 0 or true_positives + false_negatives == 0:
        # Handle case when precision or recall is 0 to avoid ZeroDivisionError
        precision = 0
        recall = 0
        f1_score = 0
    else:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall,f1_score

def calculate_bertscore(candidate, references):
    # Specify language and model type
    lang = 'en'
    model_type = 'bert-base-uncased'
    # Calculate BERTScore
    precision, recall, f1 = score([candidate], [references], lang=lang, model_type=model_type)
    return precision.item(), recall.item(), f1.item()

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    last_answer = st.session_state.chat_history[-1]
    question_language = detect_language(user_question)
    last_answer_language = detect_language(last_answer.content)
    print(last_answer.content)
    reference_translations = ["Popkomm was held in Cologne before moving to Berlin."]

    grade= evaluate_chatbot_performance(last_answer.content,"Popkomm was held in Cologne before moving to Berlin.")
    print(grade)

    bert_precision, bert_recall, bert_f1 = calculate_bertscore(last_answer.content, reference_translations)
    print("BERT Precision:", bert_precision)
    print("BERT Recall:", bert_recall)
    print("BERT F1 Score:", bert_f1)


    if question_language != last_answer_language:
        last_answer.content = translate_to_language(last_answer_language, question_language,
                                                    last_answer.content)
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
            "{{MSG}}", message.content), unsafe_allow_html=True)
def main():
    load_dotenv()
    st.set_page_config(page_title="Question-Answering Chatbot with multiple PDFs",page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    st.header("Chat with PDF :books:")
    user_question = st.text_input("Ask a questions about your documents:")
    if user_question:
        handle_userinput(user_question)
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDF files here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)  # get pdf text
                text_chunks = get_text_chunks(raw_text) # get the text chunks
                vectorstore = get_vectorstore(text_chunks) # create vector store
                st.session_state.vectorstore = vectorstore  # Save vectorstore for later use
                st.success("Document processed successfully!")
                if st.session_state.vectorstore is None:
                    st.warning("Please upload and process document first.")
                else:
                    if st.session_state.conversation is None:
                        # create conversation chain
                        st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)

if __name__ == '__main__':
    main()