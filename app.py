import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAI, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_file, text_split, downlaod_hugging_face_embeddings
from src.prompt import *
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import pinecone

# Initialize Streamlit page settings
st.set_page_config(layout="wide")
load_dotenv()

# API keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize embeddings and Pinecone index
@st.cache_resource
def get_embeddings():
    return downlaod_hugging_face_embeddings()

@st.cache_resource
def load_docsearch():
    embeddings = get_embeddings()
    index_name = "fachatbotwithmetadata"
    return PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

# Initialize the retrieval and RAG chain
def load_rag():
    docsearch = load_docsearch()
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    llm = ChatOpenAI(model="gpt-4o",api_key=OPENAI_API_KEY, temperature=0, max_tokens=500)
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_prompt),
            ('human', '{input}')
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

# Load RAG model
rag_chain = load_rag()

# Streamlit user interface
#st.title("Ataxia Chatbot")
logo_path = "FARA_LOGO.png"  # Ensure this image file is in your project directory

header_col1, header_col2 = st.columns([4, 5])
with header_col1:
    st.title("Ataxia Question Assistant Bot")
      # Adjust width as needed
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age = st.text_input("Age")
    with col2:
        aoo = st.text_input("AOO")
    with col3:
        aims = st.text_input("AIMS")
    with col4:
        mfars = st.text_input("mFARS")

# with header_col2:
#     st.image(logo_path, width=250)

#input_question = st.text_input("Ask your question")

# Check if any additional information fields have been provided
    additional_info = []
    if age:
        additional_info.append(f"age = {age}")
    if aoo:
        additional_info.append(f"aoo = {aoo}")
    if aims:
        additional_info.append(f"aims = {aims}")
    if mfars:
        additional_info.append(f"mfars = {mfars}")

    # Append additional information to the question if any field is filled
    # if additional_info and input_question.strip() != "":
    #     input_question += " , given by " + " , ".join(additional_info)

    additional_info_text = " , given by " + " , ".join(additional_info) if additional_info else ""

    # Main question input field
    raw_question = st.text_input("Ask your question")

    # Combine the question with additional info text
    input_question = raw_question + additional_info_text

    # Display the combined question
    if additional_info:
        st.write("### Final Question")
        st.write(input_question)

    #input_question = st.text_input("Ask your question")
    if input_question.strip() != "":
        with st.spinner("Generating Answer..."):
            prediction = rag_chain.invoke({"input": input_question})
        
        answer = prediction["answer"]
        source_documents = prediction["context"]

        
        st.write("### Answer")
        st.write(answer)

    # Create two columns for the answer and source documents
    #left_col, right_col = st.columns(2)

    # Display answer on the left
    

    # Display source documents on the right
        with header_col2:
            st.write("### Source Documents")
            for document in source_documents:
                content = document.page_content
                Title = document.metadata.get('title', 'Unknown Page')
                source = os.path.basename(document.metadata.get('source', 'Unknown Source'))

                with st.container(border=True):
                    st.markdown(f"**Content:** {content}")
                    st.markdown(f"**Title:** {Title}")
                    st.write(f"**Paper:** {source}")


