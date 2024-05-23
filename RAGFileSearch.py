import streamlit as st
import tempfile
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Useful to add documents to the chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Useful to load the URL into documents
from langchain_community.document_loaders import TextLoader

# Split the Web page into multiple chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create Embeddings
from langchain_openai import OpenAIEmbeddings

# Vector Database FAISS
from langchain_community.vectorstores.faiss import FAISS

# USeful to create the Retrieval part
from langchain.chains import create_retrieval_chain

# Streamlit UI
st.title("Document Search with RAG")
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

if not openai_api_key.startswith('sk-'):
   st.warning('Please enter your OpenAI API key!', icon='⚠')
if openai_api_key.startswith('sk-'):
    uploaded_files=st.file_uploader("Upload text files", type=["txt"], accept_multiple_files=True)
    user_question=st.text_input("Enter your question:")

    def get_docs_from_files(file_list):
        docs = []
        for uploaded_file in file_list:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
        # Load the document using TextLoader
            loader = TextLoader(tmp_file_path)
            docs.extend(loader.load())
        return docs
       
    def create_vector_store(docs):
        embedding = OpenAIEmbeddings(api_key=openai_api_key)
        vector_store = FAISS.from_documents(docs, embedding=embedding)
        return vector_store

    def create_chain(vector_store):
        model = ChatOpenAI(api_key=openai_api_key, model_name="gpt-3.5-turbo")
        prompt_template = ChatPromptTemplate(
            template="Given the following context and question, provide a relevant answer.\n\nContext: {context}\n\nQuestion: {question}",
            input_variables=["context", "question"]
        )
        document_chain = create_stuff_documents_chain(llm=model,prompt=prompt_template)
        retriever = vector_store.as_retriever(search_kwargs={"k": 1})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        return retrieval_chain

if st.button("Query Doc"):
    if uploaded_files and user_question:
        with st.spinner("Processing..."):
           try:
               docs = []
               if uploaded_files:
                   docs.extend(get_docs_from_files(uploaded_files))
                   text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
                   split_docs = text_splitter.split_documents(docs)
                   vector_store = create_vector_store(split_docs)
                   chain = create_chain(vector_store)
                   context = " ".join([doc.page_content for doc in split_docs])
                   inputs = {"context": context, "question": user_question}
                   st.write("### Debugging Info")
                   st.write(f"Context: {context}")
                   st.write(f"Question: {user_question}")
                   response = chain.invoke(inputs)   
                   st.write("### Full Response")
                   st.write(response)
                   if 'answer' in response:
                         st.write("### Answer")
                         st.write(response['answer'])
           except Exception as e:
                  st.error(f"An error occurred: {e}")
else:
    st.warning("Please enter your OpenAI API Key")
