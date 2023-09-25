
import os 
import openai
from flask import Flask, request, jsonify
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import AzureChatOpenAI 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.text_splitter import TokenTextSplitter 
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.chains import RetrievalQA

from langchain.llms import AzureOpenAI

from flask_cors import CORS, cross_origin

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from azure.storage.blob import BlobServiceClient
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma



app = Flask(__name__)

CORS(app, origins=["http://localhost:4200"])

load_dotenv(find_dotenv())


# Azure Storage configuration
AZURE_STORAGE_CONNECTION_STRING = ''
CONTAINER_NAME = 'knowledgebase/knowledgeBase'


openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPENAI_API_VERSION")

embeddings = OpenAIEmbeddings(deployment_id="text-embedding-ada-002", chunk_size=1, openai_api_key=os.getenv('OPENAI_API_KEY'), openai_api_base=os.getenv("OPENAI_API_BASE"))

llm = AzureChatOpenAI(deployment_name="gpt-3.5-turbo")

def load_documents_from_pdfs(pdf_files):
    documents = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join("uploads", pdf_file.filename)
        pdf_file.save(pdf_path)

        text = ""
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfFileReader(pdf_file)
            for page_num in range(pdf_reader.getNumPages()):
                page = pdf_reader.getPage(page_num)
                text += page.extractText()
        documents.append(text)
    
    return documents

# Endpoint for processing user-uploaded PDFs
@app.route('/process_pdfs', methods=['POST'])
def process_pdfs():
    try:
        # Get uploaded PDF files
        pdf_files = request.files.getlist('pdf_files')

        if not pdf_files:
            return jsonify({"error": "No PDF files were uploaded"}), 400

        # Create a directory for storing uploaded files
        os.makedirs("uploads", exist_ok=True)

        # Load documents from uploaded PDFs
        docs = load_documents_from_pdfs(pdf_files)

        # Split documents into chunks for embedding generation
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunked_docs = []
        for doc in docs:
            chunked_docs.extend(text_splitter.split_documents([doc]))

        # Generate and store embeddings in "vectorEmbedding.txt"
        with open("vectorEmbedding.txt", "w", encoding="utf-8") as f:
            for doc in chunked_docs:
                embeddings_list = embeddings.generate_embeddings(doc)
                # Write the embeddings to the file
                for embedding in embeddings_list:
                    embedding_str = " ".join(map(str, embedding))
                    f.write(embedding_str + "\n")

        return jsonify({"message": "PDFs processed successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/pdf_loader', methods=['POST'])
def pdf_loader():

   
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Invalid file format. Only PDF files are supported.'}), 400

    tmp_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
    os.makedirs(tmp_folder, exist_ok=True)
    pdf_file_path = os.path.join(tmp_folder, file.filename)
    file.save(pdf_file_path)

    loader = PyPDFLoader(pdf_file_path)
    pages = loader.load()

    content = []
    for i in range(len(pages)):
        content.append(pages[i].page_content)

    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=0,
        separators=["\n\n", "\n", "\. ", " ", ""],
        
    )
    j = r_splitter.split_text(content[0])

    def write_embeddings_to_file(embeddings, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            for doc in embeddings:
                f.write(str(doc) + "\n")

    # Store embeddings in a list
    embeddings_list = []
        # Generate embeddings for each chunk and store them in the list
    for doc in j:
        embeddings = openai.Embedding.create(engine="mbeddings", input=doc)
        embeddings_list.append(embeddings)

    # Write the embeddings list to a file
    write_embeddings_to_file(embeddings_list, "vectorEmbedding.txt")

    g = "done"


    return j



def query_to_embedding(query):
    response = openai.Embedding.create(engine="", input=query)
    return np.fromstring(response['embedding'], sep=',')

# Function to search for a query in the embeddings
def search_query(query, embeddings, top_n=5):
    query_embedding = query_to_embedding(query)
    similarities = cosine_similarity([query_embedding], embeddings)
    similar_indices = np.argsort(similarities[0])[::-1]
    
    # Return the top_n most similar embeddings
    top_similarities = [(similar_indices[i], similarities[0][similar_indices[i]]) for i in range(top_n)]
    return top_similarities


@app.route('/query', methods=['POST'])
def query_embeddings():
    try:
        # Get the query from the request data
        query = request.json.get('query', '')

        print("kinbdibdbod", query)

        query_to_embedding = openai.Embedding.create(engine="", input=query)
        print("query_to_embedding", query_to_embedding)
        
        # Search for the query in the embeddings
        top_similarities = search_query(query, embeddings)
        
        # Prepare the response
        response_data = {
            "query": query,
            "results": []
        }
        
        # Add the top similar embeddings to the response
        for i, (index, similarity) in enumerate(top_similarities, start=1):
            result = {
                "index": index,
                "similarity": similarity,
                "embedding": embeddings_lines[index].strip()
            }
            response_data["results"].append(result)
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)})



@app.route('/upload', methods=['POST'])
@cross_origin()
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Upload the file to Azure Storage
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        blob_client = container_client.get_blob_client(file.filename)
        if blob_client.exists():
            return jsonify({'error': 'The specified blob already exists'})
        
        blob_client.upload_blob(file)
        
        return jsonify({'message': 'File uploaded successfully to Azure Storage'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    


@app.route('/get_files', methods=['GET'])
@cross_origin()
def get_files():
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        blob_list = blob_service_client.get_container_client("knowledgebase").list_blobs(name_starts_with="knowledgeBase/")

        files = []

        for blob in blob_list:
            file_size_bytes = blob.size
            if file_size_bytes >= 1024 * 1024:
                file_size = f"{file_size_bytes / (1024 * 1024):.2f} MB"
            else:
                file_size = f"{file_size_bytes / 1024:.2f} KB"

            files.append({
                'name': blob.name.split('/')[-1],
                'size': file_size,
                'last_modified': blob.last_modified,
            })

        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



def save_pdf_to_temp_folder(file):
    temp_folder = './temp/'
    os.makedirs(temp_folder, exist_ok=True)
    pdf_path = os.path.join(temp_folder, file.filename)
    file.save(pdf_path)
    return pdf_path

def upload_pdf_to_azure_blob_storage(pdf_path, filename):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        blob_client = container_client.get_blob_client(filename)
        blob_client.upload_blob(open(pdf_path, "rb"))
        return True
    except Exception as e:
        return str(e)

def create_and_persist_vectordb(docs, deployment="", persist_directory="./data/chroma/"):
    response = OpenAIEmbeddings(deployment=deployment)
    vectordb = Chroma.from_documents(documents=docs, embedding=response, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb



def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=0,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(documents)
    return docs

@app.route('/process_pdf_with_embeddings_upload', methods=['POST'])
def process_pdf_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename.endswith('.pdf'):
            pdf_path = save_pdf_to_temp_folder(file)


            docs = load_and_split_pdf(pdf_path)
            

            create_and_persist_vectordb(docs, deployment="", persist_directory="./data/chroma/")

            if pdf_path:
                if upload_pdf_to_azure_blob_storage(pdf_path, file.filename):
                    os.remove(pdf_path)
                    return jsonify({'message': 'PDF processed successfully and uploaded to Azure Blob Storage!'})
                else:
                    return jsonify({'error': 'Failed to upload PDF to Azure Blob Storage'}), 500
            else:
                return jsonify({'error': 'Failed to save PDF to temporary folder'}), 500
        else:
            return jsonify({'error': 'Invalid file format. Please upload a PDF file.'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route("/search", methods=["POST"])
def get_similarity():
    question = request.json.get("question")
    persist_directory = './data/chroma/'
    response = OpenAIEmbeddings(deployment="bdcembeddings")
    vb1 = Chroma(persist_directory=persist_directory, embedding_function=response)
    print(question)
    result = vb1.similarity_search(question, k=1)
    return jsonify({"result": result[0].page_content})


@app.route("/postsimilarity", methods=["POST"])
def similarity():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Invalid input"}), 400

    persist_directory = './data/chroma/'
    response = OpenAIEmbeddings(deployment="bdcembeddings")
    vb1 = Chroma(persist_directory=persist_directory, embedding_function=response)
    result = vb1.similarity_search(question, k=1)

    if not result:
        return jsonify({"error": "No results found"}), 404

    return jsonify({"result": result[0].page_content})

def similaritySearch(question):

    persist_directory = './data/chroma/'
    response = OpenAIEmbeddings(deployment="ghh")
    vb1 = Chroma(persist_directory=persist_directory, embedding_function=response)
    result = vb1.similarity_search(question, k=1)

    if not result:
        return jsonify({"error": "No results found"}), 404

    return  result[0].page_content
    # return jsonify({"result": result[0].page_content})

@app.route("/postsimilarityWithGPT", methods=["POST"])
def get_semantic_answer():
    # question += "\n"

    data = request.get_json()
    question = data.get("question")
    print(question)
    res = similaritySearch(question)

    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
    {res}
    Question: {question}
    Helpful Answer:"""

    print(res)


    prompt = "Give answer only using the paragraphs provided and be very particular to the user question or else reply I dont know"
    prompt = f"{res}\n\n{prompt}"

    print(template)


    response_data = {"summary": template}

    return jsonify(response_data)


@app.route('/get_selected_files', methods=['POST'])
def get_selected_files():
    try:
        data = request.json 
        selected_files = data.get('selectedFiles', [])  

        print(selected_files)
        return jsonify(selected_files), 200

    except Exception as e:
        error_response = {
            'error': 'An error occurred while processing the request',
            'details': str(e)
        }
        return jsonify(error_response), 500
    
@app.route('/delete-blobs', methods=['POST'])
def delete_blobs():
    try:

        data = request.json
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400

        selected_files = data.get('selectedFiles', [])

        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)

        deleted_blobs = []

        for file_name in selected_files:
            blob_client = container_client.get_blob_client(file_name)
            
            # Check if the blob exists before deleting it
            if blob_client.exists():
                blob_client.delete_blob()
                deleted_blobs.append(file_name)
        
        return jsonify({'message': 'Blobs deleted successfully', 'deleted_blobs': deleted_blobs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/query_vector', methods=['POST'])
@cross_origin()
def query_vector():

    question = request.json.get('user_input')
    print(question)
    persist_directory = './data/chroma/'
    if (question.lower()== "hi" or question.lower()=="hello"):
        return jsonify({'summary': "Hello, how may I help you!!"}), 200
    # embedding = OpenAIEmbeddings(model="text-embedding-ada-002",openai_api_key='e3f00b3e25644e28b2bd9928b2559662',engine="bdcembeddings")
    embedding = OpenAIEmbeddings(deployment="bdcembeddings")
    db123 = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    print(db123.get())
    qa = RetrievalQA.from_chain_type(
    llm=AzureOpenAI(temperature=0, engine="", max_tokens=300,openai_api_key = "",openai_api_base= "",openai_api_version="2023-03-15-preview"),
    chain_type="stuff",
    retriever=db123.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    verbose=False,)
    res=qa(question)
    print(res["result"])
    ans=res["result"]
    loc=ans.find("Question:")
    if(ans.find("I don't know.") != -1):
        return jsonify({'summary': "I don't know. Please add related pdf or docx file in knowledge base"}), 200
    elif(loc != -1):
        loc=ans.find("Question:")
        answer=ans[:loc]
        return jsonify({'summary': answer}), 200
    return jsonify({'summary': res["result"]}), 200



def summarize_product_review(content):

    #1

    # prompt = f"""
    # Your task is to generate a short summary in 30 words only Don't exceed word limits. 

    # Give in bullet points.
    # for example:
    # 1. 
    # 2.

    # Summarize the review below, delimited by triple 
    # backticks, in only 30 words. 

    # Review: ```{review}```
    # """

    #2

    # prompt = f"""
    # You are a highly advanced language model tasked with generating a concise and informative summary of the given content.

    # Content: {content}

    # Please provide a summary in at most 50 words.
    # """

    prompt = f"""
    Your task is to provide a short and informative summary in at most 30 words. Use bullet points to list the key points.

    Summary:
    - 
    - 

    Content: {content}
    """



    response = openai.ChatCompletion.create(
    engine="bdcgpt35turbo",
    messages = [{"role": "user", "content": prompt}],
    temperature=0,

    )

    
    return response.choices[0].message["content"]

@app.route('/pdf_summariser', methods=['POST'])
@cross_origin()
def pdf_summariser():

    openai.api_key = os.getenv('OPENAI_API_KEY')
    openai.api_type = os.getenv("OPENAI_API_TYPE")
    openai.api_base = os.getenv("OPENAI_API_BASE")
    openai.api_version = os.getenv("OPENAI_API_VERSION")

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Invalid file format. Only PDF files are supported.'}), 400

    tmp_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
    os.makedirs(tmp_folder, exist_ok=True)
    pdf_file_path = os.path.join(tmp_folder, file.filename)
    file.save(pdf_file_path)
    loader = PyPDFLoader(pdf_file_path)
    pages = loader.load()
    # pages[0].page_content

    print(len(pages))
    content = []
    for i in range(len(pages)):
        content.append(pages[i].page_content)

    summary = summarize_product_review(content[0])
    print(summary)
    response_data = {"summary": summary}

    return jsonify(response_data)
    # return jsonify({'pages': summary})


if __name__ == '__main__':
    app.run(debug=True)
    