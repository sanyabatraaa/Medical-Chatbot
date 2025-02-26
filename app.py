from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
import google.generativeai as genai
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Load embeddings and create Pinecone retriever
embeddings = download_hugging_face_embeddings()
index_name = "llmapp"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def generate_gemini_response(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        
        # Extract text response safely
        if response and response.candidates:
            return response.candidates[0].content.parts[0].text
        else:
            return "I'm sorry, but I couldn't generate a response."
    
    except Exception as e:
        print("Gemini API Error:", str(e))
        return "There was an issue generating a response."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "")
    
    if not msg.strip():
        return "Please enter a valid query."

    print("User Input:", msg)

    retrieved_docs = retriever.get_relevant_documents(msg)
    
    if retrieved_docs:
        context = " ".join([doc.page_content for doc in retrieved_docs])
        full_prompt = f"Context: {context}\n\nUser Query: {msg}\n\nAnswer:"
    else:
        full_prompt = f"User Query: {msg}\n\nAnswer:"

    response = generate_gemini_response(full_prompt)

    print("Response:", response)
    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
