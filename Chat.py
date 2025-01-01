import os
import groq
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import logging
from dotenv import load_dotenv
import streamlit as st
import hashlib
import time
from groq import RateLimitError
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

st.write("""

# AlpacaBot
""")

client = groq.Groq(
    api_key=os.environ.get("GROQ_API_KEY")

)

model_name = 'all-mpnet-base-v2'
model = SentenceTransformer(model_name)

def extract_text_from_txt(txt_path):
    content_text = ""
    with open(txt_path, 'r', encoding='utf-8') as file:
        content_text = file.read()
    return content_text

def create_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def process_txts(_hash=None):
    txt_directory = './docs/'
    all_chunks = []
    chunk_to_doc = {}
    for filename in os.listdir(txt_directory):
        if filename.endswith('.txt'):
            txt_path = os.path.join(txt_directory, filename)
            text = extract_text_from_txt(txt_path)
            chunks = create_chunks(text)
            all_chunks.extend(chunks)
            for chunk in chunks:
                chunk_to_doc[chunk] = filename

    return all_chunks, chunk_to_doc

def create_faiss_index(all_chunks):
    embeddings = model.encode(all_chunks)
    dimension = embeddings.shape[1]
    num_chunks = len(all_chunks)

    if num_chunks < 100:
        logging.info("Using FlatL2 index due to small number of chunks")
        index = faiss.IndexFlatL2(dimension)
    else:
        logging.info("Using IVFFlat index")
        n_clusters = min(int(np.sqrt(num_chunks)), 100)  # Balancing clustering and search efficiency
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)
        index.train(embeddings.astype('float32'))

    index.add(embeddings.astype('float32'))
    return index

def retrieve_relevant_chunks(query, index, all_chunks,chunk_to_doc, top_k=10):
    query_vector = model.encode([query])[0]
    top_k = min(top_k, len(all_chunks))
    D, I = index.search(np.array([query_vector]).astype('float32'), top_k)
   
    relevant_chunks = [all_chunks[i] for i in I[0]]
    selected_dict = {key: value for i, (key, value) in enumerate(chunk_to_doc.items()) if i in I[0]}

    plot_query_and_neighbors(I, relevant_chunks, query_vector,selected_dict, D, top_k)
    return relevant_chunks

def plot_query_and_neighbors(I, relevant_chunks, query_vector, selected_dict, D, top_k):
    """
    Plots the query vector and its nearest neighbors in a 2D space using PCA and displays the plot in Streamlit.
    
    Parameters:
    - I: List of indices of the top-k nearest neighbors.
    - all_chunks: List of all text chunks (or documents).
    - query_vector: The vector representation of the query (768-dimensional).
    - top_k: Number of nearest neighbors to plot (default is 10).
    """
    # Get the top-k nearest vectors based on I
    # Flatten I to ensure it's 1D, and take the first row if necessary
    # Flatten I to ensure it's 1D, and take the first row if necessary

    # Encode the top-k nearest chunks into vectors (using the same model as for the query)

    top_k_vectors = model.encode(relevant_chunks)  # Encode the nearest neighbors into vectors

    # Combine the query vector with the top-k nearest vectors
    vectors = np.array([query_vector] + list(top_k_vectors))  # Combine query vector with top-k vectors

    # Perform PCA to reduce to 3D
    pca = PCA(n_components=3)  # Reduce to 3D
    reduced_vectors = pca.fit_transform(vectors)

    # Create a color gradient based on distances (D)
    colors = [f'rgba({255-int(d*255)}, {100+int(d*100)}, 255, 0.9)' for d in D.flatten()]  # Gradient blue to pink
    #colors = [f'rgba({int(d*255)}, {100+int((1-d)*100)}, {255-int(d*255)}, 0.9)' for d in D.flatten()]
    #colors = [f'rgba({int(d*255)}, {50+int((1-d)*205)}, {int((1-d)*255)}, 0.9)' for d in D.flatten()]
    # Create the plot
    fig = go.Figure()
    # Add the query vector as a special marker
    fig.add_trace(go.Scatter3d(
        x=[reduced_vectors[0, 0]],
        y=[reduced_vectors[0, 1]],
        z=[reduced_vectors[0, 2]],
        mode='markers+text',
        marker=dict(size=12, color='red', symbol='diamond'),
        name='Query',
        text=["Query Vector"],
        textposition="top center"
    ))

    # Add the nearest neighbors
    fig.add_trace(go.Scatter3d(
        x=reduced_vectors[1:, 0],
        y=reduced_vectors[1:, 1],
        z=reduced_vectors[1:, 2],
        mode='markers+text',
        marker=dict(size=10, color=colors, symbol='circle'),
        name='Nearest Neighbors',
        text = [f"{key[4:6]} {value[:5]}" for key, value in selected_dict.items()],  # Annotate with chunk names
        textposition="top center"
    ))

    # Update layout for a cleaner look
    fig.update_layout(
        title="Query Vector and Nearest Neighbors (3D)",
        scene=dict(
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            zaxis_title="Principal Component 3"
        ),
        legend=dict(
            x=0,
            y=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)



def generate_response(query: str, relevant_chunks: List[str], primary_model: str = "llama-3.1-8b-instant", fallback_model: str = "gemma2-9b-it", max_retries: int = 3):
    # I have used a language model to generate responses based on retrieved chunks.
    # This allows for more natural and contextually appropriate answers.
    context = "\n".join(relevant_chunks)
    prompt = f"""Based on the following context, please answer the question. If the answer is not fully contained in the context, provide the most relevant information available and indicate any uncertainty.

            Context:
            {context}

            Question: {query}

            Answer:"""

    # I have implemented a fallback mechanism and retry logic for robustness.
    # This ensures the system can handle API errors and rate limits gracefully.
    models = [primary_model, fallback_model]
    for model in models:
        for attempt in range(max_retries):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that answers questions based on the given context."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=model,
                    temperature=0.7,
                    max_tokens=1024,
                    top_p=1,
                    stream=False,
                    stop=None
                )

                response = chat_completion.choices[0].message.content.strip()
                usage_info = {
                    "prompt_tokens": chat_completion.usage.prompt_tokens,
                    "completion_tokens": chat_completion.usage.completion_tokens,
                    "total_tokens": chat_completion.usage.total_tokens,
                    "model_used": model
                }
                logging.info(f"Usage Info: {usage_info}")
                return response, usage_info, relevant_chunks

            except RateLimitError as e:
                if model == fallback_model and attempt == max_retries - 1:
                    logging.error(f"Rate limit exceeded for both models after {max_retries} attempts.")
                    raise e
                logging.warning(f"Rate limit exceeded for model {model}. Retrying in 5 seconds...")
                time.sleep(5)
            except Exception as e:
                logging.error(f"Error occurred with model {model}: {str(e)}")
                break  # Move to the next model if any other error occurs

    raise Exception("Failed to generate response with all available models.")

def rag_query(query: str, index, all_chunks, chunk_to_doc, top_k: int = 10) -> tuple:
    relevant_chunks = retrieve_relevant_chunks(query, index, all_chunks,chunk_to_doc, top_k)
    response, usage_info, used_chunks = generate_response(query, relevant_chunks)

    # I have tracked source documents for transparency and citation.
    source_docs = list(set([chunk_to_doc.get(chunk, "Unknown Source") for chunk in used_chunks]))

    return response, usage_info, source_docs

def main():
    st.write("Ask questions about Alpakas.")

    # I have processed PDFs and created the index at the start to ensure up-to-date information.
    all_chunks, chunk_to_doc = process_txts()
    index = create_faiss_index(all_chunks)

    # I have provided default questions to guide users and demonstrate system capabilities.
    default_questions = [
        "What do Alpacas eat?",
        "Where do Alpacas live?",
        "Other (Type your own question)"
    ]

    # I have used a dropdown for ease of use, but also allowed custom questions for flexibility.
    selected_question = st.selectbox("Choose a question or select 'Other' to type your own:", default_questions)

    if selected_question == "Other (Type your own question)":
        user_query = st.text_input("Enter your question:")
    elif selected_question != "Select a question":
        user_query = selected_question
    else:
        user_query = ""

    if user_query:
        pass

    # I have used a button to trigger the query process, giving users control over when to send a request.
    if st.button("Get Answer"):
        if user_query and user_query != "Select a question":
            with st.spinner("Generating answer..."):
                # I have rechecked for changes in PDFs to ensure we're using the latest data.
                response, usage_info, source_docs = rag_query(user_query, index, all_chunks, chunk_to_doc)
            st.subheader("Answer:")
            st.write(response)

            st.subheader("Source Documents:")
            for doc in source_docs:
                st.write(f"- {doc}")

            with st.expander("Usage Information"):
                st.json({
                    "Prompt Tokens": usage_info["prompt_tokens"],
                    "Completion Tokens": usage_info["completion_tokens"],
                    "Total Tokens": usage_info["total_tokens"],
                    "Model Used": usage_info["model_used"]
                })
        else:
            st.warning("Please select a question or enter your own.")

if __name__ == "__main__":
    main()