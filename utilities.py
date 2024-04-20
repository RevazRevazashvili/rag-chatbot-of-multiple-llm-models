from sentence_transformers import SentenceTransformer
import pinecone
import streamlit as st
import os
from langchain_community.llms import Ollama


PINECONE_API_KEY = "pinecone api here"
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

model = SentenceTransformer('all-MiniLM-L6-v2')

pc = pinecone.Pinecone(
    pinecone_api_key=os.environ['PINECONE_API_KEY'],
    environment='gcp-starter'
)

index = pc.Index('chat', host="your index host here")


def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(vector=input_em, top_k=10, include_values=True, includeMetadata=True)
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string