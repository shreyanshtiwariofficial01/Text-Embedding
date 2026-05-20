🚗 AI Manual Assistant
Semantic Search using Text Embeddings

An AI-powered semantic search system that helps users ask natural language questions and instantly find the most relevant sections from a vehicle manual.
--------------------------------------------------------------------------------------------
📌 About the Project

AI Manual Assistant is a semantic search application built to understand intent, not just keywords.
It allows users to ask questions in plain English and retrieves the most relevant vehicle manual section using text embeddings and cosine similarity.

This project focuses on the retrieval component of a RAG (Retrieval-Augmented Generation) system, implemented without using an LLM, making it lightweight, fast, and easy to understand.
------------------------------------------------------------------------------
✨ Features

🔍 Semantic Search over vehicle manual content

🧠 Sentence Embeddings using SentenceTransformers

📊 Cosine Similarity Ranking for relevance scoring

⚡ Cached Model & Embeddings for faster responses

🖥️ Interactive Streamlit UI

📈 Similarity Score Visualization across all topics
-----------------------------------------------------------------------------------------
🏗️ System Flow
User Question
     ↓
Query Embedding
     ↓
Cosine Similarity
     ↓
Rank Manual Sections
     ↓
Best Matching Answer
----------------------------------------------------------------------------------------
🛠️ Technology Stack

Python

Streamlit – Web Interface

SentenceTransformers (all-MiniLM-L6-v2)

NumPy

Pandas
----------------------------------------------------------------------------------------
📂 Knowledge Base

The system uses predefined vehicle manual sections, including:

Windshield Wipers

Bluetooth Connectivity

Adaptive Cruise Control

Parking Brake

Voice Commands

Charging the Vehicle

Each section is embedded once and reused for all user queries to ensure fast performance.
--------------------------------------------------------------------------------------
🚀 How It Works

Manual content is converted into vector embeddings

User submits a natural language query

The query is embedded using the same model

Cosine similarity is calculated against all documents

The most relevant section is returned with a confidence score
------------------------------------------------------------------------------------
▶️ Getting Started
🔧 Installation
pip install streamlit sentence-transformers pandas numpy
-----------------------------------------------------------------------------------
▶️ Run the Application
python -m streamlit run app.py
----------------------------------------------------------------------------------
📊 Application Output

✅ Best-matching manual section

✅ Relevance (similarity) score

✅ Comparison of similarity across all topics
----------------------------------------------------------------------------------
🎯 Use Cases

AI-powered product manuals

FAQ & help-desk search systems

Knowledge base assistants

Retrieval layer for RAG pipelines

🔮 Future Enhancements

📄 PDF / DOCX document ingestion

✂️ Text chunking for large documents

🧠 Vector database integration (FAISS / ChromaDB)

🤖 LLM-powered answer generation

📁 Multi-document support
------------------------------------------------------------------------------
📁 Project Structure
.
├── app.py
├── README.md
├── requirements.txt
└── data/
------------------------------------------------------------------------------
👨‍💻 Author

Saif Ibrahim
Beginner AI & Data Science Practitioner
Interested in NLP, embeddings, and applied AI systems
-----------------------------------------------------------------------------
⭐ Final Note

This project demonstrates core AI retrieval concepts used in modern search engines and RAG-based systems and is well-suited for learning, portfolio building, and interviews.
