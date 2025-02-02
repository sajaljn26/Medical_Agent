# Medical Chatbot with LangChain, Hugging Face, and FAISS

This project is a **Medical Chatbot** designed to provide accurate and context-aware responses to medical-related queries. It leverages the power of **LangChain**, **Hugging Face models**, and **FAISS** for efficient document retrieval and question answering. The chatbot is built to assist users with medical information while ensuring ethical use and adherence to privacy standards.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)

---

## Project Overview

The Medical Chatbot is designed to:
- Provide accurate and context-aware responses to medical queries.
- Retrieve relevant information from a pre-built FAISS vector store of medical documents.
- Use the **google/gemma-2-2b-it** model from Hugging Face for generating responses.
- Ensure ethical use by avoiding the generation of unverified or harmful medical advice.

This project is intended for educational and informational purposes only and should not replace professional medical advice.

---

## Features

- **Context-Aware Responses**: The chatbot uses a custom prompt template to ensure responses are based on the provided context.
- **Efficient Document Retrieval**: FAISS is used for fast and accurate retrieval of relevant medical documents.
- **Customizable Prompt**: The prompt template can be modified to suit specific use cases or ethical guidelines.
- **Hugging Face Integration**: Leverages state-of-the-art language models for generating high-quality responses.
- **Privacy-First Approach**: No user data is stored or shared.

---

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.8 or higher
- `langchain-huggingface`
- `langchain-core`
- `langchain-community`
- `python-dotenv`
- `faiss-cpu` (or `faiss-gpu` if you have a compatible GPU)

You also need a **Hugging Face API token**. You can obtain one by signing up at [Hugging Face](https://huggingface.co/).

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/medical-chatbot.git
   cd medical-chatbot