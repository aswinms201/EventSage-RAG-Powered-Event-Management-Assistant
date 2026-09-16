# 🎉 EventSage — RAG-Powered Event Management Assistant

A Retrieval-Augmented Generation (RAG) chatbot built with **LangChain**, **Ollama**, and **Streamlit** that answers questions grounded in a company's own knowledge base — running **fully locally**, with no cloud APIs or API keys required.

Built as a demo for an event management company (BlueSky Events & Experiences), answering questions about packages, pricing, policies, and booking processes using only the company's official service document.

## ✨ Features

- 🔍 **Retrieval-Augmented Generation** — answers are grounded in the source PDF, not the model's general knowledge
- 🧠 **Fully local LLM stack** — runs entirely offline via Ollama (no OpenAI/API keys needed)
- 📄 **PDF knowledge base ingestion** — automatically chunks and embeds any PDF you provide
- 🗂️ **Source citations** — shows which document chunks were used to generate each answer
- 🛡️ **Guardrailed responses** — system prompt enforces "don't invent pricing/availability" rules pulled directly from the source document's own response guidelines
- 💬 **Streamlit chat UI** — familiar chat-bubble interface with model switching and chat history

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| UI | [Streamlit](https://streamlit.io/) |
| Orchestration | [LangChain](https://www.langchain.com/) (LCEL) |
| LLM + Embeddings | [Ollama](https://ollama.com/) (`llama3.2`, `nomic-embed-text`) |
| Vector Store | [FAISS](https://github.com/facebookresearch/faiss) |
| PDF Parsing | [pypdf](https://pypi.org/project/pypdf/) |

## 📦 Installation

1. **Install [Ollama](https://ollama.com/)** and pull the required models:
```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
```

2. **Clone this repository**
```bash
   git clone https://github.com/<your-username>/eventsage.git
   cd eventsage
```

3. **Install dependencies**
```bash
   pip install streamlit langchain langchain-core langchain-community langchain-ollama langchain-text-splitters faiss-cpu pypdf
```

4. **Add your knowledge base PDF**
   Place your source document (e.g. `our_services.pdf`) in the project folder, or upload it via the sidebar on first launch.

5. **Run the app**
```bash
   streamlit run event_rag_chatbot.py
```

## 🚀 Usage

- Select your Ollama chat model from the sidebar (embedding-only models like `nomic-embed-text` are used internally and shouldn't be selected as the chat model).
- Ask questions about the knowledge base — pricing, packages, policies, booking process, etc.
- Expand **"Sources used"** under any answer to see which document chunks were retrieved.
- Use **Clear Chat** to reset the conversation.

## 🧩 How It Works
