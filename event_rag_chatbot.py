import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
PDF_PATH = "our_services.pdf"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4

# System prompt reflects the source document's own "RAG Response Guidelines"
# section (§14): don't invent pricing/availability, flag indicative vs
# confirmed info, cite the source section, and ask for missing booking details.
SYSTEM_PROMPT = """You are the virtual assistant for BlueSky Events & Experiences,
an event management company. Answer ONLY using the retrieved context below,
which comes from the company's official knowledge base.

Rules you must follow:
1. Never invent pricing, availability, vendors, venues, or policies that
   are not in the retrieved context.
2. If the retrieved context does not contain enough information to answer,
   say so clearly and suggest the user contact the events team
   (events@blueskyevents.example) for confirmation.
3. All prices are INDICATIVE unless the context explicitly says otherwise —
   always mention this when quoting a price.
4. Never claim a venue, vendor, artist, or date is available — availability
   must always be confirmed by the events team.
5. For booking-related questions, ask the user for any missing details
   (event date, location, guest count, event type) needed to help them.
6. Keep answers concise, friendly, and professional.

Context:
{context}
"""

st.set_page_config(page_title="BlueSky Events Assistant", page_icon="🎉", layout="wide")


# --------------------------------------------------------------------------
# Build (and cache) the RAG pipeline
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_rag_chain(pdf_path: str, model_name: str):
    # 1. Load
    docs = PyPDFLoader(pdf_path).load()

    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n====", "\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)

    # 3. Embed + store
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # 4. LLM
    llm = ChatOllama(model=model_name, temperature=0.2)

    # 5. Prompt + chain (LCEL)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    def format_docs(retrieved):
        return "\n\n---\n\n".join(d.page_content for d in retrieved)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def get_available_models():
    try:
        import ollama
        response = ollama.list()
        if hasattr(response, "models"):
            return [m.model for m in response.models]
        elif isinstance(response, dict) and "models" in response:
            return [m.get("name") or m.get("model") for m in response["models"]]
        return ["llama3.2"]
    except Exception:
        return ["llama3.2", "llama3.2:1b", "mistral", "phi3"]


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    models = get_available_models()
    selected_model = st.selectbox("Choose an Ollama model:", options=models, index=0)

    st.write("---")
    st.subheader("📄 Knowledge Base")

    pdf_ready = os.path.exists(PDF_PATH)
    if pdf_ready:
        st.success(f"Loaded: {PDF_PATH}")
    else:
        st.warning("our_services.pdf not found in app folder.")
        uploaded = st.file_uploader("Upload the knowledge base PDF", type="pdf")
        if uploaded:
            with open(PDF_PATH, "wb") as f:
                f.write(uploaded.read())
            st.success("Uploaded — reloading...")
            st.cache_resource.clear()
            st.rerun()

    st.write("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    with st.expander("ℹ️ How this works"):
        st.markdown(
            "1. PDF is split into chunks\n"
            "2. Chunks are embedded with **nomic-embed-text**\n"
            "3. Your question retrieves the top matching chunks (FAISS)\n"
            "4. The chunks + question are sent to the local LLM\n"
            "5. Answer is grounded only in the retrieved content"
        )

# --------------------------------------------------------------------------
# Main chat interface
# --------------------------------------------------------------------------
st.title("🎉 BlueSky Events — AI Assistant")
st.caption(f"Answers grounded in the company knowledge base · Model: **{selected_model}**")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not pdf_ready:
    st.info("👆 Upload `our_services.pdf` in the sidebar to start chatting.")
else:
    chain, retriever = build_rag_chain(PDF_PATH, selected_model)

    if prompt := st.chat_input("Ask about packages, pricing, booking, policies..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"Thinking using {selected_model}..."):
                try:
                    answer = chain.invoke(prompt)
                except Exception as e:
                    answer = f"⚠️ Error generating response: {e}"

                st.markdown(answer)

                with st.expander("🔍 Sources used"):
                    for i, doc in enumerate(retriever.invoke(prompt), 1):
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**Chunk {i} (page {page}):**")
                        st.caption(doc.page_content[:300] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
