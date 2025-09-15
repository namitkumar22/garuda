import streamlit as st
import os
import shutil
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="GARUDA - Military Protocol Assistant",
    page_icon="🎖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .status-panel {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .protocol-btn {
        background: white;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 0.7rem;
        margin: 0.3rem 0;
        width: 100%;
        text-align: left;
        transition: all 0.2s;
        cursor: pointer;
    }
    
    .protocol-btn:hover {
        background: #f1f5f9;
        border-color: #3b82f6;
        transform: translateX(3px);
    }
    
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
    }
    
    .chat-input {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 1rem;
        border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_llm():
    """Initialize language model"""
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        st.error("❌ GROQ_API_KEY not found in environment variables")
        st.stop()
    
    return ChatGroq(
        groq_api_key=api_key,
        model_name="openai/gpt-oss-120b",
        temperature=0.1
    )

@st.cache_resource
def initialize_embeddings():
    """Initialize embeddings model"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def load_vectorstore():
    """Load or create vector store with persistent caching"""
    # Check data directory
    if not os.path.exists("./Data"):
        os.makedirs("./Data", exist_ok=True)
        st.error("❌ ./Data directory was created. Please add PDF files and restart.")
        st.stop()
    
    pdf_files = [f for f in os.listdir("./Data") if f.endswith('.pdf')]
    if not pdf_files:
        st.error("❌ No PDF files found in ./Data directory")
        st.info("📄 Please add PDF documents to the ./Data folder")
        st.stop()
    
    embeddings = initialize_embeddings()
    cache_path = "./embeddings_cache"
    
    # Try to load existing embeddings
    if os.path.exists(cache_path):
        try:
            with st.spinner("📂 Loading cached embeddings..."):
                vectorstore = FAISS.load_local(
                    cache_path, 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                st.success(f"✅ Loaded cached embeddings for {len(pdf_files)} documents")
                return vectorstore
        except Exception as e:
            st.warning(f"⚠️ Cache corrupted: {str(e)}. Rebuilding...")
            shutil.rmtree(cache_path)
    
    # Create new embeddings
    with st.spinner("🔧 Processing documents and creating embeddings... This may take a few minutes."):
        try:
            # Load documents
            loader = PyPDFDirectoryLoader("./Data")
            docs = loader.load()
            
            if not docs:
                st.error("❌ No documents could be loaded from PDF files")
                st.stop()
            
            # Split documents
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            split_docs = splitter.split_documents(docs)
            
            st.info(f"📝 Processing {len(split_docs)} document chunks...")
            
            # Create vector store
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            
            # Save to cache
            vectorstore.save_local(cache_path)
            st.success(f"✅ Created and cached embeddings for {len(pdf_files)} PDF files")
            
            return vectorstore
            
        except Exception as e:
            st.error(f"❌ Error creating embeddings: {str(e)}")
            st.stop()

def get_system_prompt():
    """System prompt template"""
    return ChatPromptTemplate.from_template("""
You are GARUDA, a Military Protocol Assistant. Provide accurate, actionable guidance based on official military protocols.

**Guidelines:**
- Use only information from the provided documents
- Give clear, step-by-step instructions
- Include safety warnings when relevant
- Maintain professional military communication
- If information is not available, state clearly

**Context:** {context}
**Question:** {input}

Provide a structured response with:
1. Immediate Actions (if emergency)
2. Detailed Procedure
3. Safety Considerations
4. Reference Information
""")

def process_query(query, vectorstore, llm):
    """Process user query"""
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        document_chain = create_stuff_documents_chain(llm, get_system_prompt())
        rag_chain = create_retrieval_chain(retriever, document_chain)
        
        response = rag_chain.invoke({"input": query})
        return response["answer"]
    
    except Exception as e:
        return f"❌ Error processing query: {str(e)}"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "llm" not in st.session_state:
    st.session_state.llm = None

# Header
st.markdown("""
<div class="main-header">
    <h1>🎖️ GARUDA</h1>
    <h3>Military Protocol Assistant</h3>
    <p>Emergency Response • Protocol Guidance • Mission Support</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🚀 Quick Protocols")
    
    protocols = {
        "📡 Communication Failure": "Communication equipment failure emergency procedures",
        "🔥 Fire Emergency": "Fire outbreak immediate response protocol",
        "💥 Explosive Threat": "Explosive device or bomb threat procedures", 
        "🚑 Medical Emergency": "Medical emergency and casualty response",
        "⚠️ CBRN Alert": "Chemical, biological, radiological threat response",
        "❄️ Cold Injury": "Hypothermia and cold weather injury protocol",
        "🧭 Survival": "Survival procedures when lost or isolated",
        "🚁 Evacuation": "Emergency evacuation and extraction protocols"
    }
    
    for label, query in protocols.items():
        if st.button(label, key=f"btn_{label}", help=query):
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()
    
    st.divider()
    
    # System status
    st.header("📊 System Status")
    
    cache_exists = os.path.exists("./embeddings_cache")
    cache_size = "Unknown"
    
    if cache_exists:
        try:
            cache_files = os.listdir("./embeddings_cache")
            cache_size = f"{len(cache_files)} files"
        except:
            cache_size = "Error reading"
    
    pdf_count = 0
    if os.path.exists("./Data"):
        pdf_count = len([f for f in os.listdir("./Data") if f.endswith('.pdf')])
    
    st.markdown(f"""
    <div class="status-panel">
        <p>🟢 <strong>System:</strong> Online</p>
        <p>📄 <strong>Documents:</strong> {pdf_count} PDFs</p>
        <p>💾 <strong>Embeddings:</strong> {'✅ Cached' if cache_exists else '❌ Not Built'}</p>
        <p>📊 <strong>Cache Size:</strong> {cache_size}</p>
        <p>💬 <strong>Messages:</strong> {len(st.session_state.messages)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Management buttons
    st.header("⚙️ Management")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🔄 Rebuild Cache", use_container_width=True):
        if os.path.exists("./embeddings_cache"):
            shutil.rmtree("./embeddings_cache")
            st.session_state.vectorstore = None
            # Clear the cache so it rebuilds on next load
            st.cache_resource.clear()
            st.success("🔄 Cache cleared! Reloading page...")
            time.sleep(2)
            st.rerun()
        else:
            st.info("No cache found to clear")
    
    # Display cache info
    if os.path.exists("./embeddings_cache"):
        try:
            cache_files = os.listdir("./embeddings_cache")
            st.caption(f"📂 Cache contains {len(cache_files)} files")
        except:
            st.caption("📂 Cache status unknown")

# Main chat area
st.header("💬 Protocol Chat")

# Initialize components with proper caching
try:
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = load_vectorstore()
    
    if st.session_state.llm is None:
        st.session_state.llm = initialize_llm()
        
except Exception as e:
    st.error(f"❌ Initialization failed: {str(e)}")
    st.info("💡 Try clearing the cache and restarting the application")
    st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Enter your protocol query..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing protocols..."):
            response = process_query(
                prompt, 
                st.session_state.vectorstore, 
                st.session_state.llm
            )
            st.write(response)
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})

# Welcome message
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write("""
        👋 Welcome to **GARUDA Military Protocol Assistant**!
        
        I'm here to provide you with accurate military protocol guidance from official documents.
        
        **How to use:**
        - Use the Quick Protocols in the sidebar for common emergencies
        - Type your specific protocol questions in the chat
        - I'll provide structured, actionable responses
        
        **Ready to assist with your protocol queries!** 🎖️
        """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 1rem;">
    <p><strong>GARUDA v2.0</strong> | Military Protocol Assistant | 
    For training and reference purposes</p>
</div>
""", unsafe_allow_html=True)