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
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="GARUDA - Military Protocol Assistant",
    page_icon="🎖️",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "GARUDA Military Protocol Assistant v2.0"
    }
)

# Professional Military UI Styling
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
    /* CSS Variables for consistent theming */
    :root {
        --primary-color: #1a365d;
        --secondary-color: #2d3748;
        --accent-color: #3182ce;
        --success-color: #38a169;
        --warning-color: #d69e2e;
        --danger-color: #e53e3e;
        --text-primary: #ffffff;
        --text-secondary: #a0aec0;
        --bg-primary: linear-gradient(135deg, #1a202c 0%, #2d3748 50%, #4a5568 100%);
        --bg-secondary: rgba(45, 55, 72, 0.8);
        --bg-card: rgba(255, 255, 255, 0.05);
        --border-color: rgba(255, 255, 255, 0.1);
        --shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        --shadow-hover: 0 15px 35px rgba(0, 0, 0, 0.4);
    }

    /* Global Styles */
    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Container Layouts */
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 1rem;
    }

    /* Professional Header */
    .app-header {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
    }

    .app-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-color), var(--success-color), var(--accent-color));
    }

    .app-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, #ffffff, #a0aec0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .app-header h3 {
        font-size: 1.1rem;
        font-weight: 400;
        margin: 0.5rem 0 0 0;
        color: var(--text-secondary);
    }

    /* Status Bar */
    .status-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: var(--bg-card);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow);
    }

    .status-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 500;
        font-size: 0.9rem;
    }

    .status-online { color: var(--success-color); }
    .status-warning { color: var(--warning-color); }

    /* Card Components */
    .card {
        background: var(--bg-card);
        backdrop-filter: blur(15px);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
        height: fit-content;
    }

    .card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-hover);
        border-color: rgba(255, 255, 255, 0.2);
    }

    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0 0 1rem 0;
        color: var(--text-primary);
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 0.5rem;
    }

    /* Protocol Buttons */
    .protocol-grid {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .protocol-btn {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        padding: 0.75rem 1rem !important;
        text-align: left !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        margin: 0 !important;
    }

    .protocol-btn:hover {
        background: rgba(49, 130, 206, 0.1) !important;
        border-color: var(--accent-color) !important;
        transform: translateX(4px) !important;
        box-shadow: 0 4px 12px rgba(49, 130, 206, 0.2) !important;
    }

    /* Chat Interface */
    .chat-container {
        background: var(--bg-card);
        backdrop-filter: blur(15px);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow);
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
    }

    /* Custom Scrollbar */
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }

    .chat-container::-webkit-scrollbar-track {
        background: transparent;
    }

    .chat-container::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 3px;
    }

    .chat-container::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.3);
    }

    /* Message Bubbles */
    .message-user {
        background: linear-gradient(135deg, var(--accent-color), #2b6cb0);
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.75rem 0 0.25rem auto;
        max-width: 80%;
        box-shadow: 0 4px 12px rgba(49, 130, 206, 0.3);
        font-weight: 500;
    }

    .message-bot {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        padding: 1rem 1.25rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.25rem 0 0.75rem 0;
        max-width: 85%;
        box-shadow: var(--shadow);
        backdrop-filter: blur(10px);
    }

    .message-meta {
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-align: right;
        margin-top: 0.5rem;
        opacity: 0.8;
    }

    /* Input Components */
    .stTextInput > div > div > input {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 25px !important;
        color: var(--text-primary) !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--accent-color) !important;
        box-shadow: 0 0 0 3px rgba(49, 130, 206, 0.1) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-color), #2b6cb0) !important;
        border: none !important;
        border-radius: 25px !important;
        color: white !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(49, 130, 206, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(49, 130, 206, 0.4) !important;
    }

    /* Secondary Buttons */
    .btn-secondary {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
    }

    .btn-secondary:hover {
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: var(--accent-color) !important;
    }

    /* References */
    .reference-item {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        line-height: 1.4;
    }

    /* Expander Styling */
    .streamlit-expanderHeader {
        background: var(--bg-card);
        border-radius: 8px;
        padding: 0.5rem;
    }

    /* Loading Spinner */
    .stSpinner > div {
        border-color: var(--accent-color) transparent transparent transparent !important;
    }

    /* Alerts */
    .stAlert {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
    }

    .stSuccess {
        border-left: 4px solid var(--success-color) !important;
    }

    .stError {
        border-left: 4px solid var(--danger-color) !important;
    }

    .stInfo {
        border-left: 4px solid var(--accent-color) !important;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .app-header h1 { font-size: 2rem; }
        .message-user, .message-bot { max-width: 95%; }
        .status-bar { flex-direction: column; gap: 0.5rem; }
    }

    /* Animation Classes */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Professional Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    p, div, span {
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

def check_environment():
    """Check if all required environment variables and directories exist"""
    required_vars = ['GROQ_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        st.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        st.info("💡 Create a .env file with: GROQ_API_KEY=your_api_key_here")
        return False
    
    if not os.path.exists("./Data"):
        st.warning("⚠️ ./Data directory not found. Creating it now...")
        os.makedirs("./Data", exist_ok=True)
        st.info("📁 Please add your PDF documents to the ./Data directory and restart the application.")
        return False
    
    pdf_files = [f for f in os.listdir("./Data") if f.endswith('.pdf')]
    if not pdf_files:
        st.warning("⚠️ No PDF files found in ./Data directory")
        st.info("📄 Please add PDF documents to proceed.")
        return False
    
    return True

@st.cache_resource
def initialize_llm():
    """Initialize the language model"""
    try:
        return ChatGroq(
            groq_api_key=os.getenv('GROQ_API_KEY'),
            model_name="openai/gpt-oss-120b",
            temperature=0.1
        )
    except Exception as e:
        st.error(f"❌ Failed to initialize LLM: {str(e)}")
        return None

@st.cache_resource
def initialize_embeddings():
    """Initialize HuggingFace embeddings"""
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        st.error(f"❌ Failed to initialize embeddings: {str(e)}")
        return None

@st.cache_resource
def load_documents():
    """Load or create vector embeddings from documents"""
    embeddings_path = "./embeddings_cache"
    
    try:
        embeddings = initialize_embeddings()
        if not embeddings:
            return None
        
        # Check if embeddings cache exists
        if os.path.exists(embeddings_path):
            with st.spinner("📂 Loading cached embeddings..."):
                return FAISS.load_local(embeddings_path, embeddings, allow_dangerous_deserialization=True)
        
        # Create new embeddings
        with st.spinner("🔧 Creating embeddings... This may take a few minutes."):
            loader = PyPDFDirectoryLoader("./Data")
            docs = loader.load()
            
            if not docs:
                st.error("❌ No documents loaded from ./Data directory")
                return None
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            final_documents = text_splitter.split_documents(docs)
            
            # Create and cache vector store
            vector_store = FAISS.from_documents(final_documents, embeddings)
            vector_store.save_local(embeddings_path)
            st.success("✅ Embeddings created and cached successfully!")
            
            return vector_store
            
    except Exception as e:
        logger.error(f"Error in load_documents: {str(e)}")
        st.error(f"❌ Document processing error: {str(e)}")
        return None

def get_prompt_template():
    """Get the system prompt template"""
    return ChatPromptTemplate.from_template("""
    You are **GARUDA**, a Military Emergency Protocol Assistant providing **strictly accurate** guidance from official military protocol documents.

    **RESPONSE GUIDELINES:**
    1. Use ONLY official protocol documents for responses
    2. Provide complete, actionable steps with precise details
    3. For irrelevant queries, respond: "Information not available in the database"
    4. For relevant but missing protocols, provide standard emergency guidance
    5. Maintain professional military communication standards

    **Response Format:**
    - **Immediate Actions:** Critical steps requiring immediate execution
    - **Standard Procedure:** Detailed step-by-step instructions
    - **Protocol Reference:** Source section or document reference
    - **Additional Notes:** Warnings, considerations, or follow-up actions

    **Official Protocol Context:** {context}
    **Query:** {input}
    """)

def display_protocol_buttons():
    """Display quick access protocol buttons"""
    protocols = {
        "📡 Communications Emergency": "Communications equipment failure emergency procedures",
        "🔥 Fire Response": "Fire outbreak immediate response protocol",
        "💥 Explosive Threat": "Explosive device or bomb threat procedures",
        "🌡️ Medical Emergency": "Heat exhaustion and medical emergency response",
        "⚠️ CBRN Threat": "Chemical, biological, radiological, nuclear threat response",
        "❄️ Cold Weather Injuries": "Hypothermia and cold weather injury protocol",
        "📍 Survival Operations": "Survival procedures when lost or isolated",
        "🚁 Evacuation Procedures": "Emergency evacuation and extraction protocols"
    }
    
    st.markdown('<div class="protocol-grid">', unsafe_allow_html=True)
    for emoji_label, query in protocols.items():
        if st.button(emoji_label, key=f"protocol_{emoji_label}", 
                    help=query, use_container_width=True):
            return query
    st.markdown('</div>', unsafe_allow_html=True)
    return None

def process_query(query, vectors, llm):
    """Process user query and return response"""
    try:
        document_chain = create_stuff_documents_chain(llm, get_prompt_template())
        retrieval_chain = create_retrieval_chain(vectors.as_retriever(search_kwargs={"k": 3}), document_chain)
        
        start_time = time.time()
        response = retrieval_chain.invoke({'input': query})
        processing_time = time.time() - start_time
        
        return {
            'answer': response['answer'],
            'time': processing_time,
            'context': response.get('context', [])
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {
            'answer': f"❌ Error processing query: {str(e)}", 
            'time': 0, 
            'context': []
        }

def display_status_bar():
    """Display system status bar"""
    cache_exists = os.path.exists("./embeddings_cache")
    pdf_count = len([f for f in os.listdir("./Data") if f.endswith('.pdf')]) if os.path.exists("./Data") else 0
    
    st.markdown(f"""
    <div class="status-bar">
        <div class="status-item">
            <span class="status-online">●</span>
            <span>System Online</span>
        </div>
        <div class="status-item">
            <span>📅</span>
            <span>{datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
        </div>
        <div class="status-item">
            <span class="{'status-online' if cache_exists else 'status-warning'}">●</span>
            <span>Database {'Active' if cache_exists else 'Building'}</span>
        </div>
        <div class="status-item">
            <span>📄</span>
            <span>{pdf_count} Documents</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    # Environment check
    if not check_environment():
        return
    
    # Header
    st.markdown("""
    <div class="app-header fade-in">
        <h1>🎖️ GARUDA</h1>
        <h3>Advanced Military Protocol Assistant</h3>
        <p style="color: var(--text-secondary); margin-top: 0.5rem;">
            Critical Emergency Response • Real-time Protocol Access • Mission Support
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status Bar
    display_status_bar()
    
    # Initialize components
    vectors = load_documents()
    if not vectors:
        st.error("❌ Failed to initialize protocol database. Please check your setup.")
        return
    
    llm = initialize_llm()
    if not llm:
        return
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ''
    
    # Main layout
    col1, col2, col3 = st.columns([1, 2, 1], gap="large")
    
    # Left Column - Quick Protocols
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="card-title">🚀 Quick Access Protocols</h3>', unsafe_allow_html=True)
        selected_protocol = display_protocol_buttons()
        if selected_protocol:
            st.session_state.current_query = selected_protocol
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Middle Column - Chat Interface
    with col2:
        # Input area
        st.markdown('<div class="card">', unsafe_allow_html=True)
        input_col1, input_col2 = st.columns([4, 1])
        
        with input_col1:
            query = st.text_input(
                "",
                placeholder="🎯 Enter your emergency protocol query...",
                label_visibility="collapsed",
                value=st.session_state.current_query,
                key="main_input"
            )
        
        with input_col2:
            send_clicked = st.button("🚀 Send", use_container_width=True, type="primary")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process query
        if send_clicked and query:
            with st.spinner("🔍 Analyzing protocols..."):
                response = process_query(query, vectors, llm)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': query,
                    'answer': response['answer'],
                    'time': response['time'],
                    'context': response['context'],
                    'timestamp': datetime.now()
                })
                
                # Clear input
                st.session_state.current_query = ''
                st.rerun()
        
        # Chat History
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: var(--text-secondary);">
                <h3>🎖️ GARUDA Protocol Assistant Ready</h3>
                <p>Select a quick protocol or type your emergency query above</p>
                <p><em>Providing critical military protocol guidance when you need it most</em></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Display recent chats (last 5)
            for chat in reversed(st.session_state.chat_history[-5:]):
                st.markdown(f"""
                <div class="message-user fade-in">
                    <strong>🎯 Query:</strong> {chat['question']}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="message-bot fade-in">
                    <strong>📋 Protocol Response:</strong><br>
                    {chat['answer'].replace(chr(10), '<br>')}
                    <div class="message-meta">
                        ⏱️ {chat['time']:.2f}s • {chat['timestamp'].strftime('%H:%M:%S')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Right Column - References & Management
    with col3:
        # References
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="card-title">📚 Protocol References</h3>', unsafe_allow_html=True)
        
        if st.session_state.chat_history:
            latest = st.session_state.chat_history[-1]
            if latest.get('context'):
                for idx, doc in enumerate(latest['context'][:3], 1):
                    with st.expander(f"📄 Reference Document {idx}", expanded=False):
                        content = doc.page_content
                        preview = content[:250] + "..." if len(content) > 250 else content
                        st.markdown(f'<div class="reference-item">{preview}</div>', unsafe_allow_html=True)
            else:
                st.info("No references available for the last query.")
        else:
            st.info("Protocol references will appear here after your first query.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # System Management
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="card-title">⚙️ System Management</h3>', unsafe_allow_html=True)
        
        # Management buttons
        col_clear, col_cache = st.columns(2)
        
        with col_clear:
            if st.button("🗑️ Clear Chat", use_container_width=True, help="Clear chat history"):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
                time.sleep(1)
                st.rerun()
        
        with col_cache:
            if st.button("🔄 Reset Cache", use_container_width=True, 
                        help="Clear embeddings cache to rebuild"):
                cache_path = "./embeddings_cache"
                if os.path.exists(cache_path):
                    shutil.rmtree(cache_path)
                    st.success("Cache cleared! Please restart to rebuild.")
                else:
                    st.info("No cache found.")
        
        # System info
        cache_exists = os.path.exists("./embeddings_cache")
        st.markdown(f"""
        <div style="margin-top: 1rem; padding: 1rem; background: rgba(255,255,255,0.03); border-radius: 8px; border: 1px solid var(--border-color);">
            <div><strong>Cache Status:</strong> {'🟢 Active' if cache_exists else '🟡 Building'}</div>
            <div><strong>Sessions:</strong> {len(st.session_state.chat_history)} queries</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()