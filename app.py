import streamlit as st
import os
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

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Page configuration
st.set_page_config(
    page_title="Garuda - Military Protocol Assistant",
    page_icon="🎖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern UI styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
    }
    
    .chat-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .message-user {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .message-bot {
        background: rgba(255, 255, 255, 0.9);
        color: #333;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .status-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .protocol-btn {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        color: white !important;
        margin: 0.2rem 0 !important;
        transition: all 0.3s ease !important;
    }
    
    .protocol-btn:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2) !important;
    }
    
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 25px !important;
        color: white !important;
        padding: 0.75rem 1.5rem !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        border: none !important;
        border-radius: 25px !important;
        color: white !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3) !important;
    }
    
    h1, h2, h3 {
        color: white !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
    }
    
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_llm():
    return ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-specdec")

@st.cache_resource
def initialize_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def load_documents():
    try:
        embeddings = initialize_embeddings()
        loader = PyPDFDirectoryLoader("./Data")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        return FAISS.from_documents(final_documents, embeddings)
    except Exception as e:
        st.error(f"❌ Protocol database initialization error: {str(e)}")
        return None

def get_prompt_template():
    return ChatPromptTemplate.from_template("""
    You are a **Military Emergency Protocol Assistant** providing **strictly accurate** guidance from **official military protocol documents**.

    **RESPONSE GUIDELINES:**
    1. Use ONLY official protocol documents for responses
    2. Provide complete, actionable steps - no vague references
    3. For irrelevant queries, respond: "Information not available in the database"
    4. For relevant but missing protocols, provide standard emergency guidance

    **Format:**
    - **Immediate Actions:** Critical steps (if applicable)
    - **Procedure:** Detailed step-by-step instructions
    - **Reference:** Protocol source

    **Context:** {context}
    **Query:** {input}
    """)

def display_protocol_buttons():
    protocols = {
        "📡 Communications Failure": "Communications equipment failure procedures",
        "🔥 Fire Emergency": "Fire outbreak response in the field",
        "💥 Explosive Threat": "Explosive or bomb threat immediate steps",
        "🌡️ Heat Emergency": "Heat exhaustion signs and first aid",
        "⚠️ Chemical Attack": "Biological or chemical attack response",
        "❄️ Cold Injuries": "Hypothermia prevention and treatment",
        "📍 Navigation Emergency": "Survival steps when lost"
    }
    
    for emoji_label, query in protocols.items():
        if st.button(emoji_label, key=f"protocol_{emoji_label}", help=query, use_container_width=True):
            return query
    return None

def process_query(query, vectors, llm):
    try:
        document_chain = create_stuff_documents_chain(llm, get_prompt_template())
        retrieval_chain = create_retrieval_chain(vectors.as_retriever(), document_chain)
        
        start_time = time.time()
        response = retrieval_chain.invoke({'input': query})
        processing_time = time.time() - start_time
        
        return {
            'answer': response['answer'],
            'time': processing_time,
            'context': response.get('context', [])
        }
    except Exception as e:
        return {'answer': f"Error processing query: {str(e)}", 'time': 0, 'context': []}

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎖️ GARUDA</h1>
        <h3>Military Emergency Response System</h3>
        <p>Advanced Protocol Assistant for Critical Situations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components
    vectors = load_documents()
    if not vectors:
        st.error("Failed to initialize protocol database")
        return
    
    llm = initialize_llm()
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    # Left Column - Quick Protocols
    with col1:
        st.markdown("### 🚀 Quick Protocols")
        selected_protocol = display_protocol_buttons()
        if selected_protocol:
            st.session_state.current_query = selected_protocol
    
    # Middle Column - Chat Interface
    with col2:
        # Status indicators
        status_col1, status_col2, status_col3 = st.columns(3)
        with status_col1:
            st.markdown('<div class="status-card">🟢 System Online</div>', unsafe_allow_html=True)
        with status_col2:
            st.markdown(f'<div class="status-card">📅 {datetime.now().strftime("%Y-%m-%d")}</div>', unsafe_allow_html=True)
        with status_col3:
            st.markdown('<div class="status-card">🗄️ Database Active</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Input area
        input_col1, input_col2 = st.columns([4, 1])
        with input_col1:
            query = st.text_input(
                "Enter your emergency protocol query...",
                placeholder="What emergency procedure do you need?",
                label_visibility="collapsed",
                value=st.session_state.get('current_query', '')
            )
        with input_col2:
            send_clicked = st.button("🚀 Send", use_container_width=True)
        
        # Process query
        if (send_clicked and query) or st.session_state.get('current_query'):
            current_query = query or st.session_state.get('current_query', '')
            if current_query:
                with st.spinner("🔍 Analyzing protocols..."):
                    response = process_query(current_query, vectors, llm)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': current_query,
                        'answer': response['answer'],
                        'time': response['time'],
                        'context': response['context']
                    })
                    
                    # Clear current query
                    st.session_state.current_query = ''
                    st.rerun()
        
        # Display chat history
        st.markdown("### 💬 Protocol Responses")
        chat_container = st.container()
        with chat_container:
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                st.markdown(f'<div class="message-user"><strong>Query:</strong> {chat["question"]}</div>', 
                           unsafe_allow_html=True)
                st.markdown(f'<div class="message-bot"><strong>Protocol Response:</strong><br>{chat["answer"].replace(chr(10), "<br>")}</div>', 
                           unsafe_allow_html=True)
                st.markdown(f"<small>⏱️ Response time: {chat['time']:.2f}s</small>", unsafe_allow_html=True)
                st.markdown("---")
    
    # Right Column - References
    with col3:
        st.markdown("### 📚 References")
        if st.session_state.chat_history:
            latest = st.session_state.chat_history[-1]
            if latest.get('context'):
                for idx, doc in enumerate(latest['context'][:3], 1):  # Show top 3
                    with st.expander(f"📄 Reference {idx}", expanded=False):
                        st.markdown(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()