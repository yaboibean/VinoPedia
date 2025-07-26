
import streamlit as st
from openai import OpenAI
# --- Follow-up and Common Questions Section ---
import random

def generate_followup_questions(last_question):
    if not last_question:
        # Commonly asked questions
        return [
            "What are the best wine pairings for summer dishes?",
            "How should I store my wine collection properly?",
            "What's the difference between Old World and New World wines?"
        ]
    prompt = f"""Based on this wine-related question: \"{last_question}\"\n\nGenerate 5 natural follow-up questions that someone might ask next. Make them specific and relevant to wine knowledge that would likely be covered in wine magazines.\n\nFormat as a simple list:\n1. [question]\n2. [question]\n3. [question]\n4. [question]\n5. [question]"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        lines = answer.split('\n')
        questions = []
        for line in lines:
            if line.strip() and any(line.startswith(f'{i}.') for i in range(1, 6)):
                q = line.split('.', 1)[1].strip()
                questions.append(q)
        # Fallbacks if not enough
        fallback_questions = [
            "Tell me more about wine terminology",
            "What are some wine tasting techniques?",
            "How do wine regions affect flavor?"
        ]
        for fallback in fallback_questions:
            if len(questions) >= 5:
                break
            if fallback not in questions:
                questions.append(fallback)
        return questions[:5]
    except Exception as e:
        return [
            "Tell me more about wine styles",
            "What are some wine tasting tips?",
            "How do I choose the right wine?"
        ]
import faiss
import numpy as np
import pickle
import os
import logging

# Configure logging (for local debug)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Load environment and data ---
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY not set in environment or .env file")
    st.stop()
client = OpenAI(api_key=openai_api_key)

# Load index + chunk text with error handling
try:
    if not os.path.exists("magazine_index.faiss"):
        st.error("❌ magazine_index.faiss not found! Run extract_and_index.py first.")
        st.stop()
    if not os.path.exists("magazine_chunks.pkl"):
        st.error("❌ magazine_chunks.pkl not found! Run extract_and_index.py first.")
        st.stop()
    index = faiss.read_index("magazine_index.faiss")
    with open("magazine_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    if len(chunks) == 0:
        st.error("❌ No chunks found in pickle file!")
        st.stop()
except Exception as e:
    st.error(f"❌ Failed to load index/chunks: {str(e)}")
    index = None
    chunks = []


def embed_query(query):
    response = client.embeddings.create(
        input=[query],
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding, dtype="float32")

# --- Custom CSS for exact match ---
st.markdown('''<style>
body, .stApp {
    background: radial-gradient(ellipse at center, #3d0d16 0%, #2a0710 100%) !important;
}
.main-box {
    background: #f7f3f3;
    border-radius: 22px;
    width: 1100px;
    margin: 48px auto 0 auto;
    box-shadow: 0 4px 36px rgba(60,0,20,0.13), 0 1.5px 8px rgba(60,0,20,0.07);
    min-height: 900px;
    display: flex;
    flex-direction: column;
    align-items: stretch;
    position: relative;
    padding: 44px 54px 44px 54px;
    border: 1.5px solid #e9e3ea;
}
.header-title {
    text-align: center;
    font-size: 2.9em;
    font-weight: 900;
    color: #2a0710;
    margin-bottom: 22px;
    margin-top: 0px;
    letter-spacing: 1.2px;
    font-family: 'Lato', 'Arial', sans-serif;
    text-shadow: 0 2px 8px #f7e9f3, 0 1px 0 #fff;
}
.main-content-row {
    display: flex;
    flex-direction: row;
    align-items: flex-start;
    justify-content: stretch;
    width: 100%;
    flex: 1;
    min-height: 600px;
    gap: 32px;
}
.main-chat-col {
    flex: 2;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    padding: 0 0 0 0;
    min-height: 600px;
    background: #fff;
    border-radius: 18px;
    box-shadow: 0 2px 18px rgba(60,0,20,0.07);
    border: 1.5px solid #ede6f0;
    margin-bottom: 0;
    margin-top: 0;
    position: relative;
}
.main-followup-col {
    flex: 1;
    min-width: 320px;
    max-width: 340px;
    margin-left: 32px;
    margin-top: 24px;
}
.chat-area-bg {
    background: #d3d3d3;
    border-radius: 16px;
    min-height: 420px;
    max-width: 800px;
    margin: 0 auto 0 auto;
    padding: 32px 24px 0 24px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
}
.empty-state {
    color: #7a2a3a;
    font-size: 1.18em;
    text-align: center;
    margin-top: 38px;
    font-family: 'Lato', 'Arial', sans-serif;
    font-weight: 600;
    letter-spacing: 0.2px;
    text-shadow: 0 1px 0 #fff, 0 2px 6px #e9e3ea;
}
.input-row {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: flex-start;
    margin: 32px auto 0 auto;
    max-width: 700px;
    width: 100%;
    gap: 16px;
    background: #f7f3f6;
    border-radius: 12px;
    box-shadow: 0 1.5px 8px rgba(60,0,20,0.06);
    padding: 12px 18px 12px 18px;
    border: 1.2px solid #e9e3ea;
}
.followup-panel {
    background: #f7f3f6;
    border-radius: 18px;
    box-shadow: 0 2px 16px rgba(90,24,50,0.09);
    padding: 28px 22px 22px 22px;
    border: 1.5px solid #e9e3ea;
}
.followup-title {
    color:#291010; 
    font-size:1.18em; 
    font-weight:700; 
    letter-spacing:0.5px; 
    text-shadow:0 1px 0 #fff, 0 2px 6px #e9e3ea; 
    margin-bottom: 18px;
    font-family: 'Lato', 'Arial', sans-serif;
}
.followup-spinner {
    margin:10px 0 16px 0; 
    color:#a8325a; 
    background:#f7f3f6; 
    border-radius:10px; 
    padding:7px 14px; 
    font-size:0.98em; 
    font-weight:500; 
    display:flex; 
    align-items:center; 
    border:1px solid #e9e3ea; 
    justify-content:left;
}
.followup-spinner .spinner {
    display:inline-block;
    width:15px;
    height:15px;
    border:2.5px solid #e9ecef;
    border-top:2.5px solid #a8325a;
    border-radius:50%;
    margin-right:8px;
    animation:spin 1s linear infinite;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>''', unsafe_allow_html=True)







# --- Single main box: all content inside a single .main-box div ---


# --- All content inside a single main-box div, but use Streamlit widgets for interactivity ---
st.markdown('<div class="main-box">', unsafe_allow_html=True)
st.markdown('<div class="header-title">Sommelier India\'s Cellar Sage</div>', unsafe_allow_html=True)

# Main content row: chat and followup columns
chat_col, followup_col = st.columns([2, 1], gap="large")

with chat_col:
    # Chat area (empty state or chat bubbles)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if not st.session_state.chat_history:
        st.markdown('<div class="empty-state">Tap into decades of wine wisdom from the Sommelier India Archives</div>', unsafe_allow_html=True)
    else:
        # Render chat bubbles (optional, can be improved)
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f'<div style="text-align:left;margin:10px 0 0 0;"><span style="background:#f7e3ea;color:#2a0710;padding:10px 18px 10px 16px;border-radius:13px 13px 13px 3px;font-size:1.08em;font-family:Lato,Arial,sans-serif;display:inline-block;max-width:90%;">{msg["content"]}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="text-align:left;margin:10px 0 0 0;"><span style="background:#e9e3ea;color:#1a1a1a;padding:10px 18px 10px 16px;border-radius:13px 13px 3px 13px;font-size:1.08em;font-family:Lato,Arial,sans-serif;display:inline-block;max-width:90%;">{msg["content"]}</span></div>', unsafe_allow_html=True)

    # Input row at the bottom
    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
    with st.container():
        input_cols = st.columns([8,2])
        with input_cols[0]:
            question = st.text_input(
                "",
                placeholder="What would you like to know about wine?",
                key="question_input_box",
                label_visibility="collapsed",
                value=st.session_state.get("question_input_box", "")
            )
        with input_cols[1]:
            ask_button = st.button("Ask", key="ask_button", use_container_width=True)

with followup_col:
    st.markdown('<div class="followup-title">Follow-up & Common Questions</div>', unsafe_allow_html=True)
    thinking = st.session_state.get('thinking', False)
    if thinking:
        st.markdown('<div class="followup-spinner"><span class="spinner"></span><span>Generating follow-up questions...</span></div>', unsafe_allow_html=True)
    else:
        followup_questions = generate_followup_questions(st.session_state.get('last_question', ''))
        import hashlib
        last_q = st.session_state.get('last_question', '')
        key_prefix = hashlib.md5(last_q.encode('utf-8')).hexdigest()[:8] if last_q else "init"
        for i, q in enumerate(followup_questions):
            btn_key = f"followup_btn_{key_prefix}_{i}"
            if st.button(q, key=btn_key, help="Click to ask this question", disabled=thinking):
                if not thinking:
                    st.session_state.last_question = q
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    st.session_state.question_input_box = ""
                    st.session_state.thinking = True
                    st.experimental_rerun()

st.markdown('</div>', unsafe_allow_html=True)


# --- Handle question submission and response ---
if ask_button and question:
    st.session_state.last_question = question
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.question_input_box = ""
    with st.spinner("Thinking..."):
        try:
            query_embedding = embed_query(question)
            D, I = index.search(np.array([query_embedding]), k=3)
            relevant_chunks = []
            for idx in I[0]:
                if idx < len(chunks):
                    chunk = chunks[idx]
                    truncated_chunk = chunk[:800] + "..." if len(chunk) > 800 else chunk
                    relevant_chunks.append(truncated_chunk)
            relevant = "\n\n".join(relevant_chunks)
            prompt = f"""You are a helpful wine expert assistant answering questions based on wine magazine content.\n\nHere is relevant context from the wine magazines:\n{relevant}\n\nQuestion: {question}\n\nInstructions:\n- Keep responses concise but informative (2-4 paragraphs max)\n- Use bullet points for key information\n- Include specific wine terminology and expert insights\n- Quote directly from magazines when relevant (use quotation marks)\n- If magazines don't contain specific info, state this briefly\n- End with source citations: \"Sommelier India, <issue number>, <year>\"\n\nBe direct and focused - provide depth without being wordy."""
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.3
            )
            answer = response.choices[0].message.content
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            error_message = f"Error generating answer: {str(e)}\n\nTraceback:\n{tb}\n\nOPENAI_API_KEY present: {'Yes' if openai_api_key else 'No'}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})
    st.experimental_rerun()


def calculate_recency_bias(chunk):
    """Calculate recency bias score - higher score for more recent content"""
    try:
        # Extract year from chunk text (look for patterns like "2023", "2024", "2025")
        import re
        
        # Look for 4-digit years in the chunk
        years = re.findall(r'\b(20[1-2][0-9])\b', chunk)
        
        # Also look for "Issue" numbers and patterns
        issue_patterns = [
            r'Issue\s+(\d+),?\s+20([1-2][0-9])',  # "Issue 1, 2023"
            r'SI Issue\s+(\d+),?\s+20([1-2][0-9])',  # "SI Issue 2, 2024"
            r'20([1-2][0-9])',  # Just year
        ]
        
        latest_year = 2018  # Default fallback year
        issue_number = 1     # Default issue number
        
        # Find the most recent year mentioned
        if years:
            latest_year = max(int(year) for year in years)
        
        # Look for issue numbers
        for pattern in issue_patterns:
            matches = re.findall(pattern, chunk)
            if matches:
                if len(matches[0]) == 2:  # Issue number and year
                    issue_number = int(matches[0][0])
                    year = int('20' + matches[0][1])
                    if year > latest_year:
                        latest_year = year
                break
        
        # Calculate recency score (0-1, higher for more recent)
        current_year = 2025
        year_diff = current_year - latest_year
        
        # Recent content gets higher scores
        if year_diff <= 1:  # 2024-2025
            year_score = 1.0
        elif year_diff <= 2:  # 2023
            year_score = 0.8
        elif year_diff <= 3:  # 2022
            year_score = 0.6
        elif year_diff <= 5:  # 2020-2021
            year_score = 0.4
        else:  # Older than 2020
            year_score = 0.2
        
        # Boost for higher issue numbers (later in year)
        issue_boost = min(issue_number * 0.1, 0.3)
        
        final_score = year_score + issue_boost
        
        logger.debug(f"Recency bias for year {latest_year}, issue {issue_number}: {final_score}")
        return final_score
        
    except Exception as e:
        logger.debug(f"Error calculating recency bias: {str(e)}")
        return 0.3  # Default moderate score



# --- Follow-up and Common Questions Section ---
import random

def generate_followup_questions(last_question):
    if not last_question:
        # Commonly asked questions
        return [
            "What are the best wine pairings for summer dishes?",
            "How should I store my wine collection properly?",
            "What's the difference between Old World and New World wines?"
        ]
    prompt = f"""Based on this wine-related question: \"{last_question}\"\n\nGenerate 5 natural follow-up questions that someone might ask next. Make them specific and relevant to wine knowledge that would likely be covered in wine magazines.\n\nFormat as a simple list:\n1. [question]\n2. [question]\n3. [question]\n4. [question]\n5. [question]"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        lines = answer.split('\n')
        questions = []
        for line in lines:
            if line.strip() and any(line.startswith(f'{i}.') for i in range(1, 6)):
                q = line.split('.', 1)[1].strip()
                questions.append(q)
        # Fallbacks if not enough
        fallback_questions = [
            "Tell me more about wine terminology",
            "What are some wine tasting techniques?",
            "How do wine regions affect flavor?"
        ]
        for fallback in fallback_questions:
            if len(questions) >= 5:
                break
            if fallback not in questions:
                questions.append(fallback)
        return questions[:5]
    except Exception as e:
        return [
            "Tell me more about wine styles",
            "What are some wine tasting tips?",
            "How do I choose the right wine?"
        ]




