
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

# --- Streamlit UI ---
st.markdown('<div class="header"><h1>Sommelier India\'s Cellar Sage</h1></div>', unsafe_allow_html=True)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_question' not in st.session_state:
    st.session_state.last_question = ''



# --- Two-column layout: Chat (center), Questions (right) ---
col1, col2, col3 = st.columns([0.2, 1, 0.7], gap="large")

# --- Chat Section (Center Panel) ---
with col2:
    chat_container = st.container()
    if not st.session_state.chat_history:
        chat_container.markdown('<div class="empty-state">Tap into decades of wine wisdom from the Sommelier India Archives</div>', unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                chat_container.markdown(f'<div style="background:#291010;color:#ac9c9c;padding:12px 18px;border-radius:18px 18px 5px 18px;margin-left:auto;margin-right:0;max-width:80%;word-wrap:break-word;box-shadow:0 2px 8px rgba(90,24,50,0.09);border:1px solid #7a1c3a;margin-bottom:10px;">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                chat_container.markdown(f'<div style="background:#fff;color:#2d0a18;padding:15px 20px;border-radius:18px 18px 18px 5px;border:1px solid #b7aeb4;max-width:90%;word-wrap:break-word;line-height:1.6;box-shadow:0 2px 10px rgba(90,24,50,0.04);margin-bottom:20px;">{msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)

    # --- Thinking indicator state ---
    if 'thinking' not in st.session_state:
        st.session_state.thinking = False

    with st.form(key="question_form", clear_on_submit=True):
        question = st.text_input("", placeholder="What would you like to know about wine?", key="question_input")
        ask_button = st.form_submit_button("Ask")

    # Show thinking indicator if active
    if st.session_state.thinking:
        st.markdown('<div style="margin:10px 0 20px 0; color:#a8325a; font-size:1.1em; font-weight:600; display:flex; align-items:center;"><span class="spinner" style="display:inline-block;width:22px;height:22px;border:3px solid #e9ecef;border-top:3px solid #a8325a;border-radius:50%;margin-right:10px;animation:spin 1s linear infinite;"></span>Thinking...</div>', unsafe_allow_html=True)

    # Add spinner CSS
    st.markdown('''<style>@keyframes spin {0%{transform:rotate(0deg);}100%{transform:rotate(360deg);}}</style>''', unsafe_allow_html=True)

    if not last_question:
        # Commonly asked questions
        return [
            "What are the best wine pairings for summer dishes?",
            "How should I store my wine collection properly?",
            "What's the difference between Old World and New World wines?"
        ]
    prompt = f"Give me 3 followup questions based on this: {last_question}\nFormat as a simple list:\n1. [question]\n2. [question]\n3. [question]"
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        lines = answer.split('\n')
        questions = []
        for line in lines:
            if line.strip() and any(line.startswith(f'{i}.') for i in range(1, 4)):
                q = line.split('.', 1)[1].strip()
                questions.append(q)
        # Fallbacks if not enough
        fallback_questions = [
            "Tell me more about wine terminology",
            "What are some wine tasting techniques?",
            "How do wine regions affect flavor?"
        ]
        for fallback in fallback_questions:
            if len(questions) >= 3:
                break
            if fallback not in questions:
                questions.append(fallback)
        return questions[:3]
    except Exception as e:
        return [
            "Tell me more about wine styles",
            "What are some wine tasting tips?",
            "How do I choose the right wine?"
        ]
        import re
        
        # Look for years 2020+ (most relevant)
        recent_years = re.findall(r'\b(202[0-9])\b', chunk)
        if recent_years:
            latest_year = max(int(year) for year in recent_years)
            if latest_year >= 2024:
                return 1.0
            elif latest_year >= 2022:
                return 0.8
            else:
                return 0.6
        
        # Quick fallback - look for any year pattern
        if '202' in chunk:  # Any 2020s mention
            return 0.7
        elif '201' in chunk:  # Any 2010s mention
            return 0.4
        else:
            return 0.3  # Default
            
    except Exception:
        return 0.3  # Default on any error

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


# --- Follow-up and Common Questions (Right Panel) ---
with col3:
    st.markdown('<div class="followup-section" style="margin-top:32px;">', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#291010; font-size:1.18em; font-weight:700; letter-spacing:0.5px; text-shadow:0 1px 0 #fff, 0 2px 6px #e9e3ea;">Follow-up & Common Questions</h3>', unsafe_allow_html=True)
    followup_questions = generate_followup_questions(st.session_state.get('last_question', ''))
    for i, q in enumerate(followup_questions):
        btn_style = (
            "background:linear-gradient(90deg,#fff 60%,#c9c7c7 100%);color:#291010;"
            "border:1.5px solid #291010;border-radius:32px;padding:14px 24px;font-size:17px;cursor:pointer;margin-bottom:8px;font-weight:500;outline:none;width:100%;text-align:left;"
            "transition:background 0.2s,color 0.2s,box-shadow 0.2s,border-color 0.2s,transform 0.15s;"
        )
        hover_style = (
            f"<style>div[data-testid='stButton'] button[data-testid='followup_btn_{i}']:hover {{"
            "background:linear-gradient(90deg,#a8325a 10%,#291010 90%) !important;"
            "color:#fff !important;border-color:#a8325a !important;box-shadow:0 3px 10px rgba(168,50,90,0.09);"
            "transform:translateY(-1px) scale(1.01);}}</style>"
        )
        st.markdown(hover_style, unsafe_allow_html=True)
        if st.button(q, key=f"followup_btn_{i}", help="Click to ask this question"):
            if not st.session_state.get('thinking', False):
                st.session_state.last_question = q
                st.session_state.chat_history.append({"role": "user", "content": q})
                st.session_state.thinking = True
                st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)

