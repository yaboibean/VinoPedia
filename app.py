
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

# --- Robust session state initialization ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_question' not in st.session_state:
    st.session_state.last_question = ""
if 'question_input_box' not in st.session_state:
    st.session_state.question_input_box = ""
if 'thinking' not in st.session_state:
    st.session_state.thinking = False

# --- Callback for Ask button ---
def handle_ask():
    q = st.session_state.question_input_box
    if q:
        st.session_state.last_question = q
        st.session_state.question_input_box = ""
        st.session_state.thinking = True
        st.session_state.chat_history.append({"role": "user", "content": q})

# --- Main UI: single main box, Streamlit layout ---
st.markdown('<div class="main-box">', unsafe_allow_html=True)
st.markdown('<div class="header-title">Sommelier India\'s Cellar Sage</div>', unsafe_allow_html=True)

main_col1, main_col2 = st.columns([2.2, 1], gap="large")

# --- Chat column ---
with main_col1:
    st.markdown('<div class="main-chat-col">', unsafe_allow_html=True)
    # Render Q&A as magazine-style cards
    if not st.session_state.chat_history:
        st.markdown('<div class="empty-state">Tap into decades of wine wisdom from the Sommelier India Archives</div>', unsafe_allow_html=True)
    else:
        history = st.session_state.chat_history
        i = 0
        while i < len(history):
            if history[i]["role"] == "user":
                q = history[i]["content"]
                a = ""
                if i+1 < len(history) and history[i+1]["role"] == "assistant":
                    a = history[i+1]["content"]
                st.markdown(f'<div class="qa-card"><div class="qa-question">Q: {q}</div>' + (f'<div class="qa-answer">{a}</div>' if a else '') + '</div>', unsafe_allow_html=True)
                i += 2 if a else 1
            else:
                st.markdown(f'<div class="qa-card"><div class="qa-answer">{history[i]["content"]}</div></div>', unsafe_allow_html=True)
                i += 1
    # Input row (widgets, only one instance)
    st.markdown('<div class="input-row">', unsafe_allow_html=True)
    input_col1, input_col2 = st.columns([8,2], gap="small")
    with input_col1:
        st.text_input(
            "",
            placeholder="What would you like to know about wine?",
            key="question_input_box",
            label_visibility="collapsed"
        )
    with input_col2:
        st.button("Ask", key="ask_button", use_container_width=True, on_click=handle_ask)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Follow-up/Recommended Questions column ---
with main_col2:
    st.markdown('<div class="main-followup-col">', unsafe_allow_html=True)
    st.markdown('<div class="followup-title">Follow-up & Common Questions</div>', unsafe_allow_html=True)
    last_q = st.session_state.get('last_question', '')
    import hashlib
    if not last_q:
        recommended_questions = generate_followup_questions("")
        key_prefix = "init"
        def make_recommended_callback(q):
            def cb():
                st.session_state.last_question = q
                st.session_state.question_input_box = ""
                st.session_state.thinking = True
                st.session_state.chat_history.append({"role": "user", "content": q})
            return cb
        for i, q in enumerate(recommended_questions):
            btn_key = f"recommended_btn_{key_prefix}_{i}"
            st.button(q, key=btn_key, help="Click to ask this question", use_container_width=True, on_click=make_recommended_callback(q))
    else:
        followup_questions = generate_followup_questions(last_q)
        key_prefix = hashlib.md5(last_q.encode('utf-8')).hexdigest()[:8]
        def make_followup_callback(q):
            def cb():
                st.session_state.last_question = q
                st.session_state.question_input_box = ""
                st.session_state.thinking = True
                st.session_state.chat_history.append({"role": "user", "content": q})
            return cb
        for i, q in enumerate(followup_questions):
            btn_key = f"followup_btn_{key_prefix}_{i}"
            st.button(q, key=btn_key, help="Click to ask this question", use_container_width=True, on_click=make_followup_callback(q))
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close main-box
    font-size: 1.08em;
    margin-bottom: 8px;
    font-family: 'Lato', 'Arial', sans-serif;
}
.qa-card .qa-answer {
    margin-top: 6px;
    color: #2a0710;
    font-size: 1.01em;
    font-family: 'Lato', 'Arial', sans-serif;
}
.empty-state {
    color: #7a2a3a;
    font-size: 1.18em;
    text-align: left;
    margin-top: 38px;
    font-family: 'Lato', 'Arial', sans-serif;
    font-weight: 600;
    letter-spacing: 0.2px;
    text-shadow: 0 1px 0 #fff, 0 2px 6px #e9e3ea;
    padding-left: 12px;
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
    background: #fff;
    border-radius: 14px;
    box-shadow: 0 1.5px 8px rgba(60,0,20,0.06);
    padding: 16px 22px 16px 22px;
    border: 1.2px solid #e9e3ea;
}
.main-followup-col {
    flex: 1;
    min-width: 340px;
    max-width: 360px;
    margin-left: 32px;
    margin-top: 24px;
    background: #f7f3f6;
    border-radius: 18px;
    box-shadow: 0 2px 16px rgba(90,24,50,0.09);
    padding: 32px 24px 24px 24px;
    border: 1.5px solid #e9e3ea;
}
.followup-title {
    color:#291010; 
    font-size:1.22em; 
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
.followup-btn {
    width: 100%;
    margin-bottom: 12px;
    font-size: 1.04em;
    font-family: 'Lato', 'Arial', sans-serif;
    background: #fff;
    border-radius: 10px;
    border: 1.2px solid #e9e3ea;
    color: #a8325a;
    font-weight: 600;
    box-shadow: 0 1px 4px rgba(60,0,20,0.04);
    transition: background 0.15s, color 0.15s;
}
.followup-btn:hover {
    background: #f7e9f3;
    color: #2a0710;
}
</style>''', unsafe_allow_html=True)







# --- Single main box: all content inside a single .main-box div ---

# --- All content inside a single main-box div ---
main_box_html = """
<div class="main-box">
  <div class="header-title">Sommelier India's Cellar Sage</div>
  <div class="main-content-row">
    <div class="main-chat-col">
      <div style="width:100%;flex:1;display:flex;flex-direction:column;justify-content:flex-start;align-items:center;">
        {chat_content}
      </div>
      <div style="width:100%;">{input_row}</div>
    </div>
    <div class="main-followup-col">
      <div class="followup-title">Follow-up & Common Questions</div>
      {followup_content}
    </div>
  </div>
</div>
"""


# --- Chat content ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if not st.session_state.chat_history:
    chat_content = '<div class="empty-state">Tap into decades of wine wisdom from the Sommelier India Archives</div>'
else:
    # Render Q&A as magazine-style cards, pairing each user question with the next assistant answer
    chat_content = ''
    history = st.session_state.chat_history
    i = 0
    while i < len(history):
        if history[i]["role"] == "user":
            q = history[i]["content"]
            a = ""
            if i+1 < len(history) and history[i+1]["role"] == "assistant":
                a = history[i+1]["content"]
            chat_content += f'<div class="qa-card"><div class="qa-question">Q: {q}</div>'
            if a:
                chat_content += f'<div class="qa-answer">{a}</div>'
            chat_content += '</div>'
            i += 2 if a else 1
        else:
            # If for some reason an assistant message appears first or unpaired, show it alone
            chat_content += f'<div class="qa-card"><div class="qa-answer">{history[i]["content"]}</div></div>'
            i += 1
if 'last_question' not in st.session_state:
    st.session_state['last_question'] = ""
if 'question_input_box' not in st.session_state:
    st.session_state['question_input_box'] = ""
if 'thinking' not in st.session_state:
    st.session_state['thinking'] = False

# --- Input row (Streamlit widgets rendered in HTML layout) ---
if 'question_input_box' not in st.session_state:
    st.session_state.question_input_box = ""

# Callback for Ask button
def handle_ask():
    q = st.session_state.question_input_box
    if q:
        st.session_state.last_question = q
        st.session_state.question_input_box = ""
        st.session_state.thinking = True
        st.session_state.chat_history.append({"role": "user", "content": q})

input_placeholder = st.empty()
with input_placeholder.container():
    st.markdown('<div class="input-row">', unsafe_allow_html=True)
    input_col1, input_col2 = st.columns([8,2], gap="small")
    with input_col1:
        question = st.text_input(
            "",
            placeholder="What would you like to know about wine?",
            key="question_input_box",
            label_visibility="collapsed",
            value=st.session_state.question_input_box
        )
    with input_col2:
        ask_button = st.button("Ask", key="ask_button", use_container_width=True, on_click=handle_ask)
    st.markdown('</div>', unsafe_allow_html=True)




# --- Render all content in a single main box using Streamlit columns for layout ---
st.markdown('<div class="main-box">', unsafe_allow_html=True)

# Header
st.markdown('<div class="header-title">Sommelier India\'s Cellar Sage</div>', unsafe_allow_html=True)

# Main content row: use Streamlit columns for chat and follow-up
main_col1, main_col2 = st.columns([2.2, 1], gap="large")

with main_col1:
    st.markdown('<div class="main-chat-col">', unsafe_allow_html=True)
    st.markdown(f'<div style="width:100%;flex:1;display:flex;flex-direction:column;justify-content:flex-start;align-items:center;">{chat_content}</div>', unsafe_allow_html=True)
    # Input row (widgets)
    st.markdown('<div style="width:100%;">', unsafe_allow_html=True)
    input_col1, input_col2 = st.columns([8,2], gap="small")
    with input_col1:
        question = st.text_input(
            "",
            placeholder="What would you like to know about wine?",
            key="question_input_box",
            label_visibility="collapsed",
            value=st.session_state.question_input_box
        )
    with input_col2:
        ask_button = st.button("Ask", key="ask_button", use_container_width=True, on_click=handle_ask)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with main_col2:
    st.markdown('<div class="main-followup-col">', unsafe_allow_html=True)
    st.markdown('<div class="followup-title">Follow-up & Common Questions</div>', unsafe_allow_html=True)
    last_q = st.session_state.get('last_question', '')
    if not last_q:
        recommended_questions = generate_followup_questions("")
        import hashlib
        key_prefix = "init"
        def make_recommended_callback(q):
            def cb():
                st.session_state.last_question = q
                st.session_state.question_input_box = ""
                st.session_state.thinking = True
                st.session_state.chat_history.append({"role": "user", "content": q})
            return cb
        for i, q in enumerate(recommended_questions):
            btn_key = f"recommended_btn_{key_prefix}_{i}"
            st.button(q, key=btn_key, help="Click to ask this question", use_container_width=True, on_click=make_recommended_callback(q))
    else:
        followup_questions = generate_followup_questions(last_q)
        import hashlib
        key_prefix = hashlib.md5(last_q.encode('utf-8')).hexdigest()[:8]
        def make_followup_callback(q):
            def cb():
                st.session_state.last_question = q
                st.session_state.question_input_box = ""
                st.session_state.thinking = True
                st.session_state.chat_history.append({"role": "user", "content": q})
            return cb
        for i, q in enumerate(followup_questions):
            btn_key = f"followup_btn_{key_prefix}_{i}"
            st.button(q, key=btn_key, help="Click to ask this question", use_container_width=True, on_click=make_followup_callback(q))
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close main-box

# --- Handle question submission and response ---
if st.session_state.get("thinking", False) and st.session_state.get("last_question", ""):
    question = st.session_state.last_question
    with st.spinner("Thinking..."):
        try:
            query_embedding = embed_query(question)
            # Only fetch the single most relevant chunk for speed
            D, I = index.search(np.array([query_embedding]), k=1)
            relevant_chunks = []
            for idx in I[0]:
                if idx < len(chunks):
                    chunk = chunks[idx]
                    truncated_chunk = chunk[:400] + "..." if len(chunk) > 400 else chunk
                    relevant_chunks.append(truncated_chunk)
            relevant = "\n\n".join(relevant_chunks)
            # Short, direct prompt for speed
            prompt = f"Answer the wine question using this magazine context:\n{relevant}\n\nQ: {question}\n- Be concise, expert, and cite sources."
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            answer = response.choices[0].message.content
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            error_message = f"Error generating answer: {str(e)}\n\nTraceback:\n{tb}\n\nOPENAI_API_KEY present: {'Yes' if openai_api_key else 'No'}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})
    st.session_state.thinking = False


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


