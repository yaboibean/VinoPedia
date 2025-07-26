
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

# --- Modern Main Card Layout ---
st.markdown('''
<div class="wine-main-card">
  <div class="wine-title">Sommelier India's Cellar Sage</div>
  <div class="wine-desc">Tap into decades of wine wisdom from the Sommelier India Archives</div>
  <div class="wine-content-row">
    <div class="wine-chat-col">
      <div class="wine-chat-area">
''', unsafe_allow_html=True)

# --- Chat area ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if not st.session_state.chat_history:
    st.markdown('<div class="wine-empty">No conversation yet. Ask your first wine question!</div>', unsafe_allow_html=True)
else:
    for i, msg in enumerate(st.session_state.chat_history):
        is_user = msg['role'] == 'user'
        bubble_class = 'wine-bubble-user' if is_user else 'wine-bubble-assistant'
        avatar = '🍷' if is_user else '🤖'
        st.markdown(f'''
        <div class="wine-chat-row">
          <span class="wine-avatar">{avatar}</span>
          <span class="{bubble_class}">{msg["content"]}</span>

    key="ask_button",
    use_container_width=True,
    help="Submit your wine question"
)
st.markdown('''</div></div></div>
''', unsafe_allow_html=True)

# --- Follow-up panel (right column) ---
st.markdown('''    <div class="wine-followup-col">
      <div class="wine-followup-panel">
        <div class="wine-followup-title">Popular Questions</div>
''', unsafe_allow_html=True)
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
        st.markdown(f'''<div class="wine-followup-btn-row">''', unsafe_allow_html=True)
        if st.button(q, key=btn_key, help="Click to ask this question", disabled=thinking):
            if not thinking:
                st.session_state.last_question = q
                st.session_state.chat_history.append({"role": "user", "content": q})
                st.session_state.question_input_box = ""
                st.session_state.thinking = True
                st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)










st.markdown('''</div></div></div>




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




