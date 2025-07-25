
import streamlit as st
from openai import OpenAI
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
with st.form(key="question_form", clear_on_submit=True):
    question = st.text_input("", placeholder="What would you like to know about wine?", key="question_input")
    ask_button = st.form_submit_button("Ask")

if ask_button and question:
    st.session_state.last_question = question
    st.session_state.chat_history.append({"role": "user", "content": question})
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
    try:
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

def calculate_recency_bias_fast(chunk):
    """Fast recency bias calculation - optimized for speed"""
    try:
        # Quick regex search for recent years
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

@app.route("/ask", methods=["POST"])
def ask():
    try:
        # Check if system is properly initialized
        if index is None or not chunks:
            logger.error("❌ System not properly initialized")
            return jsonify({"error": "System not initialized. Please run extract_and_index.py first."}), 500

        data = request.get_json()
        query = data.get("question", "")
        if not query:
            return jsonify({"error": "No question provided"}), 400

        logger.info(f"❓ Received question: {query}")

        query_embedding = embed_query(query)
        
        D, I = index.search(np.array([query_embedding]), k=3)  # Reduced to 3 for maximum speed
        
        # Get top 3 chunks directly - skip recency bias for speed
        relevant_chunks = []
        for idx in I[0]:
            if idx < len(chunks):
                chunk = chunks[idx]
                # Truncate chunks to reduce context size and speed up GPT
                truncated_chunk = chunk[:800] + "..." if len(chunk) > 800 else chunk
                relevant_chunks.append(truncated_chunk)
        
        relevant = "\n\n".join(relevant_chunks)
        logger.info(f"📝 Using {len(relevant_chunks)} relevant chunks, total length: {len(relevant)} chars")
        
        # Log relevant content preview
        if relevant:
            preview = relevant[:200].replace('\n', ' ')
            logger.info(f"📋 Relevant content preview: {preview}...")
        else:
            logger.warning("⚠️  No relevant content found!")

        prompt = f"""You are a helpful wine expert assistant answering questions based on wine magazine content.

Here is relevant context from the wine magazines:
{relevant}

Question: {query}

Instructions:
- Keep responses concise but informative (2-4 paragraphs max)
- Use bullet points for key information
- Include specific wine terminology and expert insights
- Quote directly from magazines when relevant (use quotation marks)
- If magazines don't contain specific info, state this briefly
- End with source citations: "Sommelier India, <issue number>, <year>"

Be direct and focused - provide depth without being wordy."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,  # Further reduced for maximum speed
            temperature=0.3
        )

        answer = response.choices[0].message.content
        logger.info(f"✅ Generated answer: {answer[:100]}...")
        
        # Preserve formatting by not modifying the response
        return jsonify({"answer": answer})

    except Exception as e:
        logger.error(f"❌ Error in ask endpoint: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/")
def serve_html():
    return send_from_directory(".", "index.html")

@app.route("/debug")
def debug_info():
    """Debug endpoint to check system status"""
    info = {
        "index_loaded": index is not None,
        "chunks_loaded": len(chunks) if chunks else 0,
        "index_vectors": index.ntotal if index else 0,
        "files_exist": {
            "index": os.path.exists("magazine_index.faiss"),
            "chunks": os.path.exists("magazine_chunks.pkl")
        }
    }
    if chunks:
        info["sample_chunk"] = chunks[0][:200] + "..." if len(chunks[0]) > 200 else chunks[0]
    return jsonify(info)

@app.route("/followup", methods=["POST"])
def get_followup_questions():
    try:
        data = request.get_json()
        previous_question = data.get("previous_question", "")
        
        if not previous_question:
            # Return Popular Questions without validation for speed
            return jsonify({
                "title": "Popular Questions",
                "questions": [
                    "What are the best wine pairings for summer dishes?",
                    "How should I store my wine collection properly?",
                    "What's the difference between Old World and New World wines?"
                ]
            })
        
        # Generate follow-up questions based on previous query
        prompt = f"""Based on this wine-related question: "{previous_question}"

Generate 5 natural follow-up questions that someone might ask next. Make them specific and relevant to wine knowledge that would likely be covered in wine magazines.

Format as a simple list:
1. [question]
2. [question]
3. [question]
4. [question]
5. [question]"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,  # Reduced for speed
            temperature=0.7
        )

        answer = response.choices[0].message.content.strip()
        
        # Parse the response to extract questions
        lines = answer.split('\n')
        questions = []
        for line in lines:
            if line.strip() and any(line.startswith(f'{i}.') for i in range(1, 6)):
                question = line.split('.', 1)[1].strip()
                questions.append(question)
                if len(questions) >= 3:  # Stop at 3 for speed
                    break
        
        # If we don't have enough questions, add fallbacks
        if len(questions) < 3:
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
        
        return jsonify({
            "title": "Follow-up Questions",
            "questions": questions[:3]  # Ensure max 3 questions
        })

    except Exception as e:
        logger.error(f"❌ Error generating follow-up questions: {str(e)}")
        return jsonify({
            "title": "Suggested Questions",
            "questions": [
                "Tell me more about wine styles",
                "What are some wine tasting tips?",
                "How do I choose the right wine?"
            ]
        })

def validate_question_has_good_answer(question):
    """Check if a question has relevant content in the magazine database"""
    try:
        if index is None or not chunks:
            return False
        
        # Generate embedding for the question
        query_embedding = embed_query(question)
        
        # Search for relevant content
        D, I = index.search(np.array([query_embedding]), k=3)
        
        # Check if we have good matches (distance threshold)
        # Lower distance = better match
        if len(D[0]) > 0 and D[0][0] < 1.2:  # Adjust threshold as needed
            return True
        
        return False
        
    except Exception as e:
        logger.warning(f"⚠️  Error validating question '{question}': {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("🍷 Wine Magazine Assistant starting...")
    if index is None or not chunks:
        logger.error("❌ System not properly initialized. Please run extract_and_index.py first!")
    else:
        logger.info("✅ System ready!")
    app.run(host="0.0.0.0", port=8001)
    logger.info("🌐 Running on http://0.0.0.0:8001")

