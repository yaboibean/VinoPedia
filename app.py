import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import faiss
import numpy as np
import pickle
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Streamlit does not use Flask app objects

# Load environment variables from .env
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

# Streamlit UI
st.title("🍷 Wine Magazine Assistant")
st.write("Ask questions about wine using magazine content!")

if index is None or not chunks:
    st.error("System not initialized. Please run extract_and_index.py first.")
    st.stop()

question = st.text_input("Enter your wine question:")
ask_button = st.button("Ask")

if ask_button and question:
    st.info(f"❓ Received question: {question}")
    query_embedding = embed_query(question)
    D, I = index.search(np.array([query_embedding]), k=3)
    relevant_chunks = []
    for idx in I[0]:
        if idx < len(chunks):
            chunk = chunks[idx]
            truncated_chunk = chunk[:800] + "..." if len(chunk) > 800 else chunk
            relevant_chunks.append(truncated_chunk)
    relevant = "\n\n".join(relevant_chunks)
    st.write("### Relevant magazine content:")
    st.write(relevant)
    prompt = f"""You are a helpful wine expert assistant answering questions based on wine magazine content.

Here is relevant context from the wine magazines:
{relevant}

Question: {question}

Instructions:
- Keep responses concise but informative (2-4 paragraphs max)
- Use bullet points for key information
- Include specific wine terminology and expert insights
- Quote directly from magazines when relevant (use quotation marks)
- If magazines don't contain specific info, state this briefly
- End with source citations: "Sommelier India, <issue number>, <year>"

Be direct and focused - provide depth without being wordy."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.3
        )
        answer = response.choices[0].message.content
        st.success("Wine Expert Answer:")
        st.write(answer)
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")


# Follow-up questions UI
st.write("---")
st.subheader("Get follow-up wine questions")
previous_question = st.text_input("Enter your previous wine question for follow-ups:")
followup_button = st.button("Get Follow-up Questions")

if followup_button and previous_question:
    prompt = f"""Based on this wine-related question: "{previous_question}"

Generate 5 natural follow-up questions that someone might ask next. Make them specific and relevant to wine knowledge that would likely be covered in wine magazines.

Format as a simple list:
1. [question]
2. [question]
3. [question]
4. [question]
5. [question]"""
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
                if len(questions) >= 3:
                    break
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
        st.success("Follow-up Questions:")
        for q in questions[:3]:
            st.write(f"- {q}")
    except Exception as e:
        st.error(f"Error generating follow-up questions: {str(e)}")

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

# Remove Flask app.run for Streamlit

