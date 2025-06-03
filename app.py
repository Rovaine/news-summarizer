import streamlit as st
from newspaper import Article
from textblob import TextBlob
from collections import Counter
from urllib.parse import urlparse
import re
import nltk
import os
import google.generativeai as genai

# Page config must come first
st.set_page_config(page_title="News Summarizer", page_icon="🧠")

# Download NLTK resources
#nltk.download('stopwords')
from nltk.corpus import stopwords

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY"))
@st.cache_resource
def load_model():
    return genai.GenerativeModel('gemini-1.5-flash')

model = load_model()

# Summarize using Gemini
def summarize_with_gemini(text, language="English"):
    prompt = f"Summarize the following news article in 5-6 bullet points in {language}, preserving key facts:{text}"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error generating summary: {e}"

# Keyword extraction
def extract_keywords(text, num=5):
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    common_words = [word for word in words if word not in stopwords.words('english')]
    most_common = Counter(common_words).most_common(num)
    return [word for word, freq in most_common]

# Fact check using Gemini
def check_facts(text):
    prompt = f"Review the following news article and point out any factual inaccuracies, bias, or exaggerations: {text}"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error during fact-checking: {e}"

# Explain Like I'm 5
def explain_eli5(text):
    prompt = f"Explain the following news article in a way a 5-year-old would understand: {text}"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error generating ELI5 explanation: {e}"

# App layout
st.title("🧠 AI-Powered News Summarizer")
st.markdown("Paste any news article URL below to get a summary, sentiment, keywords, fact-checks and more!")

url = st.text_input("🔗 Enter News Article URL")
lang = st.selectbox("🌐 Choose summary language", ["English", "Hindi", "French"])
eli5_mode = st.toggle("🧒 Explain like I'm 5")

if st.button("Summarize"):
    if not url.strip():
        st.warning("Please enter a valid URL.")
    else:
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text
            title = article.title
            domain = urlparse(url).netloc

            st.subheader("📰 Title")
            st.write(title)
            st.markdown(f"🗞 **Source:** {domain}")

            # NSFW content warning
            banned_words = ['violence', 'kill', 'murder', 'assault']
            if any(word in text.lower() for word in banned_words):
                st.warning("⚠️ This article may contain sensitive content.")

            with st.expander("📜 Click to view full article"):
                st.write(text)

            # Summary
            st.subheader("📝 Summary")
            summary = summarize_with_gemini(text, language=lang)
            st.success(summary)

            # ELI5 mode
            if eli5_mode:
                st.subheader("🧒 ELI5 Summary")
                eli5 = explain_eli5(text)
                st.info(eli5)

            # Sentiment
            st.subheader("📊 Sentiment")
            sentiment = TextBlob(text).sentiment.polarity
            if sentiment > 0.1:
                st.success("Positive 😊")
            elif sentiment < -0.1:
                st.error("Negative 😞")
            else:
                st.info("Neutral 😐")

            # Keywords
            st.subheader("🔑 Top Keywords")
            keywords = extract_keywords(text)
            st.write(", ".join(keywords))

            # Fact-checking
            st.subheader("🔍 Fact Check")
            facts = check_facts(text)
            st.warning(facts)

            # Save to history
            # Download
            st.download_button("📥 Download Summary", summary, file_name="summary.txt")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
