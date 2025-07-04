import streamlit as st

# 🚩 Ensure set_page_config is first
st.set_page_config(page_title="GenAI Customer Service ChatBot GPT-4o", page_icon="🤖", layout="wide")

import pandas as pd
import re
import json
from textblob import TextBlob
from openai import OpenAI
from datetime import datetime
import numpy as np

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_chat_twitter_data.csv", low_memory=False)
    df.dropna(subset=["message"], inplace=True)
    return df

df = load_data()

def clean_message(text):
    text = re.sub(r"@\\w+", "[brand]", str(text))
    text = re.sub(r"#\\w+", "", text)
    text = re.sub(r"\\b\\d{6,}\\b", "[ticket_id]", text)
    text = re.sub(r"http\\S+", "[link]", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()

df["message"] = df["message"].apply(clean_message)

def detect_urgency(text):
    high = ["urgent", "emergency", "asap", "immediately", "now", "critical", "broken", "not working"]
    med = ["soon", "quickly", "fast", "help", "issue", "problem"]
    tl = text.lower()
    if sum(w in tl for w in high) >= 2 or any(w in tl for w in ["urgent", "emergency", "critical"]):
        return "high"
    elif sum(w in tl for w in high) >= 1 or sum(w in tl for w in med) >= 2:
        return "medium"
    else:
        return "low"

def detect_complexity(text):
    tech = ["API", "server", "database", "configuration", "integration", "authentication", "SSL", "DNS"]
    comp = ["multiple", "several", "various", "different", "also", "additionally", "furthermore"]
    tl = text.lower()
    tc = sum(t.lower() in tl for t in tech)
    cc = sum(c in tl for c in comp)
    wc = len(text.split())
    if tc >= 2 or cc >= 2 or wc > 50:
        return "high"
    elif tc >= 1 or cc >= 1 or wc > 25:
        return "medium"
    else:
        return "low"

def determine_tone(sentiment, urgency, complexity):
    if sentiment <= -0.5:
        return "empathetic"
    elif sentiment >= 0.3:
        return "friendly"
    elif urgency == "high" or complexity == "high":
        return "professional"
    else:
        return "neutral"

def create_prompt(user_msg, sentiment, urgency, complexity, tone):
    return (
        f"You are a professional customer service assistant. Tone: {tone}. "
        f"Sentiment: {sentiment}. Urgency: {urgency}. Complexity: {complexity}. "
        f"Respond concisely (1-2 sentences) to the following customer message: {user_msg}"
    )

st.title("🤖 GenAI Customer Service ChatBot (GPT-4o)")

# Initialize chat history cleanly
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
with st.container():
    for i, entry in enumerate(st.session_state.history):
        st.text_area(f"Customer Message {i+1}", entry["user"], disabled=True, height=80)
        st.text_area(f"Assistant Response {i+1}", entry["reply"], disabled=True, height=80)
        st.caption(f"Sentiment: {entry['sentiment']}, Urgency: {entry['urgency']}, Complexity: {entry['complexity']}, Tone: {entry['tone']}")

# Always provide a fresh empty input box each rerun
user_msg = st.text_area("Next Customer Message:", value="", placeholder="Enter your next message here...", height=100, key=f"input_{len(st.session_state.history)}")

if st.button("Send") and user_msg.strip():
    sentiment = round(TextBlob(user_msg).sentiment.polarity, 3)
    urgency = detect_urgency(user_msg)
    complexity = detect_complexity(user_msg)
    tone = determine_tone(sentiment, urgency, complexity)
    prompt = create_prompt(user_msg, sentiment, urgency, complexity, tone)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        reply = response.choices[0].message.content.strip()
        st.session_state.history.append({
            "user": user_msg,
            "reply": reply,
            "sentiment": sentiment,
            "urgency": urgency,
            "complexity": complexity,
            "tone": tone
        })
        st.rerun()
    except Exception as e:
        st.error(f"Error generating response: {e}")

st.markdown("---")
st.markdown("*GenAI Customer Service ChatBot - Now upgraded with GPT-4o for stable, efficient, clean, scrollable multi-turn structured conversation with input cleared reliably each turn.*")
