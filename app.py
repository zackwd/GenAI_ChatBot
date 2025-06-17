import streamlit as st
import pandas as pd
import re
import random
import json
from textblob import TextBlob
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# 🔑 Load API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 📥 Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_chat_twitter_data.csv", low_memory=False)
    df.dropna(subset=["message"], inplace=True)
    return df

df = load_data()

# 🧹 Clean messages
def clean_message(text):
    text = re.sub(r"@\w+", "[brand]", str(text))
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\b\d{6,}\b", "[ticket_id]", text)
    return text.strip()

df["message"] = df["message"].apply(clean_message)

# 🤝 Create few-shot examples from top 100 customer-agent pairs
@st.cache_data
def generate_examples(df):
    examples = []
    for chat_id, group in df.groupby("chat_id"):
        customers = group[group["from_customer"] == True]
        agents = group[group["from_customer"] == False]
        if not customers.empty and not agents.empty:
            examples.append({
                "customer": clean_message(customers.iloc[0]["message"]),
                "agent": clean_message(agents.iloc[0]["message"])
            })
        if len(examples) >= 100:
            break
    return examples

examples = generate_examples(df)

# 🧠 Session state for chat memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.logs = []

# 🚀 Streamlit UI
st.title("🤖 GenAI Customer Service ChatBot Using GPT3.5-turbo")
user_msg = st.text_area("Customer Message", "My flight was canceled without notice.")

if st.button("Send") and user_msg.strip():
    # 🎯 Metadata extraction
    msg_len = len(user_msg)
    sentiment = round(TextBlob(user_msg).sentiment.polarity, 2)
    urgency = any(word in user_msg.lower() for word in ["urgent", "now", "immediately", "asap", "10 days"])
    needs_escalation = sentiment < -0.3 or urgency
    tone = "friendly" if sentiment > 0.3 else "empathetic" if sentiment < -0.2 else "neutral"

    # 🧠 Compose prompt
    prompt = f"""You are a helpful and professional customer service assistant.
Respond concisely like a real agent on Twitter, in a {tone} tone.\n\n"""
    for ex in examples[:5]:
        prompt += f"Customer: {ex['customer']}\nAgent: {ex['agent']}\n\n"
    prompt += f"Customer: {user_msg}\n"
    prompt += f"[Metadata: sentiment={sentiment}, urgency={urgency}, length={msg_len}, escalation={needs_escalation}]\nAgent:"

    # 🤖 GPT response
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Faster than gpt-4
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )
    reply = response.choices[0].message.content.strip()

    # 🔖 Update session chat
    st.session_state.chat_history.append((user_msg, reply))

    # 🗋 Evaluation metrics
    matched_id = df[df["message"] == user_msg]["chat_id"].values[0] if user_msg in df["message"].values else None
    bleu_score, cosine_score, historical_reply = None, None, None

    if matched_id:
        try:
            historical_reply = df[(df["chat_id"] == matched_id) & (df["from_customer"] == False)]["message"].iloc[0]
            historical_reply = clean_message(historical_reply)
            smoothie = SmoothingFunction().method4
            bleu_score = sentence_bleu([historical_reply.split()], reply.split(), weights=(0.5, 0.5), smoothing_function=smoothie)
            vec = TfidfVectorizer().fit_transform([historical_reply, reply])
            cosine_score = cosine_similarity(vec[0:1], vec[1:2])[0][0]
        except:
            historical_reply = None

    # 📃 Log result
    log = {
        "customer": user_msg,
        "gpt_reply": reply,
        "sentiment": sentiment,
        "urgency": urgency,
        "tone": tone,
        "escalation": needs_escalation,
        "bleu": round(bleu_score, 4) if bleu_score else None,
        "cosine": round(cosine_score, 4) if cosine_score else None,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    st.session_state.logs.append(log)

    # 🔺 Display results
    st.subheader(":robot_face: GPT-4 Response")
    st.success(reply)
    st.markdown(f"**Tone:** {tone}  ")
    st.markdown(f"**Escalation Required:** {'Yes' if needs_escalation else 'No'}")

    if historical_reply:
        st.markdown("---")
        st.subheader("🔹 Compare With Real Agent")
        st.code(historical_reply, language='text')
        st.markdown("**BLEU Score:** {:.4f}".format(bleu_score))
        st.markdown("**Cosine Similarity:** {:.4f}".format(cosine_score))

# 🔗 Show conversation history
if st.checkbox("Show Chat History"):
    st.write(st.session_state.chat_history)

# 💾 Download logs
if st.download_button("💾 Download Logs (JSON)", data=json.dumps(st.session_state.logs, indent=2), file_name="chat_logs.json"):
    st.success("Logs saved!")
