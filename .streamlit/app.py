import streamlit as st
import pandas as pd
import re
import random
from textblob import TextBlob
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# 🔑 Load OpenAI API Key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 📥 Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_chat_twitter_data.csv", low_memory=False)
    df.dropna(subset=["message"], inplace=True)
    return df

df = load_data()

# 🧹 Clean message
def clean_message(text):
    text = re.sub(r"@\w+", "[brand]", str(text))
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\b\d{6,}\b", "[ticket_id]", text)
    return text.strip()

df["message"] = df["message"].apply(clean_message)

# 🤝 Create examples from top 100 customer-agent pairs
@st.cache_data
def generate_examples(df):
    examples = []
    for chat_id, group in df.groupby("chat_id"):
        customers = group[group["from_customer"] == True]
        agents = group[group["from_customer"] == False]
        if not customers.empty and not agents.empty:
            cust_msg = clean_message(customers.iloc[0]["message"])
            agent_msg = clean_message(agents.iloc[0]["message"])
            examples.append({"customer": cust_msg, "agent": agent_msg})
        if len(examples) >= 100:
            break
    return examples

examples = generate_examples(df)

# 🚀 Streamlit UI
st.title("🤖 GPT-4 Customer Service Assistant")
st.markdown("Enter a customer message and see how a tone-aware GPT-4 agent responds.")

user_msg = st.text_area("Customer Message", "My order hasn't arrived yet and it's been 10 days!")

if st.button("Generate Reply"):
    msg_len = len(user_msg)
    sentiment = round(TextBlob(user_msg).sentiment.polarity, 2)
    urgency = any(word in user_msg.lower() for word in ["urgent", "now", "immediately", "asap", "10 days"])
    needs_escalation = sentiment < -0.3 or urgency

    tone = "friendly" if sentiment > 0.3 else "empathetic" if sentiment < -0.2 else "neutral"

    # 🧠 Build prompt
    prompt = f"You are a helpful and professional customer service assistant with a {tone} tone.\nRespond concisely like a real agent on Twitter.\n\n"
    for ex in examples[:5]:
        prompt += f"Customer: {ex['customer']}\nAgent: {ex['agent']}\n\n"
    prompt += f"Customer: {user_msg}\n"
    prompt += f"[Metadata: sentiment={sentiment}, urgency={urgency}, length={msg_len}, escalation={needs_escalation}]\nAgent:"

    # 🔁 Call GPT-4
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )

    generated_reply = response.choices[0].message.content.strip()

    st.markdown("---")
    st.subheader("🤖 GPT-4 Response")
    st.success(generated_reply)

    # 🧪 Optional Evaluation (BLEU & Cosine)
    chat_id = df[df["message"] == user_msg]["chat_id"].values[0] if user_msg in df["message"].values else None
    if chat_id:
        try:
            historical_reply = df[(df["chat_id"] == chat_id) & (df["from_customer"] == False)]["message"].iloc[0]
            historical_reply = clean_message(historical_reply)

            smoothie = SmoothingFunction().method4
            bleu_score = sentence_bleu([historical_reply.split()], generated_reply.split(), weights=(0.5, 0.5), smoothing_function=smoothie)

            vec = TfidfVectorizer().fit_transform([historical_reply, generated_reply])
            cosine_score = cosine_similarity(vec[0:1], vec[1:2])[0][0]

            st.markdown("---")
            st.subheader("📏 Evaluation (optional)")
            st.write("**BLEU Score:**", round(bleu_score, 4))
            st.write("**Cosine Similarity:**", round(cosine_score, 4))
        except:
            st.warning("⚠️ No historical agent reply found for evaluation.")
