#  GenAI Customer Service ChatBot

A domain-specific, GPT-4-powered customer service assistant built for real-time query resolution, escalation detection, and human-like support automation.

## Overview

This project implements a Generative AI ChatBot that simulates a customer service agent using structured prompt engineering, sentiment analysis, and metadata-driven response control. It handles FAQ queries, troubleshooting, escalation detection, and urgent response escalation — all via a natural, conversational interface.

## Key Features

- **LLM-Driven Responses**: Uses OpenAI GPT-4 to generate human-like replies based on historical support data.
- **Prompt Engineering**: Contextual prompt templates based on detected intent and urgency.
- **Escalation Detection**: Automatically flags frustrated or urgent users via sentiment and metadata.
- **BLEU & ROUGE Evaluation**: Compares generated responses against real historical answers for accuracy benchmarking.
- **Streamlit Interface**: Interactive front-end for real-time demo and testing.
- **Human-in-the-Loop Support**: Routes complex or flagged queries to human agents when needed.

## 📚 Datasets

- [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- [Customer Service Chat Data - 30K Rows](https://www.kaggle.com/datasets/aimack/customer-service-chat-data-30k-rows)

Both datasets were cleaned, tokenized, and used to extract intents, design prompt structures, and test the ChatBot's response accuracy.

##  Tech Stack

| Component         | Tools & Libraries                                |
|------------------|---------------------------------------------------|
| LLM & NLP         | OpenAI GPT-4 API, Transformers, NLTK              |
| Data Processing   | Pandas, NumPy                                     |
| Evaluation        | BLEU, ROUGE, cosine similarity                    |
| Front-End         | Streamlit                                         |
| Deployment        | Streamlit Cloud, GitHub Actions                   |
| Ethics & Safety   | PII redaction, bias checks, escalation safeguards |

##  Performance Highlights

- Response accuracy** improved via iterative prompt tuning.
- Latency** reduced from ~10s to ~3–5s per query.
- Escalation detection accuracy**: ~82% on test set.
- BLEU similarity score**: Avg. 0.62 against human responses.

## Future Work

- Fine-tune escalation module using transformer-based classifiers.
- Add multilingual support and user sentiment memory.
- Integrate with CRM tools like Zendesk or Salesforce.

## Demo

👉 [Click here to try it live (Streamlit App)](https://genaichatbot-m9fqf6d8fifzczmnbgo5vr.streamlit.app/)  
*


Feel free to fork, extend, or contribute!

