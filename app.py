import re
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Ticket Triage", page_icon="🎫", layout="wide")
st.title("AI Ticket Triage System")
st.caption("Educational demo using a synthetic ticket dataset.")

DATA_PATH = Path("data/tickets.csv")

team_map = {
    "Billing": "Finance Support",
    "Login": "Account Support",
    "Technical Issue": "Technical Support",
    "Refund": "Finance Support",
    "Feature Request": "Product Team"
}

knowledge_base = [
    "Billing: Review invoices in Account Settings, Billing, and Invoices. Contact billing support about incorrect charges.",
    "Login: Use Forgot Password on the sign-in page. Check your spam folder and confirm you used the correct account email.",
    "Technical Issue: Refresh the page, update the app, and try again. Include steps to reproduce an error.",
    "Refund: Request a refund from Account Settings and Orders. Include your order ID.",
    "Feature Request: Describe the problem you want solved. The product team reviews suggestions."
]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def predict_priority(message):
    text = clean_text(message)
    if "urgent outage" in text or "security breach" in text:
        return "Urgent"
    if "affecting several users" in text or "charged twice" in text:
        return "High"
    if "not working" in text or "failed" in text or "incorrect" in text:
        return "Medium"
    return "Low"

def analyze_sentiment(message):
    text = clean_text(message)
    positive = ["thanks", "thank you", "great", "helpful", "excellent", "happy"]
    negative = ["angry", "terrible", "broken", "frustrated", "disappointed", "worst"]
    pos_score = sum(word in text for word in positive)
    neg_score = sum(word in text for word in negative)

    if pos_score > neg_score:
        return "Positive"
    if neg_score > pos_score:
        return "Negative"
    return "Neutral"

def search_knowledge_base(query, top_k=1):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(knowledge_base)
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).flatten()
    indices = scores.argsort()[::-1][:top_k]
    return [knowledge_base[i] for i in indices if scores[i] > 0]

def generate_response(message, category, priority):
    matches = search_knowledge_base(message)

    if matches:
        answer = "Based on our support guide: " + matches[0]
    else:
        answer = f"Our {team_map[category]} team will review your request."

    if priority == "Urgent":
        answer += " This ticket is marked urgent for prompt review."

    return "Hello, thank you for contacting support. " + answer

tickets = pd.read_csv(DATA_PATH)

category_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])
category_model.fit(tickets["customer_message"], tickets["category"])

def classify_category(message):
    return category_model.predict([message])[0]

st.sidebar.header("Ticket filters")
categories = st.sidebar.multiselect(
    "Category",
    sorted(tickets["category"].unique()),
    default=sorted(tickets["category"].unique())
)
priorities = st.sidebar.multiselect(
    "Priority",
    ["Urgent", "High", "Medium", "Low"],
    default=["Urgent", "High", "Medium", "Low"]
)

filtered = tickets[
    tickets["category"].isin(categories)
    & tickets["priority"].isin(priorities)
]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Tickets", len(filtered))
col2.metric("Open tickets", int((filtered["status"] == "Open").sum()))
col3.metric("Urgent tickets", int((filtered["priority"] == "Urgent").sum()))
col4.metric("Negative sentiment", int((filtered["sentiment"] == "Negative").sum()))

dashboard_tab, submit_tab, search_tab = st.tabs(
    ["Ticket dashboard", "Submit a ticket", "Knowledge search"]
)

with dashboard_tab:
    st.subheader("Ticket queue")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    left.bar_chart(filtered["category"].value_counts())
    right.bar_chart(filtered["priority"].value_counts())

with submit_tab:
    st.subheader("Submit a support ticket")
    with st.form("ticket_form"):
        customer_name = st.text_input("Customer name")
        customer_email = st.text_input("Customer email")
        message = st.text_area("Customer message")
        submitted = st.form_submit_button("Analyze and submit")

    if submitted:
        if not message.strip():
            st.error("Please enter a customer message.")
        else:
            category = classify_category(message)
            priority = predict_priority(message)
            sentiment = analyze_sentiment(message)
            team = team_map[category]

            new_ticket = {
                "ticket_id": f"T{1001 + len(tickets)}",
                "customer_message": message,
                "date": pd.Timestamp.now().date().isoformat(),
                "customer_name": customer_name or "Guest",
                "customer_email": customer_email or "Not provided",
                "status": "Open",
                "category": category,
                "priority": priority,
                "sentiment": sentiment,
                "assigned_team": team
            }

            updated = pd.concat(
                [tickets, pd.DataFrame([new_ticket])],
                ignore_index=True
            )
            updated.to_csv(DATA_PATH, index=False)

            st.success(f"Ticket created and routed to {team}.")
            st.write(new_ticket)
            st.markdown("**Suggested response**")
            st.write(generate_response(message, category, priority))

with search_tab:
    st.subheader("Search support guidance")
    query = st.text_input("Enter a question")
    if query:
        matches = search_knowledge_base(query, top_k=3)
        if matches:
            for result in matches:
                st.write(result)
        else:
            st.info("No matching knowledge article was found.")
