import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# 1. Define your FAQs
faqs = {
    "What is artificial intelligence?": "Artificial Intelligence (AI) is the simulation of human intelligence in machines.",
    "What are your working hours?": "We operate 24/7.",
    "How can I reset my password?": "Please click on the 'Forgot Password' link on the login page.",
    "Who created you?": "I am an AI assistant created for the CodeAlpha internship project.",
    "How do I contact support?": "You can reach support via the helpdesk email provided on our website."
}

questions = list(faqs.keys())
answers = list(faqs.values())

# 2. Preprocess and vectorize using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(questions)

# 3. Define the response matching function
def get_bot_response(user_message, history):
    # Vectorize user input
    user_vec = vectorizer.transform([user_message])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(user_vec, tfidf_matrix)
    best_match_idx = np.argmax(similarities)
    
    # Set a threshold so it doesn't answer completely unrelated queries
    if similarities[0, best_match_idx] > 0.3:
        return answers[best_match_idx]
    else:
        return "I'm sorry, I don't understand that question. Could you rephrase?"

# 4. Create the Chat UI
demo = gr.ChatInterface(
    fn=get_bot_response,
    title="FAQ Chatbot",
    description="Ask me frequently asked questions!"
)

if __name__ == "__main__":
    demo.launch()