import streamlit as st
from agent import get_agent

st.title("Chatbot Agent avec Fonction Temps")
st.write("Posez votre question.")

# Conserver l'historique de conversation 
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Zone de saisie 
user_input = st.text_input("Votre question :", key="input")

# Bouton question / réponse
if st.button("Envoyer") and user_input:
    agent = get_agent()
    
    st.session_state.conversation.append(("Vous", user_input))
    response = agent.invoke(user_input)
    st.session_state.conversation.append(("Agent", response))

# Afficher l'historique de conversation
for speaker, message in st.session_state.conversation:
    st.markdown(f"**{speaker}**: {message}")
