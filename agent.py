import os
import datetime
from dotenv import load_dotenv
from langchain_community.chat_models import AzureChatOpenAI  
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType

load_dotenv()

# Tool pour obtenir la date actuelle
@tool
def obtenir_heure_actuelle(dummy: str = "") -> str:
    """Retourne l'heure actuelle au format YYYY-MM-DD HH:MM:SS."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_agent():
    # Charger les variables d'environnement
    deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    temperature = float(os.environ.get("TEMPERATURE", 0.7))
    max_tokens = int(os.environ.get("MAX_TOKENS", 150))

    # Initialisation du modèle LLM Azure
    llm = AzureChatOpenAI(
        deployment_name=deployment_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Liste des outils disponibles pour l'agent
    tools = [obtenir_heure_actuelle]

    # Initialisation de l'agent avec gestion des erreurs
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True  # Gère les erreurs de parsing
    )
    
    return agent
