import os
import datetime
from dotenv import load_dotenv
from langchain_community.chat_models import AzureChatOpenAI  
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType

load_dotenv()

# tool pour obtenir la date
@tool
def obtenir_heure_actuelle(dummy: str = "") -> str:
    """Retourne l'heure actuelle au format YYYY-MM-DD HH:MM:SS."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_agent():
    # Charger les variables d'environnement
    deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    temperature = float(os.environ.get("TEMPERATURE", 0.7))
    max_tokens = int(os.environ.get("MAX_TOKENS", 150))
    
    # LLM
    llm = AzureChatOpenAI(
        deployment_name=deployment_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    # Lancer l'agent avec le tool
    tools = [obtenir_heure_actuelle]
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return agent
