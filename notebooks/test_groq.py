# notebooks/test_groq.py
#  Test de l'API Groq avec Llama 3

import os
from dotenv import load_dotenv
from groq import Groq

# Charger la cle depuis .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERREUR : GROQ_API_KEY non  trouvée dans .env")
    exit()

# Créer le client Groq
client = Groq(api_key=api_key)

# Premier appel: question simple
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role" : "system",
            "content" : "Tu es un assistant medical senegalais. "
                        "Reponds en francais simple."
                        "Maximum 3 phrases."
        },
        {
            "role" : "user",
            "content" : "Quels sont les symptomes du paludisme ?"
        }
    ],
    max_tokens=200,
    temperature=0.3
)

# Afficher la réponse 
print("=== Réponse de Llama 3 ===")
print(response.choices[0].message.content)
print(f"\nTokens utilisés : {response.usage.total_tokens}")


# ============== Test le format SenSante ==================
response2= client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content": """Tu es un assitant médical sénégalais.
                          Tu reçois un diagnostic et des données patient.
                            Expliques le résultat en français simple, comme un médecin parlerait à son patient.
                            Sois rassurant mais recommande une consultation.
                            Maximum 3 phrases.
                            Ne fais JAMAIS de diagnostic toi-même."""
        },
        {
            "role": "user",
            "content" : """Patient : Femme, 28 ans, region Dakar, symptomes: temperature 39.5, toux, fatigue, maux de tête
            Diagnostic du modèle : paludisme (probabilité 72%)
            Explique ce résultat au patient."""
        }
    ],
    max_tokens=200,
    temperature=0.3
)

print("\n\n=== Explication SenSante ===")
print(response2.choices[0].message.content)
