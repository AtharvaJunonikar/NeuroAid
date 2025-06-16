import requests
import json

# Call Mistral API via Together.ai
def generate_explanation_together_ai(api_key, user_role, symptoms_list):
    symptoms_text = ", ".join(symptoms_list)
    prompt = (
        f"You are a medical assistant. The user is a {user_role}. "
        f"Based on these symptoms: {symptoms_text}, "
        f"suggest the most likely disease and explain it appropriately for the {user_role}."
    )
    
    api_url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer 533e342b9ecf84a6f38f30596d9a06289af1dd8a45de505ac849f2efd857d4cd",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",  # model name for Mistral
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 500
    }
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        return f"❌ Failed to generate explanation! Error: {response.status_code}"
