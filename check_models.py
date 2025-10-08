import google.generativeai as genai
import os

# Certifique-se de que sua chave de API está configurada como uma variável de ambiente
# GOOGLE_API_KEY="SUA_CHAVE_API"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("Modelos disponíveis que suportam 'generateContent':")
print("-" * 50)

for m in genai.list_models():
  # O erro original mencionava o método 'generateContent'. 
  # Vamos listar apenas os modelos que o suportam.
  if 'generateContent' in m.supported_generation_methods:
    # A API retorna o nome completo, como "models/gemini-pro"
    # Para usar na biblioteca, você usa apenas "gemini-pro"
    model_name = m.name.replace("models/", "")
    print(f"- {model_name}")

print("-" * 50)