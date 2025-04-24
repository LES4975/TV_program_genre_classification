import google.generativeai as genai
import AI_Synopsis_Classifier

for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)