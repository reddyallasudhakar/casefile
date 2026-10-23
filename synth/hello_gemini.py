from google import genai

client = genai.Client(vertexai=True, project="casefile-ml", location="global")

reply = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="In one sentence: what is a cashless claim denial in Indian health insurance?",
)
print(reply.text)