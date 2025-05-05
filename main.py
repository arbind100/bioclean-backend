from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import os, openai, pinecone, requests

# Load keys from environment
openai.api_key        = os.getenv("OPENAI_API_KEY")
pinecone_api_key      = os.getenv("PINECONE_API_KEY")
pinecone_env          = os.getenv("PINECONE_ENVIRONMENT")
weather_api_key       = os.getenv("OPENWEATHER_API_KEY")

# Init Pinecone
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
INDEX_NAME = "bio-clean-ai"
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(INDEX_NAME, dimension=1536)
index = pinecone.Index(INDEX_NAME)

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PollutionInput(BaseModel):
    pollutionType: str
    description: str
    location: str
    severity: int
    image: Optional[str] = None

def get_embeddings(text: str):
    resp = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return resp.data[0].embedding

def store_case(case_id: str, description: str):
    emb = get_embeddings(description)
    index.upsert([(case_id, emb, {"description": description})])

def get_similar_cases(text: str, top_k: int = 3):
    emb = get_embeddings(text)
    qres = index.query(vector=emb, top_k=top_k, include_metadata=True)
    return [m.metadata["description"] for m in qres.matches]

def get_weather_data(loc: str):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={loc}&appid={weather_api_key}&units=metric"
    r = requests.get(url)
    if r.ok:
        d = r.json().get("main", {})
        w = r.json().get("weather", [{}])[0]
        return {"temp": d.get("temp"), "humidity": d.get("humidity"), "desc": w.get("description")}
    return {}

@app.post("/generate-plan")
def generate_plan(input: PollutionInput):
    # 1) Record this case
    desc = f"{input.pollutionType} at {input.location}: {input.description}, severity {input.severity}"
    store_case(input.location + "-" + os.urandom(4).hex(), desc)

    # 2) Find similar past cases
    similar = get_similar_cases(desc)

    # 3) Get weather
    weather = get_weather_data(input.location)

    # 4) Build prompt
    prompt = (
        f"You are BioClean.AI...\n\n"
        f"User reported:\n • Type: {input.pollutionType}\n • Desc: {input.description}\n"
        f" • Loc: {input.location}\n • Sev: {input.severity}\n\n"
        f"Weather: {weather}\n\nSimilar cases:\n"
    )
    for i, c in enumerate(similar,1):
        prompt += f" {i}. {c}\n"
    prompt += "\nGenerate:\n1. Diagnosis\n2. Organisms\n3. Instructions\n4. Monitoring tips\n"

    # 5) Ask AI
    res = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role":"system","content":"You are BioClean.AI, expert in eco-remediation."},
                  {"role":"user","content":prompt}]
    )
    plan = res.choices[0].message.content
    return {"plan": plan, "similarCases": similar, "weather": weather}
