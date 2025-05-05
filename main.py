from fastapi import FastAPI, Request

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello from BioClean.AI!"}

@app.post("/analyze")
async def analyze_data(request: Request):
    data = await request.json()
    # Just echoing data for now (mock response)
    return {
        "input": data,
        "insight": "This is a simulated AI response based on your input."
    }
