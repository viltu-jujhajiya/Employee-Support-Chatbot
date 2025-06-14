from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os
from pathlib import Path

project_path = Path(__file__).resolve().parent
os.chdir(project_path)

from src.main import escb




app = FastAPI()

obj = escb()

class QueryInput(BaseModel):
    query: str

@app.post("/query")
async def generate_response(payload: QueryInput):
    response = obj.main(payload.query)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=3456)