from src.main import escb
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class QueryInput(BaseModel):
    query: str

@app.post("\query")
async def generate_response(payload: QueryInput):
    obj = escb()
    response = obj.main("How many holidays do we get?")
    print(response)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=3456)