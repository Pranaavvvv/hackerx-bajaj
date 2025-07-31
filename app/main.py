from fastapi import FastAPI
from .api.endpoints import evaluation, hackrx
from .utils.error_handlers import APIException, api_exception_handler

app = FastAPI(
    title="PolicyEval-GPT & HackRx Q&A",
    description="An AI-powered API for answering questions about policy documents.",
    version="2.0.0"
)

# API Routers
# app.include_router(evaluation.router, prefix="/v1", tags=["Legacy Evaluation"]) # Deprecated
app.include_router(hackrx.router, prefix="/hackrx", tags=["Q&A"])

# Exception Handler
app.add_exception_handler(APIException, api_exception_handler)

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the Simplified PolicyEval-GPT API"}
