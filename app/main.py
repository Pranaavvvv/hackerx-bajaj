import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.endpoints import evaluation, hackrx
from .utils.error_handlers import APIException, api_exception_handler

def create_app() -> FastAPI:
    app = FastAPI(
        title="PolicyEval-GPT & HackRx Q&A",
        description="An AI-powered API for answering questions about policy documents.",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API Routers
    app.include_router(hackrx.router, prefix="/hackrx", tags=["Q&A"])

    # Exception Handler
    app.add_exception_handler(APIException, api_exception_handler)

    @app.get("/", tags=["Root"])
    async def read_root():
        return {
            "message": "Welcome to the Simplified PolicyEval-GPT API",
            "docs": "/docs",
            "version": "2.0.0"
        }

    return app

# Create the FastAPI app
app = create_app()

# This allows the app to be run directly with: python -m app.main
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
