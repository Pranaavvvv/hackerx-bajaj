from fastapi import APIRouter, Depends, HTTPException
from app.api.schemas.evaluation import HackRxRequest, HackRxResponse
from app.core.security import get_api_key
from app.services.qa_service import QAService

router = APIRouter()

qa_service = QAService()

@router.post("/run", response_model=HackRxResponse, tags=["Q&A"])
async def run_hackrx_evaluation(
    request: HackRxRequest,
    api_key: str = Depends(get_api_key)
):
    """
    This endpoint processes a document from a URL against a list of questions.

    - **Authentication**: Requires a Bearer token in the `Authorization` header.
    - **Request**: Takes a document URL and a list of questions.
    - **Response**: Returns a list of answers corresponding to the questions.
    """
    try:
        answers = await qa_service.answer_questions(request)
        return HackRxResponse(answers=answers)
    except Exception as e:
        # For debugging, it's helpful to see the error.
        # In production, you might want a more generic error message.
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
