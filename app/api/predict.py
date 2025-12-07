from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from app.models.model import predict_roles
from app.db import save_result, get_recent_results

router = APIRouter()


class PredictRequest(BaseModel):
    skills: list[str]
    interests: list[str]
    experience_years: int = 0
    top_k: int = 3

    @field_validator("experience_years")
    @classmethod
    def non_negative_experience(cls, v: int) -> int:
        if v < 0:
            raise ValueError("experience_years cannot be negative")
        return v

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("top_k must be at least 1")
        if v > 10:
            # hard cap to avoid silly values
            raise ValueError("top_k must not be greater than 10")
        return v


@router.post("/predict")
def predict(req: PredictRequest):
    # Basic input validation: require some signal
    if not req.skills and not req.interests:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one skill or interest.",
        )

    combined_text = ";".join(req.skills) + " " + ";".join(req.interests)
    if len(combined_text) > 2000:
        raise HTTPException(
            status_code=400,
            detail="Input text too long. Please provide a shorter list of skills/interests.",
        )

    texts = [combined_text]

    # Get top_k predictions (the model function will handle ranking)
    results = predict_roles(texts, top_k=req.top_k)
    top_result = results[0]

    # Save to database
    save_result(
        skills=req.skills,
        interests=req.interests,
        experience_years=req.experience_years,
        prediction=top_result["role"],
        confidence=top_result["confidence"],
    )

    return {
        "input": {
            "skills": req.skills,
            "interests": req.interests,
            "experience_years": req.experience_years,
            "top_k": req.top_k,
        },
        "predictions": results,
    }


@router.get("/history")
def history(limit: int = 10) -> dict[str, Any]:
    """
    Return the most recent predictions.
    """
    if limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be positive")
    if limit > 50:
        raise HTTPException(status_code=400, detail="limit must not exceed 50")

    rows = get_recent_results(limit=limit)
    items = []
    for skills, interests, exp, prediction, confidence, created_at in rows:
        items.append(
            {
                "skills": skills.split(";") if skills else [],
                "interests": interests.split(";") if interests else [],
                "experience_years": exp,
                "prediction": prediction,
                "confidence": confidence,
                "created_at": created_at,
            }
        )
    return {"results": items}
