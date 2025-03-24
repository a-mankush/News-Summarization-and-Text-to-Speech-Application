from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Article(BaseModel):
    title: str = Field(description="Title of the article")
    summary: str = Field(description="Summary of the article")
    sentiment: Literal["negative", "positive", "neutral"] = Field(
        description="Sentiment of the article"
    )
    topics: list[str] = Field(description="Key Topics of the article")


class CoverageDifference(BaseModel):
    Comparison: str = Field(
        ..., description="Description of difference between articles"
    )
    Impact: str = Field(..., description="Potential impact of this difference")


class ComparativeSentimentScore(BaseModel):

    Coverage_Differences: List[CoverageDifference] = Field(
        ...,
        alias="Coverage Differences",
        description="Comparisons between article coverages",
    )
