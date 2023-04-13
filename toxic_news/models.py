from typing import Literal

from detoxify import Detoxify
from pydantic import BaseModel, parse_obj_as
from transformers import pipeline


class SentimentAnalysisResults(BaseModel):
    positive: list[float]
    neutral: list[float]
    negative: list[float]


class DetoxifyResults(BaseModel):
    toxicity: list[float]
    severe_toxicity: list[float]
    obscene: list[float]
    identity_attack: list[float]
    insult: list[float]
    threat: list[float]
    sexual_explicit: list[float]


class Scores(BaseModel):
    # detoxify scores
    toxicity: float
    severe_toxicity: float
    obscene: float
    identity_attack: float
    insult: float
    threat: float
    sexual_explicit: float
    # sentiment analysis scores
    positive: float
    neutral: float
    negative: float


def parse_results(
    detoxify_results: DetoxifyResults, sa_results: SentimentAnalysisResults
) -> list[Scores]:
    detoxify_scores_dict = detoxify_results.dict()
    sa_scores_dict = sa_results.dict()
    scores_dict = dict(detoxify_scores_dict, **sa_scores_dict)

    keys = scores_dict.keys()
    vals = zip(*scores_dict.values())

    # create a list of dictionaries
    scores_list = [dict(zip(keys, v)) for v in vals]

    return [Scores.parse_obj(s) for s in scores_list]


class SAModel:
    def __init__(self):
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        self.model = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            return_all_scores=True,
        )

    def predict(self, texts: list[str]) -> SentimentAnalysisResults:
        class ModelOutput(BaseModel):
            label: Literal["positive", "neutral", "negative"]
            score: float

        outputs = parse_obj_as(list[list[ModelOutput]], self.model(texts))
        results: dict[str, list[float]] = {
            "positive": [],
            "neutral": [],
            "negative": [],
        }
        for prediction in outputs:
            for single_score in prediction:
                results[single_score.label].append(single_score.score)

        return SentimentAnalysisResults.parse_obj(results)


class AllModels:
    def __init__(self):
        self.detoxify_model = Detoxify("multilingual")
        self.sa_model = SAModel()

    def predict(self, texts: list[str]) -> list[Scores]:
        detoxify_results = DetoxifyResults.parse_obj(self.detoxify_model.predict(texts))
        sa_results = self.sa_model.predict(texts)

        return parse_results(detoxify_results, sa_results)
