from enum import Enum
from typing import Literal

from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.pipelines import pipeline as opt_pipeline
from pydantic import BaseModel, parse_obj_as
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


class SentimentAnalysisResults(BaseModel):
    positive: list[float]
    neutral: list[float]
    negative: list[float]


class DetoxifyCategory(str, Enum):
    toxicity = "toxicity"
    severe_toxicity = "severe_toxicity"
    obscene = "obscene"
    identity_attack = "identity_attack"
    insult = "insult"
    threat = "threat"
    sexual_explicit = "sexual_explicit"


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


class DetoxifyModel:
    def __init__(self, local_files_only=True):
        model_name = "dcferreira/detoxify-optimized"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        model = ORTModelForSequenceClassification.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model = opt_pipeline(
            model=model,
            task="text-classification",
            function_to_apply="sigmoid",
            accelerator="ort",
            tokenizer=tokenizer,
            top_k=None,
        )

    def predict(self, texts: list[str]) -> DetoxifyResults:
        class ModelOutput(BaseModel):
            label: DetoxifyCategory
            score: float

        label_set = set(k.value for k in DetoxifyCategory)
        preds = self.model(texts)
        preds_without_identity_classes = [
            list(filter(lambda x: x["label"] in label_set, p)) for p in preds
        ]

        outputs = parse_obj_as(list[list[ModelOutput]], preds_without_identity_classes)
        results: dict[str, list[float]] = {k: [] for k in label_set}
        for prediction in outputs:
            for single_score in prediction:
                results[single_score.label].append(single_score.score)

        return DetoxifyResults.parse_obj(results)


class SAModel:
    def __init__(self, local_files_only=True):
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-xlm-roberta-base-sentiment",
            local_files_only=local_files_only,
        )

        self.model = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            top_k=None,
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
            # give 100% to the most predicted class, 0% to the others
            # when these percentages are averaged, they will correspond to the ratio
            # of positive/neutral/negative headlines instead of avgs of scores
            sorted_preds = sorted(prediction, key=lambda x: x.score, reverse=True)
            results[sorted_preds[0].label].append(100)  # 100%
            for single_score in sorted_preds[1:]:
                results[single_score.label].append(0)  # 0%

        return SentimentAnalysisResults.parse_obj(results)


class AllModels:
    def __init__(self):
        self.detoxify_model = DetoxifyModel()
        self.sa_model = SAModel()

    def predict(self, texts: list[str]) -> list[Scores]:
        detoxify_results = self.detoxify_model.predict(texts)
        sa_results = self.sa_model.predict(texts)

        return parse_results(detoxify_results, sa_results)
