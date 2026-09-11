# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Model wrappers: detoxify scoring plus Twitter sentiment analysis."""

from enum import Enum
from typing import Literal

from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.pipelines import pipeline as opt_pipeline
from pydantic import BaseModel, parse_obj_as
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


class SentimentAnalysisResults(BaseModel):
    """Sentiment probabilities per headline, one list per class."""

    positive: list[float]
    neutral: list[float]
    negative: list[float]


class DetoxifyCategory(str, Enum):
    """Toxicity categories the detoxify model can report."""

    toxicity = "toxicity"
    severe_toxicity = "severe_toxicity"
    obscene = "obscene"
    identity_attack = "identity_attack"
    insult = "insult"
    threat = "threat"
    sexual_explicit = "sexual_explicit"


class DetoxifyResults(BaseModel):
    """Toxicity scores per headline, one list per `DetoxifyCategory`."""

    toxicity: list[float]
    severe_toxicity: list[float]
    obscene: list[float]
    identity_attack: list[float]
    insult: list[float]
    threat: list[float]
    sexual_explicit: list[float]


class Scores(BaseModel):
    """All scores stored for a single headline: detoxify plus sentiment."""

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
    """Transpose the two sets of per-text results into per-headline `Scores`.

    Both models return parallel lists, one entry per input text, so the merged
    dictionaries are zipped into one score record per headline. The dicts are
    merged in class order, which puts the detoxify keys first.
    """
    detoxify_scores_dict = detoxify_results.dict()
    sa_scores_dict = sa_results.dict()
    scores_dict = dict(detoxify_scores_dict, **sa_scores_dict)

    keys = scores_dict.keys()
    vals = zip(*scores_dict.values(), strict=False)

    # create a list of dictionaries
    scores_list = [dict(zip(keys, v, strict=False)) for v in vals]

    return [Scores.parse_obj(s) for s in scores_list]


class DetoxifyModel:
    """Detoxify model, loading the ONNX-optimised variant from the Hub."""

    def __init__(self, *, local_files_only: bool = True) -> None:
        """Load the tokenizer and ONNX model into a sigmoid text-classifier.

        Args:
            local_files_only: refuse network access, using only files already
                present in the HuggingFace cache.

        """
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
        """Score each text against every `DetoxifyCategory`.

        Labels the model emits that aren't a known category are dropped, so
        every returned list holds exactly one score per input text.
        """

        class ModelOutput(BaseModel):
            """One `label`/`score` pair as returned by the pipeline."""

            label: DetoxifyCategory
            score: float

        label_set = {k.value for k in DetoxifyCategory}
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
    """Sentiment-analysis model, wrapping the Twitter XLM-R classifier."""

    def __init__(self, *, local_files_only: bool = True) -> None:
        """Load the tokenizer and model used for sentiment classification.

        Args:
            local_files_only: refuse network access, using only files already
                present in the HuggingFace cache.

        """
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
        """Classify each text as positive, neutral or negative.

        Winner takes all: the top-scoring class gets 100 and the others 0, so
        averaging these over headlines yields class ratios rather than mean
        probabilities.
        """

        class ModelOutput(BaseModel):
            """One sentiment `label`/`score` pair as returned by the pipeline."""

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
    """Both scoring models, combined into one per-headline `Scores`."""

    def __init__(self) -> None:
        """Load the detoxify and sentiment models."""
        self.detoxify_model = DetoxifyModel()
        self.sa_model = SAModel()

    def predict(self, texts: list[str]) -> list[Scores]:
        """Score each text with both models and merge the results."""
        detoxify_results = self.detoxify_model.predict(texts)
        sa_results = self.sa_model.predict(texts)

        return parse_results(detoxify_results, sa_results)
