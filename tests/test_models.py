from toxic_news.models import (
    DetoxifyResults,
    Scores,
    SentimentAnalysisResults,
    parse_results,
)


def test_parse_results():
    detoxify_results = DetoxifyResults(
        toxicity=[1, 2],
        severe_toxicity=[3, 4],
        obscene=[5, 6],
        identity_attack=[7, 8],
        insult=[9, 10],
        threat=[11, 12],
        sexual_explicit=[13, 14],
    )
    sa_results = SentimentAnalysisResults(
        positive=[15, 16],
        neutral=[17, 18],
        negative=[19, 20],
    )
    result = parse_results(detoxify_results, sa_results)

    expected = [
        Scores(
            toxicity=1,
            severe_toxicity=3,
            obscene=5,
            identity_attack=7,
            insult=9,
            threat=11,
            sexual_explicit=13,
            positive=15,
            neutral=17,
            negative=19,
        ),
        Scores(
            toxicity=2,
            severe_toxicity=4,
            obscene=6,
            identity_attack=8,
            insult=10,
            threat=12,
            sexual_explicit=14,
            positive=16,
            neutral=18,
            negative=20,
        ),
    ]

    assert result == expected
