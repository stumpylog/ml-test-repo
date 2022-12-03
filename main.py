import time
from dataclasses import dataclass

import pandas as pd
import psutil
from humanfriendly import format_size

from classifier import NltkDocumentClassifier, OriginalDocumentClassifier
from data import Tag, Paper


@dataclass
class Result:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0


def _get_tags(row):
    tags = []
    if row["Computer Science"] == 1:
        tags.append(Tag.ComputerScience)
    if row["Physics"] == 1:
        tags.append(Tag.Physics)
    if row["Mathematics"] == 1:
        tags.append(Tag.Mathematics)
    if row["Statistics"] == 1:
        tags.append(Tag.Statistics)
    if row["Quantitative Biology"] == 1:
        tags.append(Tag.QuantBiology)
    if row["Quantitative Finance"] == 1:
        tags.append(Tag.QuantFinance)
    return tags


def _update_score(predicted, actual, score: Result):
    for tag in predicted:
        if tag in actual:
            score.true_positive += 1
        else:
            score.false_positive += 1
        missing = set(actual) - set(predicted)
        score.false_negative += len(missing)
    return score


def _main():
    """Command line interface to the module.
    """
    df = pd.read_csv("train.csv")

    papers = []
    do_train = False

    for idx, row in df.iterrows():
        tags = _get_tags(row)

        papers.append(Paper(row["TITLE"], row["ABSTRACT"], tags))

    train_papers = papers[:len(papers) // 2]
    test_papers = papers[(len(papers) // 2) + 1:]

    self = psutil.Process()

    nltk = NltkDocumentClassifier()
    orig = OriginalDocumentClassifier()

    if do_train:
        print("Re-training the models")
        start_res = self.memory_info().rss
        start = time.time()
        nltk.train(train_papers)
        nltk.save()
        end = time.time()
        end_res = self.memory_info().rss

        print(f"NLTK time: {end - start}s")
        print(f"NLTK Mem: {format_size(end_res - start_res)}")

        start_res = self.memory_info().rss
        start = time.time()
        orig.train(train_papers)
        orig.save()
        end = time.time()
        end_res = self.memory_info().rss

        print(f"Orig time: {end - start}s")
        print(f"Orig Mem: {format_size(end_res - start_res)}")
    else:
        print("Loading training from file")
        nltk.load()
        orig.load()

    original_score = Result()
    nltk_score = Result()

    for paper in test_papers:
        nltk_tags = [Tag(x) for x in nltk.predict_tags(paper)]
        orig_tags = [Tag(x) for x in orig.predict_tags(paper)]
        expected_tags = paper.tags

        nltk_score = _update_score(nltk_tags, expected_tags, nltk_score)
        original_score = _update_score(orig_tags, expected_tags, original_score)

    nltk_precision = nltk_score.true_positive / (nltk_score.true_positive + nltk_score.false_positive)
    orig_precision = original_score.true_positive / (original_score.true_positive + original_score.false_positive)

    nltk_recall = nltk_score.true_positive / (nltk_score.true_positive + nltk_score.false_negative)
    original_recall = original_score.true_positive / (original_score.true_positive + original_score.false_negative)

    print(f"NLTK: {nltk_precision * 100.0} : {nltk_recall * 100.0}")
    print(f"ORIG: {orig_precision * 100.0} : {original_recall * 100.0}")


if __name__ == '__main__':
    _main()
