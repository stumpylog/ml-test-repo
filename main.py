import time

import pandas as pd

from classifier import OriginalDocumentClassifier
from data import Tag, Paper


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


def _main():
    """
    Command line interface to the module.
    """
    # Read the training dataset
    df = pd.read_csv("train.csv")

    papers = []

    for idx, row in df.iterrows():
        tags = _get_tags(row)

        papers.append(Paper(row["TITLE"], row["ABSTRACT"], tags))

    print("Training the models", flush=True)

    total_time = 0.0

    # Create a classifier
    for idx in range(5):
        orig = OriginalDocumentClassifier()

        start = time.time()
        orig.train(papers)
        end = time.time()

        print(f"Loop {idx}, time: {end - start}s", flush=True)
        total_time += (end - start)

    print(f"Average time: {total_time / 5.0}")


if __name__ == '__main__':
    _main()
