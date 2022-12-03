import pickle
import re
from pathlib import Path
from typing import List

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.utils.multiclass import type_of_target

from data import Paper


class OriginalDocumentClassifier:

    def __init__(self):

        self.data_vectorizer = None
        self.tags_binarizer = None
        self.tags_classifier = None

        self.save_path = Path("new.dat")

    def train(self, papers: List[Paper]):

        data = []
        labels_tags = []

        # Step 1: Extract and preprocess training data from the database.
        print("Gathering data from database...")

        for paper in papers:
            content = paper.title + " " + paper.abstract

            preprocessed_content = self.preprocess_content(content)
            data.append(preprocessed_content)

            tags = sorted(
                tag.value for tag in paper.tags
            )
            labels_tags.append(tags)

        labels_tags_unique = {tag for tags in labels_tags for tag in tags}

        num_tags = len(labels_tags_unique)

        # Step 2: vectorize data
        print("Vectorizing data...")
        self.data_vectorizer = CountVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=0.01,
        )
        data_vectorized = self.data_vectorizer.fit_transform(data)

        # See the notes here:
        # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html  # noqa: 501
        # This attribute isn't needed to function and can be large
        self.data_vectorizer.stop_words_ = None

        # Step 3: train the classifiers
        if num_tags > 0:
            print("Training tags classifier...")

            if num_tags == 1:
                # Special case where only one tag has auto:
                # Fallback to binary classification.
                labels_tags = [
                    label[0] if len(label) == 1 else -1 for label in labels_tags
                ]
                self.tags_binarizer = LabelBinarizer()
                labels_tags_vectorized = self.tags_binarizer.fit_transform(
                    labels_tags,
                ).ravel()
            else:
                self.tags_binarizer = MultiLabelBinarizer()
                labels_tags_vectorized = self.tags_binarizer.fit_transform(labels_tags)

            self.tags_classifier = MLPClassifier(tol=0.01)
            self.tags_classifier.fit(data_vectorized, labels_tags_vectorized)
        else:
            self.tags_classifier = None
            print("There are no tags. Not training tags classifier.")

        return True

    def preprocess_content(self, content: str) -> str:
        # Lower case the document
        content = content.lower().strip()
        # Reduce spaces
        content = re.sub(r"\s+", " ", content)
        # Get only the letters
        content = re.sub(r"[^\w\s]", " ", content)

        return content

    def save(self):
        with self.save_path.open("wb") as outfile:
            pickle.dump(self.data_vectorizer, outfile)
            pickle.dump(self.tags_binarizer, outfile)
            pickle.dump(self.tags_classifier, outfile)

    def load(self):
        with self.save_path.open("rb") as infile:
            self.data_vectorizer = pickle.load(infile)
            self.tags_binarizer = pickle.load(infile)
            self.tags_classifier = pickle.load(infile)

    def predict_tags(self, paper: Paper):

        content = paper.title + " " + paper.abstract

        if self.tags_classifier:
            X = self.data_vectorizer.transform([self.preprocess_content(content)])
            y = self.tags_classifier.predict(X)
            tags_ids = self.tags_binarizer.inverse_transform(y)[0]
            if type_of_target(y).startswith("multilabel"):
                # the usual case when there are multiple tags.
                return list(tags_ids)
            elif type_of_target(y) == "binary" and tags_ids != -1:
                # This is for when we have binary classification with only one
                # tag and the result is to assign this tag.
                return [tags_ids]
            else:
                # Usually binary as well with -1 as the result, but we're
                # going to catch everything else here as well.
                return []
        else:
            return []
