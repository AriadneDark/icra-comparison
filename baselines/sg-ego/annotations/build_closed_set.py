import argparse
import ast  # to safely evaluate the list in the 'group' column
import csv
import json

import numpy as np
import spacy
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from transformers import logging

logging.set_verbosity_error()


def read_egoclip_csv(path) -> set[str]:
    object_list = []

    # If your file is actually .csv, change the extension; otherwise Python can still read it as text
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group = row["group"]

            # Safely convert the string representation of the list to a Python list
            group_list = ast.literal_eval(group)
            object_list.extend(group_list)

    object_list = [obj.replace("_", " ") for obj in object_list]
    object_list = set(object_list) - set("".join(words.split(" ")) for words in object_list if len(words.split(" ")) > 1)
    object_list = object_list - {"24 wall plate"}  # remove typos

    return set(object_list)


def clean_relations(relations):
    # Load the English language model
    nlp = spacy.load("en_core_web_sm")

    # Parse and convert verbs to infinitive form
    relations_list_infinitive = []

    for word in relations:
        doc = nlp(word)
        # Get the lemma (base form) of the word
        lemmatized_word = " ".join([token.lemma_ for token in doc])
        relations_list_infinitive.append(lemmatized_word)

    return set(relations_list_infinitive)


def clusterize(entities: set[str], eps: float) -> set[str]:
    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    text_encodings = encoder.encode(list(entities))

    text_encodings = torch.nn.functional.normalize(torch.tensor(text_encodings), p=2, dim=1)
    similarities = text_encodings @ text_encodings.T
    distances = np.clip(1 - similarities.numpy(), min=0)  # Convert to distance matrix

    model = DBSCAN(eps=eps, min_samples=1, metric="precomputed")
    preds = model.fit_predict(distances)

    unique_entities = []

    for _cls in range(max(preds) + 1):
        sample = [obj for obj, pred in zip(entities, preds) if pred == _cls]
        unique_entities.append(sample[0])

    return set(unique_entities)


def main(egoclip_verbs_path, egoclip_nouns_path, pvsg_relations_path):
    nouns_closed_set = read_egoclip_csv(egoclip_nouns_path)
    relations_closed_set = clean_relations(read_egoclip_csv(egoclip_verbs_path) | set(json.load(open(pvsg_relations_path, "r", encoding="utf-8"))))
    relations_closed_set = relations_closed_set | {'left', 'right', 'above', 'behind', 'beneath'}

    print("Number of unique nouns in the closed set:", len(nouns_closed_set))
    print("Number of unique relations in the closed set:", len(relations_closed_set))

    clusterized_nouns = clusterize(nouns_closed_set, eps=0.15)
    clusterized_relations = clusterize(relations_closed_set, eps=0.15)

    print("Number of unique nouns in the clusterized closed set:", len(clusterized_nouns))
    print("Number of unique relations in the clusterized closed set:", len(clusterized_relations))

    json.dump(sorted(clusterized_nouns), open("objects.json", 'w'), indent=2)
    json.dump(sorted(clusterized_relations), open("relations.json", 'w'), indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build closed set for scene graph annotation generation")
    parser.add_argument("--egoclip-verbs-path", type=str, default="egoclip_annotations/narration_verb_taxonomy.csv", help="Path to the closed set file (txt or json)")
    parser.add_argument("--egoclip-nouns-path", type=str, default="egoclip_annotations/narration_noun_taxonomy.csv", help="Path to save the processed closed set (json)")
    parser.add_argument("--pvsg-relations-path", type=str, default="pvsg_annotations/relations.json", help="Path to save the processed closed set (json)")

    args = parser.parse_args()

    main(args.egoclip_verbs_path, args.egoclip_nouns_path, args.pvsg_relations_path)
