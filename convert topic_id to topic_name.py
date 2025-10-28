import numpy as np
import json

def readDataset(file_path):
    dict = {}
    with open(file_path) as f:
        dict = json.load(f)
    return dict

def get_topics(names):
    topics = []
    for value in names:
        uri_split = value.split('//')
        uri_split = uri_split[1].split('/')
        topic = uri_split[len(uri_split) - 1]
        topics.append(topic)
    return np.unique(topics)

topics_pred = readDataset("Data/ECAI_2025_Topic_Predictions.json")
topics_pred_ids = topics_pred.keys()
topics_pred_freqs = topics_pred.values()
topics_id_names = readDataset("Data/topic_label.json")
topics_id_names_ids = topics_id_names.keys()
topics_id_names_names = list(topics_id_names.values())
topics_id_names_ids = get_topics(topics_id_names_ids)
print(f"topics_id_names_ids: {topics_id_names_ids}")
print(f"topics_id_names_names: {topics_id_names_names}")
topic_ids_names = {}
counter = 0
for id in topics_id_names_ids:
    topic_ids_names[str(id)] = topics_id_names_names[counter]
    counter += 1

topic_preds_updated = {}
for pred_id, value in topics_pred.items():
    for real_id in topic_ids_names.keys():
        if pred_id == real_id:
            topic_preds_updated[topic_ids_names[real_id]] = value
print(topic_ids_names)
print(topics_pred)
print(topic_preds_updated)
pretty_json = json.dumps(topic_preds_updated, indent=4)
print(pretty_json)
with open("Data/ECAI_2025_Topic_Name_Predictions.json", "w") as f:
    json.dump(topic_preds_updated, f, indent=4)