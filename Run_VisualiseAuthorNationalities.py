import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sympy import ceiling
from tensorflow.keras import layers, models, losses, optimizers
import json

from tensorflow.python.layers.core import dropout


def readDataset(file_path):
    dict = {}
    with open(file_path) as f:
        dict = json.load(f)
    return dict

def remove_all_zeros_years(dict):
    updated_dict = {}
    isZero = True
    for key, value in dict.items():
        for v in value:
            if v != 0:
                isZero = False
        if isZero == False:
            updated_dict[key] = value
    return updated_dict

def get_years_topics(dict):
    years = []
    topics = []
    for key, value in dict.items():
        years.append(key)
        for v in value:
            uri_split = v.split('//')
            uri_split = uri_split[1].split('/')
            topics.append(uri_split[len(uri_split)-1])
    return np.unique(years), np.unique(topics)

def get_year_topics_freq(dict):
    topic_freq = {}
    for year, value in dict.items():
        freqs = []
        for v in value:
            freqs.append(dict[year][v])
        topic_freq[year] = freqs
    return topic_freq
def get_year_topics(dict):
    year_topics = {}
    for year, value in dict.items():
        topics = []
        for v in value:
            uri_split = v.split('//')
            uri_split = uri_split[1].split('/')
            topic = uri_split[len(uri_split) - 1]
            topics.append(topic)
        year_topics[year] = topics
    return year_topics

countries_dict = readDataset("Data/countries.json")
print(f"countries_dict: {countries_dict}")
countries = list(countries_dict.keys())
counts = list(countries_dict.values())
print(f"countries: {countries}")
print(f"counts: {counts}")
# Ensure inputs are the same length
assert len(countries) == len(counts), "countries and counts must be the same length"

# Combine into pairs and sort
pairs = list(zip(countries, counts))
# Sort by count ascending for bottom, descending for top
pairs_sorted = sorted(pairs, key=lambda x: x[1])

# Bottom 10
bottom_10 = pairs_sorted[:30]
bottom_countries, bottom_counts = zip(*bottom_10) if bottom_10 else ([], [])

# Top 10
top_10 = pairs_sorted[-30:]  # last 10 after ascending sort
top_countries, top_counts = zip(*top_10) if top_10 else ([], [])

# Plotting helper
def plot_bar(x_labels, y_values, title, color=None):
    plt.figure(figsize=(10, 6))
    y = np.array(y_values, dtype=float)
    x = np.arange(len(x_labels))
    plt.barh(x, y, color=color if color else 'steelblue')
    plt.yticks(x, x_labels)
    plt.xlabel('Author Count')
    plt.title(title)
    plt.tight_layout()
    plt.gca().invert_yaxis()  # Optional: highest at top
    plt.show()

# Top 10 plot
plot_bar(list(top_countries), list(top_counts),
         title='Top 30 Countries by Author Frequency')

# Bottom 10 plot
plot_bar(list(bottom_countries), list(bottom_counts),
         title='Bottom 30 Countries by Author Frequency')