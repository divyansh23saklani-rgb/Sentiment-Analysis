# data_augmentation.py

import random
import re
import string
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from tqdm import tqdm

# Download required NLTK resources
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Mapping NLTK POS tags to WordNet POS
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

def synonym_replacement(sentence, n=2):
    words = word_tokenize(clean_text(sentence))
    tagged_words = pos_tag(words)

    eligible = [(word, get_wordnet_pos(tag)) for word, tag in tagged_words if get_wordnet_pos(tag)]

    if not eligible:
        return sentence  # nothing to replace

    new_words = words[:]
    replaced = 0

    for _ in range(n):
        word, pos = random.choice(eligible)
        synsets = wordnet.synsets(word, pos=pos)
        synonyms = list(set(lemma.name().replace("_", " ") for s in synsets for lemma in s.lemmas() if lemma.name().lower() != word.lower()))

        if synonyms:
            synonym = random.choice(synonyms)
            new_words = [synonym if w == word else w for w in new_words]
            replaced += 1

    if replaced == 0:
        return sentence
    return " ".join(new_words)

# Augment multiple times
def augment_n_times(text, n=5, replacements_per_text=2):
    return [synonym_replacement(text, replacements_per_text) for _ in range(n)]

if __name__ == "__main__":
    text = input("Enter text: ").strip()
    num_aug = int(input("How many augmented versions? (e.g., 5): ").strip())
    replacements = int(input("How many words to replace per version? (e.g., 2): ").strip())

    print("\n🔁 Generating Augmented Versions:\n")
    for i, aug in enumerate(augment_n_times(text, num_aug, replacements), 1):
        print(f"{i}. {aug}")
