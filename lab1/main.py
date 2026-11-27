import os
import sys
import nltk
import pymorphy3
from nltk.tokenize import sent_tokenize, word_tokenize

for resource in ('tokenizers/punkt', 'tokenizers/punkt_tab'):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1], quiet=True)

morph = pymorphy3.MorphAnalyzer()
TARGET_POS = {'NOUN', 'ADJF', 'ADJS'}


def is_word(token: str) -> bool:
    cleaned = token.replace('-', '')
    return bool(cleaned) and cleaned.isalpha()


def get_valid_parses(word: str):
    return [
        p for p in morph.parse(word)
        if p.tag.POS in TARGET_POS and 'Abbr' not in p.tag
    ]


def check_agreement(p1, p2) -> bool:
    if not (p1.tag.number and p2.tag.number):
        return False
    if p1.tag.number != p2.tag.number:
        return False

    # род сверяем в единственном числе
    if p1.tag.number == 'sing' and p1.tag.gender and p2.tag.gender:
        if p1.tag.gender != p2.tag.gender:
            return False

    # падеж сверяем, если он выражен у обоих слов
    if p1.tag.case and p2.tag.case and p1.tag.case != p2.tag.case:
        return False

    return True


def find_pairs(text: str):
    pairs = []
    for sentence in sent_tokenize(text, language='russian'):
        tokens = word_tokenize(sentence, language='russian')

        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            if not (is_word(w1) and is_word(w2)):
                continue

            parses1 = get_valid_parses(w1)
            parses2 = get_valid_parses(w2)
            if not parses1 or not parses2:
                continue

            best_pair = None
            best_score = -1.0

            for p1 in parses1:
                for p2 in parses2:
                    if check_agreement(p1, p2):
                        score = p1.score + p2.score
                        if score > best_score:
                            best_score = score
                            best_pair = (p1, p2)

            if best_pair:
                p1, p2 = best_pair
                pairs.append((w1, w2, p1.normal_form, p2.normal_form))

    return pairs


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, 'text.txt')

    if not os.path.exists(file_path):
        print(f"Файл не найден: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    for w1, w2, l1, l2 in find_pairs(text):
        print(f"{w1} {w2} -> {l1} {l2}")


if __name__ == '__main__':
    main()