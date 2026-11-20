import os
import sys
import nltk
import pymorphy3
from nltk.tokenize import sent_tokenize, word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

morph = pymorphy3.MorphAnalyzer()
target_pos = {'NOUN', 'ADJF', 'ADJS'}

def is_word(token):
    return token.isalpha() or ('-' in token and all(p.isalpha() for p in token.split('-') if p))

def check_agreement(p1, p2):
    # проверяем, что число и падеж есть и совпадают
    if not (p1.tag.number and p2.tag.number and p1.tag.case and p2.tag.case):
        return False
    if p1.tag.number != p2.tag.number or p1.tag.case != p2.tag.case:
        return False
    # род проверяем только в единственном числе
    if p1.tag.gender and p2.tag.gender and p1.tag.gender != p2.tag.gender:
        return False
    return True

def find_pairs(text):
    pairs = []
    for sentence in sent_tokenize(text, language='russian'):
        tokens = word_tokenize(sentence, language='russian')

        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            if not (is_word(w1) and is_word(w2)):
                continue

            top1, top2 = morph.parse(w1)[0], morph.parse(w2)[0]
            if top1.tag.POS not in target_pos or top2.tag.POS not in target_pos:
                continue
            if 'Abbr' in top1.tag or 'Abbr' in top2.tag:
                continue

            parses1 = [p for p in morph.parse(w1) if p.tag.POS in target_pos and 'Abbr' not in p.tag]
            parses2 = [p for p in morph.parse(w2) if p.tag.POS in target_pos and 'Abbr' not in p.tag]

            # если второе слово существительное, первое предпочтительно разбираем как прилагательное
            if top2.tag.POS == 'NOUN':
                parses1.sort(key=lambda p: (p.tag.POS != 'ADJF', -p.score))

            # перебираем все разборы обоих слов в поисках согласования
            match = None
            for p1 in parses1:
                for p2 in parses2:
                    if check_agreement(p1, p2):
                        match = (p1, p2)
                        break
                if match:
                    break

            if match:
                pairs.append((w1, w2, match[0].normal_form, match[1].normal_form))

    return pairs

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, 'text.txt')

    if not os.path.exists(file_path):
        print(f"файл не найден: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    for w1, w2, l1, l2 in find_pairs(text):
        print(f"{w1} {w2} -> {l1} {l2}")

if __name__ == '__main__':
    main()
