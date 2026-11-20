import os
import sys
import nltk
import pymorphy3
from nltk.tokenize import sent_tokenize, word_tokenize

morph = pymorphy3.MorphAnalyzer()
target_pos = {'NOUN', 'ADJF', 'ADJS'}


def is_word(token):
    return token.isalpha() or ('-' in token and all(p.isalpha() for p in token.split('-') if p))


def check_agreement(p1, p2):
    # число и падеж должны совпадать
    if not (p1.tag.number and p2.tag.number and p1.tag.case and p2.tag.case):
        return False
    if p1.tag.number != p2.tag.number or p1.tag.case != p2.tag.case:
        return False
    # род проверяем только в ед. числе
    if p1.tag.gender and p2.tag.gender and p1.tag.gender != p2.tag.gender:
        return False
    return True


def to_nominative(p1, p2):
    # если пара во множественном числе, возвращаем тоже во множественном
    if p1.tag.number == 'plur' or p2.tag.number == 'plur':
        inf1 = p1.inflect({'nomn', 'plur'})
        inf2 = p2.inflect({'nomn', 'plur'})
        return (inf1.word if inf1 else p1.normal_form), (inf2.word if inf2 else p2.normal_form)

    is_adj1 = p1.tag.POS in ('ADJF', 'ADJS')
    is_adj2 = p2.tag.POS in ('ADJF', 'ADJS')

    # прил + сущ
    if is_adj1 and p2.tag.POS == 'NOUN':
        gender = p2.tag.gender
        inf1 = p1.inflect({'nomn', gender, 'sing'} if gender else {'nomn', 'sing'})
        return (inf1.word if inf1 else p1.normal_form), p2.normal_form

    # сущ + прил
    if p1.tag.POS == 'NOUN' and is_adj2:
        gender = p1.tag.gender
        inf2 = p2.inflect({'nomn', gender, 'sing'} if gender else {'nomn', 'sing'})
        return p1.normal_form, (inf2.word if inf2 else p2.normal_form)

    # прил + прил
    if is_adj1 and is_adj2:
        gender = p2.tag.gender or p1.tag.gender or 'masc'
        inf1 = p1.inflect({'nomn', gender, 'sing'} if gender else {'nomn', 'sing'})
        inf2 = p2.inflect({'nomn', gender, 'sing'} if gender else {'nomn', 'sing'})
        return (inf1.word if inf1 else p1.normal_form), (inf2.word if inf2 else p2.normal_form)

    # сущ + сущ
    return p1.normal_form, p2.normal_form


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

            # не цепляем связку сущ + прил + сущ ('раз молодой человек')
            if top1.tag.POS == 'NOUN' and top2.tag.POS in ('ADJF', 'ADJS') and i + 2 < len(tokens):
                if morph.parse(tokens[i + 2])[0].tag.POS == 'NOUN':
                    continue

            parses1 = [p for p in morph.parse(w1) if p.tag.POS in target_pos and 'Abbr' not in p.tag]
            parses2 = [p for p in morph.parse(w2) if p.tag.POS in target_pos and 'Abbr' not in p.tag]

            # если второе слово существительное, первое пробуем как прилагательное
            if top2.tag.POS == 'NOUN':
                parses1.sort(key=lambda p: (p.tag.POS != 'ADJF', -p.score))

            match = None
            for p1 in parses1:
                for p2 in parses2:
                    if check_agreement(p1, p2):
                        match = (p1, p2)
                        break
                if match:
                    break

            if match:
                nom1, nom2 = to_nominative(match[0], match[1])
                pairs.append((w1, w2, nom1, nom2))

    return pairs


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, 'text.txt')

    if not os.path.exists(file_path):
        print(f"файл не найден: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    for w1, w2, nom1, nom2 in find_pairs(text):
        print(f"{w1} {w2} -> {nom1} {nom2}")


if __name__ == '__main__':
    main()
