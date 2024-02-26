# For Spacy
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import spacy

#import cld_tools.utils as utils
import gloss_lost.utils as utils


### Create a custom tokeniser to keep hyphenated words ###
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, \
CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

# From spaCy/spacy/lang/en/punctuation.py
_infixes = (
    LIST_ELLIPSES
    + LIST_ICONS
    + [
        r"(?<=[0-9])[+\-\*^](?=[0-9-])",
        r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
            al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
        ),
        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
        #r"(?<=[{a}0-9])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
        r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
    ]
)

# From spaCy/spacy/lang/es/punctuation.py
_concat_quotes = CONCAT_QUOTES + "—–"
_infixes_es = (
    LIST_ELLIPSES
    + LIST_ICONS
    + [
        r"(?<=[0-9])[+\*^](?=[0-9-])",
        r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
            al=ALPHA_LOWER, au=ALPHA_UPPER, q=_concat_quotes
        ),
        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
        r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
    ]
)

### Load nlp ###
nlp = spacy.load('en_core_web_sm')
nlp_hyphen = spacy.load('en_core_web_sm') # NOT keeping the hyphens
nlp_hyphen.tokenizer.infix_finditer = compile_infix_regex(_infixes).finditer # For hyphens
nlp_es = spacy.load('es_core_news_sm')
nlp_es_hyphen = spacy.load('es_core_news_sm') # NOT keeping the hyphens
nlp_es_hyphen.tokenizer.infix_finditer = compile_infix_regex(_infixes_es).finditer

# Processor
def nlp_processor(raw_sentence, language='en', hyphen=True):
    '''Use Spacy with the defined language.

    hyphen: True means KEEPING hyphens in words.'''
    if language == 'en':
        if hyphen:
            process_sent = nlp_hyphen(raw_sentence)
        else:
            process_sent = nlp(raw_sentence)
    elif language == 'es':
        if hyphen:
            process_sent = nlp_es_hyphen(raw_sentence)
        else:
            process_sent = nlp_es(raw_sentence)
    else:
        raise ValueError(f'Unknown language: {language}')
    return process_sent

# Tokenise a sentence
def tokenise_sentence(raw_sentence, language='en', hyphen=True):
    '''Tokenise a sentence.'''
    process_sent = nlp_processor(raw_sentence, language, hyphen)
    return ' '.join([str(word) for word in process_sent])

def tokenise_file(raw_file, language='en', hyphen=True):
    '''Tokenise a text.'''
    split_raw_file = utils.text_to_line(raw_file, empty=False)
    tokenised_list = []
    for line in split_raw_file:
        tokenised_list.append(
                    tokenise_sentence(line, language=language, hyphen=hyphen))
    return '\n'.join(tokenised_list)

# Lemmatise a sentence
def lemmatise_sentence(raw_sentence, language='en', hyphen=True):
    '''Lemmatise a sentence with spaCy.

    The list is as follows: [lemma].'''
    #split_sent = utils.line_to_word(raw_sentence)
    #n = len(split_sent)
    process_sent = nlp_processor(raw_sentence, language, hyphen)
    tokenised_sent = [str(word) for word in process_sent]
    lemmatised_sentence = []
    for i in range(len(process_sent)):
        token = process_sent[i]
        if (token.lemma_ != token.lemma_.lower()) and (token.lemma_ != 'I'):
            print(f'Lowered token: {token.lemma_}')
        lemma = token.lemma_.lower().replace(' ', r'.')
        lemmatised_sentence.append(lemma) #token.lemma_.lower())
    utils.check_equality(len(tokenised_sent), len(lemmatised_sentence))
    return tokenised_sent, lemmatised_sentence


def lemmatise_sentence_for_alignment(raw_sentence, language='en', hyphen=True):
    '''Lemmatise a sentence with spaCy while keeping the original index.

    The list is as follows: [(lemma, PoS, original_index)].
    The original index: index from the tokenised sentence.'''
    #split_sent = utils.line_to_word(raw_sentence)
    #n = len(split_sent)
    #process_sent = nlp(raw_sentence)
    process_sent = nlp_processor(raw_sentence, language, hyphen)
    tokenised_sent = [str(word) for word in process_sent]
    lemmatised_sentence = []
    for i in range(len(process_sent)):
        token = process_sent[i]
        if (token.lemma_ != token.lemma_.lower()) and (token.lemma_ != 'I'):
            pass #print(f'Lowered token: {token.lemma_}')
        lemma = token.lemma_.lower().replace(' ', r'.')
        lemmatised_sentence.append((lemma, token.pos_, i))
    utils.check_equality(len(tokenised_sent), len(lemmatised_sentence))
    return tokenised_sent, lemmatised_sentence

def sentence_pos_tag(raw_sentence, language='en', hyphen=True):
    '''Give the PoS tags for each unit in the sentence.'''
    #process_sent = nlp(raw_sentence)
    process_sent = nlp_processor(raw_sentence, language, hyphen)
    tokenised_sent = [str(word) for word in process_sent]
    pos_sentence = []
    for i in range(len(process_sent)):
        token = process_sent[i]
        pos_sentence.append((token.lemma_.lower(), token.pos_, i))
    return tokenised_sent, pos_sentence
