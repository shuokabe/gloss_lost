#import math
import re

# Splitting functions
def text_to_line(raw_text, empty=True):
    r'''Split a raw text into a list of sentences (string) according to '\n'.'''
    split_text = re.split('\n', raw_text)
    if '' in split_text and empty: # To remove empty lines
        split_text.remove('')
    else:
        pass
    return split_text

def line_to_word(raw_line):
    '''Split a sentence into a list of words (string) according to whitespace.'''
    return re.split(' ', raw_line)

# Checking functions
def check_equality(value_left, value_right):
    '''Check that both given values are equal.'''
    assert (value_left == value_right), ('Both values must be equal; '
                         f'currently {value_left} and {value_right}.')

# Useful functions
def flatten_2D(list_of_list):
    '''Flatten a 2D list (list of list).'''
    return [element for element_list in list_of_list for element in element_list]

### New functions ###
def word_to_morpheme(raw_word):
    '''Split a word into a list of morphemes (string) according to hyphens.'''
    return re.split('-', raw_word)

# Save text file
def save_file(text, path):
    '''Save a text file in the desired path.'''
    with open(path, 'w', encoding = 'utf8') as out_text:
        out_text.write(text)

# Evaluation
# Accuracy
def compute_accuracy(label_pair_list, flat=False):
    '''Compute the accuracy of a list of label pairs (reference, predicted).

    The format of the input list can be adjusted with the flat parameter.
    flat=False if made of lists of label pairs for each sentence.'''
    if flat: # The label list is already flat
        flat_label_list = label_pair_list
    else:
        flat_label_list = flatten_2D(label_pair_list)
    match_list = [pair[0] == pair[1] for pair in flat_label_list]
    return match_list.count(True) / len(match_list)
