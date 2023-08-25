import collections
import re


# Open a file easily
def open_file(file_path):
    '''Open a file easily'''
    return open(file_path, 'r').read()

# Preprocessing by removing unnecessary characters
def remove_excessive_whitespace(string):
    '''Remove excessive whitespace.'''
    return re.sub(' +', ' ', string)

def remove_punctuation(string):
    '''Remove punctuation, except hyphens.'''
    new_string = re.sub(r'[!¡?\\:;…,.\[\]()<>«»‹›"„“]', '', string)
    return new_string

def remove_common_special_characters(string):
    '''Remove some special characters.'''
    return re.sub(r'[+_@&*]', '', string)

def remove_emojis(string):
    '''Remove emojis.'''
    emoji = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        #u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoji, '', string)

# Replace some characters
def replace_specific_space(string):
    '''Replace specific whitespaces with the standard whitespace.'''
    return re.sub(r'[\u2005\u2009\u200a\u200f\u202f\ufeff]', ' ', string)

# Simple pre-processing
def simple_preprocess(text):
    '''Carry out the most common steps of pre-processing.'''
    # Remove punctuation
    pp_text = remove_punctuation(text)
    pp_text = remove_common_special_characters(pp_text)
    # Remove digits
    pp_text = re.sub('\d+', '', pp_text)
    # Replace specific whitespaces
    pp_text = replace_specific_space(pp_text)
    # Lowercase
    pp_text = pp_text.lower()
    return pp_text

# Character inventory
def character_collection(string):
    '''Create a dictionary with all the characters in the text.'''
    return collections.Counter(string)

def find_pattern(text, pattern, start=0):
    '''Find a specific pattern in the text, starting at a specific index.'''
    index = text.find(pattern, start)
    print(index)
    print(text[(index - 1000):(index + 1000)])
    return index

# Save pre-processed file
def save_file(text, path):
    '''Save a file in a desired path.'''
    with open(path, 'w', encoding = 'utf8') as out_text:
        out_text.write(text)
