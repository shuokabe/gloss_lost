import argparse
import os
import numpy as np # For full output labels
import pickle
import re

from tqdm import tqdm

import gloss_lost.preprocess as pp #cld_tools.preprocess as pp
import gloss_lost.utils as utils
import gloss_lost.spacy_processing as sp #spacy_processing as sp

from gloss_lost.to_wapiti import LabelHandler, PUNCTUATION_LIST, UNIT_QTY_DICT, \
                copy_in_sentence, cap_position_in_sentence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('wapiti_filepath', type=str, help='main Wapiti file path')
    parser.add_argument('translation_filepath', type=str,
                         help='corresponding translation file path')

    parser.add_argument('--train_size', default=500, type=int,
                        help='training data size')
    parser.add_argument('--test_size', default=200, type=int,
                        help='test data size')

    parser.add_argument('--save_path', default='./', type=str,
                        help='path to save the output')
    parser.add_argument('-o', '--out_name', default='output', type=str,
                        help='output filename (base)')

    #parser.add_argument('--pos', default=False, type=bool,
    #                    help='when using complex outputs (with PoS tags)')
    parser.add_argument('--pos', action=argparse.BooleanOptionalAction,
                         help='when using complex outputs (with PoS tags)')
    #parser.add_argument('--both', action=argparse.BooleanOptionalAction,
    #                     help='when using both labels (gold and aligned)')
    parser.add_argument('--label_type', default='base', type=str,
                         choices=['base', 'pos', 'both', 'full', 'morph',
                         'dist', 'comp'],
                         help='label type (input & output)')
    parser.add_argument('--dev_size', default=0, type=int,
                        help='dev data size')

    parser.add_argument('--train_dict', default=None, #type=dict,
                        help='training dictionary for the search space')
    parser.add_argument('--punctuation', action=argparse.BooleanOptionalAction,
                        help='when punctuation marks should be predicted')
    parser.add_argument('--trg_language', default='en', type=str,
                        help='target (translation) language')
    parser.add_argument('--without_translation',
                        action=argparse.BooleanOptionalAction,
                        help='remove translation words from search space')
    parser.add_argument('--custom_stop_words', default=None, type=str,
                        help='custom stop words to remove from the translation')

    return parser.parse_args()


# Constant values: lengths of units depending on the number of outputs
LEN_ONE_OUTPUT = UNIT_QTY_DICT['base'] #4
LEN_THREE_OUTPUTS = UNIT_QTY_DICT['pos'] #LEN_ONE_OUTPUT + 2 # 6
LEN_MORPH = UNIT_QTY_DICT['morph'] #LEN_THREE_OUTPUTS + 2 # 8
LEN_FIVE_OUTPUTS = UNIT_QTY_DICT['both'] #LEN_MORPH + 2 # 8
LEN_FULL = UNIT_QTY_DICT['full'] #LEN_MORPH + 3 - 1 # 11 -1
LEN_DIST = UNIT_QTY_DICT['dist'] #LEN_MORPH + 2 + 2
LEN_COMP = UNIT_QTY_DICT['comp'] #LEN_DIST + 3
LEN_COMP_BOTH = UNIT_QTY_DICT['comp_both'] #LEN_DIST + 3


# Object to handle generic information about the data to process
class DataHandler:
    '''Object to handle generic information about the Wapiti data to process.

    Parameters
    ----------
    wapiti_file : string
        Text in the Wapiti format to convert in the Lost format
    raw_translation_file : string
        Translation corresponding to the sentences in the Wapiti file (BEFORE tokenisation)
    data_type : string
        Specify the type of the dataset (train, dev or test)
    pos : boolean
        Boolean parameter to choose the output format: one or three output labels
    label_type : LabelHandler object
        Label format for the input and output
    train_dict : dictionary {source_morpheme: lexical_label}
        Training dictionary for the search space
    punctuation : boolean
        Boolean parameter to indicate if punctuation marks have to be predicted
    without_translation : boolean
        Boolean parameter to prevent the use of translation in the search space
    stop_word_set : set
        Set of custom stop words to remove from the translation

    Attributes
    ----------
    split_text : list [unprocessed sentences (string)]
        Text split into sentences (still in a Wapiti format)
    translation : string
        Raw translation tokenised (with spaCy) (version used for the alignment)
    split_tranlsation : list [lemmatised sentences (string)]
        Translation split into a list of lemmatised sentences (with spaCy)
    both : boolean
        Boolean parameter to indicate the output format with five output labels
    '''
    def __init__(self, wapiti_file, raw_translation_file, data_type='train',
                 pos=False, label_type=None, train_dict=None, punctuation=False,
                 without_translation=False, stop_word_set=None):
        self.wapiti_file = wapiti_file
        self.raw_translation_file = raw_translation_file
        self.data_type = data_type
        self.pos = pos
        self.label_type = label_type
        self.train_dict = train_dict
        self.punctuation = punctuation
        self.without_translation = without_translation
        self.stop_word_set = stop_word_set

        # Label format
        self.both = self.label_type.both #bool(self.label_type == 'both')
        self.morph = self.label_type.morph #bool(self.label_type == 'morph')

        self.split_text = re.split('\n\n', wapiti_file)
        # Process the raw translation file
        print('Tokenising the translation file.')
        self.translation = sp.tokenise_file(self.raw_translation_file,
                                            self.label_type.language)
        # self.translation: this version of the text is used during the alignment.
        n = len(self.split_text)
        utils.check_equality(n, len(utils.text_to_line(self.translation, empty=False)))
        self.split_translation = [' '.join(sp.lemmatise_sentence(sentence,
                                                    self.label_type.language)[1])
                    for sentence in utils.text_to_line(self.translation, empty=False)] #_file)]
        utils.check_equality(n, len(self.split_translation))
        print(f'Processing {n} {self.data_type} sentences.')

        # Change functions
        if self.label_type.both:
            print('Change search space generation function')
            self.generate_search_space_file = self.generate_search_space_file_both


    # Get all the labels from a Wapiti file
    def get_all_possible_labels(self, label_position=3):
        '''Get all possible labels (4th element) to create the search space for Lost.'''
        #split_file = re.split('\n\n', wapiti_file)
        if self.label_type.label_position != 3: # TEMPORARY
            label_position = self.label_type.label_position
        all_labels_set = set()
        for sentence in self.split_text: #split_file:
            split_sentence = utils.text_to_line(sentence)
            #print(split_sentence)
            sentence_label = [re.split(' ', unit)[label_position]
                              for unit in split_sentence]
            all_labels_set.update(sentence_label)
        print(f'\tThere are {len(all_labels_set)} labels')
        return all_labels_set

    def get_all_pos_labels(self, pos_position=(LEN_THREE_OUTPUTS - 1)):
        '''Get all the possible PoS tags (for a complex output).'''
        pos_position = self.label_type.label_position + 2
        pos_tag_set = set()
        for sentence in self.split_text:
            split_sentence = utils.text_to_line(sentence)
            pos_tag = [re.split(' ', unit)[pos_position] for unit in split_sentence]
            pos_tag_set.update(pos_tag)
        #print(f'There are {len(all_labels_set)} possible labels')
        return pos_tag_set

    ### Generation of the texts ###
    def generate_reference_file(self):
        '''Generate the reference file for Lost from a Wapiti format file.

        Only for the training dataset, per definition.'''
        #split_text = re.split('\n\n', wapiti_file)
        reference_list = []
        for sentence in self.split_text:
            sentence_handler = SentenceHandler(sentence, self.label_type)
            reference_list.append(sentence_handler.generate_reference_sentence(
                                self.label_type)) #self.pos,
        reference_list.append('')
        return '\nEOS\n'.join(reference_list)

    def prepare_search_space_generation(self, all_gram_labels_set=None, test=False):
        '''Prepare the generation of the search space file.'''
        # Print the dataset type
        if test: output_label_print = '(no output label)'
        else: output_label_print = '(output labels included)'
        print(f'Creating a search space for a {self.data_type} dataset {output_label_print}')

        # Get all the grammatical labels if not given
        if all_gram_labels_set:
            assert type(all_gram_labels_set) == set, 'The given set of labels is not a set.'
            print('A label set for grammatical glosses is defined with '
                  f'{len(all_gram_labels_set)} labels.')

        # Defining the search space function for a sentence
        if self.label_type.full: # Full output labels
            self.generate_search_space_function = generate_search_space_sentence_full
        elif self.label_type.both: # Both output labels
            self.generate_search_space_function = generate_search_space_sentence_both
        elif self.label_type.comp_both: # Comp output labels
            self.generate_search_space_function = generate_search_space_sentence_comp_both
        elif self.label_type.comp: # Comp output labels
            self.generate_search_space_function = generate_search_space_sentence_comp
        elif self.label_type.dist: # Dist output labels
            self.generate_search_space_function = generate_search_space_sentence_dist
        else:
            self.generate_search_space_function = generate_search_space_sentence_pos

    def generate_search_space_file(self, all_gram_labels_set=None, test=False):
        '''Generate the search space file for Lost from a Wapiti format file.

        New version which needs the lemmatised translation.'''
        #split_text = re.split('\n\n', wapiti_file)
        #split_translation = [' '.join(sp.lemmatise_sentence(sentence)[1])
        #                     for sentence in utils.text_to_line(translation_file)]
        n = len(self.split_text)

        self.prepare_search_space_generation(all_gram_labels_set, test)

        # Get all the grammatical labels if not given
        if all_gram_labels_set:
            pass
        else:
            #all_labels_set = get_all_possible_labels(wapiti_file)
            all_gram_labels_set = filter_gram_label(
                                self.get_all_possible_labels(), self.punctuation)
            print(f'There are {len(all_gram_labels_set)} possible grammatical labels')

        search_space_list = []
        for i in tqdm(range(n)): #range(n):
            sentence, translation = self.split_text[i], self.split_translation[i]
            search_space_list.append(generate_search_space_sentence(sentence,
                        translation, all_gram_labels_set, test, self.train_dict,
                        self.punctuation, self.without_translation))
        search_space_list.append('')
        return '\nEOS\n'.join(search_space_list)

    # To get three outputs
    def get_all_possible_labels_pos(self, label_position=3):
        '''Get all possible labels with complex output labels to create the search space for Lost.

        The output will contain all the possible labels and the list of all PoS tags'''
        #label_position = self.label_type.label_position
        return (self.get_all_possible_labels(label_position=label_position),
                self.get_all_pos_labels())

    def generate_search_space_file_pos(self, all_gram_labels_set=None,
                                       pos_tag_set=None, test=False):
        '''Generate the search space file for Lost from a more complex Wapiti format file.

        New version which needs the lemmatised translation.
        For label_types: pos and full
        '''
        #split_text = re.split('\n\n', wapiti_file)
        #split_translation = [' '.join(sp.lemmatise_sentence(sentence)[1])
        #                     for sentence in utils.text_to_line(translation_file)]
        n = len(self.split_text)

        self.prepare_search_space_generation(all_gram_labels_set, test)

        # Get all the grammatical labels if not given
        if all_gram_labels_set:
            pass
        else:
            #label_position = self.label_type.label_position #3 #4
            print(f'\tLabel position: {self.label_type.label_position}')
            #all_labels_set = get_all_possible_labels(wapiti_file)
            all_gram_labels_set = filter_gram_label(
                self.get_all_possible_labels_pos(
                            label_position=self.label_type.label_position)[0],
                self.punctuation)
            print(f'There are {len(all_gram_labels_set)} possible grammatical labels')

        # Get all the pos tags if not given
        if pos_tag_set:
            assert type(pos_tag_set) == set, 'The given set of labels is actually not a set.'
            print(f'A label set for PoS tags is defined with {len(pos_tag_set)} PoS tags.')
            pass
        else:
            pos_tag_set = self.get_all_possible_labels_pos()[1]
            print(f'There are {len(pos_tag_set)} possible PoS tags.')

        # Change the sentence generation function depending on the label type
        #if self.label_type.full: # Full output labels
        #    generate_search_space_function = generate_search_space_sentence_full
        #else:
        #    generate_search_space_function = generate_search_space_sentence_pos
        ## Generate the search space sentences ##
        '''
        search_space_list = []
        #for sentence in split_text:
        split_raw_translation = utils.text_to_line(self.raw_translation_file)
        for i in tqdm(range(n)): #range(n):
            sentence, raw_translation = self.split_text[i], split_raw_translation[i]
                    #self.split_translation[i]
            search_space_list.append(self.generate_search_space_function(sentence,
                raw_translation, all_gram_labels_set, pos_tag_set,
                self.label_type, test, self.train_dict, self.punctuation))
        '''
        search_space_list = self.generate_search_space_list(
                                        all_gram_labels_set, pos_tag_set, test)
        return '\nEOS\n'.join(search_space_list)

    def generate_search_space_list(self, all_gram_labels_set, pos_tag_set, test):
        '''Generate list of sentences for the search space file.'''
        search_space_list = []
        split_raw_translation = utils.text_to_line(self.raw_translation_file, empty=False)
        for i in tqdm(range(len(self.split_text))): #range(n):
            sentence, raw_translation = self.split_text[i], split_raw_translation[i]
            search_space_list.append(self.generate_search_space_function(sentence,
                raw_translation, all_gram_labels_set, pos_tag_set,
                self.label_type, test, self.train_dict, self.punctuation,
                self.without_translation, self.stop_word_set))
        return search_space_list

    def generate_search_space_file_both(self, all_gram_labels_set=None,
                                       pos_tag_set=None, test=False):
        '''Generate the search space file for Lost from a Wapiti format file with both labels.
        '''
        n = len(self.split_text)

        self.prepare_search_space_generation(all_gram_labels_set, test)

        # Get all the grammatical labels if not given
        if all_gram_labels_set:
            pass
        else:
            label_position = 3 #self.label_type.label_position
            #all_labels_set = get_all_possible_labels(wapiti_file)
            all_gram_labels_set = filter_gram_label(
                self.get_all_possible_labels_pos(label_position=label_position)[0],
                self.punctuation)
            print(f'There are {len(all_gram_labels_set)} possible grammatical labels')

        # Get all the pos tags if not given
        if pos_tag_set:
            assert type(pos_tag_set) == set, 'The given set of labels is actually not a set.'
            print(f'A label set for PoS tags is defined with {len(pos_tag_set)} PoS tags.')
            pass
        else:
            pos_tag_set = self.get_all_possible_labels_pos()[1]
            print(f'There are {len(pos_tag_set)} possible PoS tags.')

        ## Generate the search space sentences ##
        search_space_list = self.generate_search_space_list(
                                        all_gram_labels_set, pos_tag_set, test)
        return '\nEOS\n'.join(search_space_list)


# Filtering functions
def filter_gram_label(all_label_set, punctuation=False):
    '''Keep grammatical labels only.'''
    if punctuation:
        return {label for label in all_label_set if label.isupper()}
    else: # There are punctuation marks
        return {label for label in all_label_set
                if label.isupper() and (label not in PUNCTUATION_LIST)}

def reference_label(unit_list, label_type):
    '''Convert a Wapiti unit into a Lost reference unit.'''
    if len(unit_list) != label_type.unit_length:
        print(unit_list, f'different length than {label_type.unit_length}')
    utils.check_equality(len(unit_list), label_type.unit_length)
    '''
    output_label = []

    if label_type.full: # Using full outputs
        relative_diff = unit_list.pop() # Fifth output: relative difference
        translation_position = unit_list.pop() # Fourth output: translation index
        output_label = [relative_diff, translation_position]

    if label_type.both: #unit_length == 8: # Using both labels
        #corr_tag = unit_list.pop() # Fifth output label: correspondence gold & aligned
        aligned = unit_list.pop() # Fourth output label: aligned gloss
        output_label = [aligned] #[corr_tag, aligned]

    if label_type.pos: #unit_length >= 6: # With PoS tag and gram_lex label
        pos_tag = unit_list.pop() # Third output label: PoS tag; last element in the list
        gram_lex = unit_list.pop() # Second output label: gram_lex label (gloss nature)
        output_label.extend([pos_tag, gram_lex])
    label = unit_list.pop() # Main output label; last element in the list
    output_label.append(label)
    output_label.reverse() # Reverse label order '''

    input_label = unit_list[:label_type.label_position]
    output_label = unit_list[(label_type.label_position):]

    #return f'{"|".join(unit_list)}\t{"|".join(output_label)}'
    return f'{"|".join(input_label)}\t{"|".join(output_label)}'

def reference_both_labels(unit_list, label_type):
    '''Convert a Wapiti unit into a comp_both Lost reference unit.

    There will be TWO possible references for a sentence.'''
    if len(unit_list) != label_type.unit_length:
        print(unit_list, f'different length than {label_type.unit_length}')
    utils.check_equality(len(unit_list), label_type.unit_length)

    input_label = unit_list[:label_type.label_position]
    #output_label = unit_list[(label_type.label_position):]
    ref_output_label = unit_list[(label_type.label_position):-1]
    align_output_label = unit_list[-1] + unit_list[(label_type.label_position + 1):]

    #return f'{"|".join(unit_list)}\t{"|".join(output_label)}'
    return [f'{"|".join(input_label)}\t{"|".join(ref_output_label)}',
            f'{"|".join(input_label)}\t{"|".join(align_output_label)}']


class SentenceHandler:
    '''Object to handle generic information about a Wapiti format sentence.

    Parameters
    ----------
    wapiti_sentence : string
        Sentence in the Wapiti format to convert in the Lost format
    label_type : LabelHandler object
        Label format for the input and output

    Attributes
    ----------
    split_sentence : list [unprocessed Wapiti units (string)]
        Sentence split into Wapiti units
    fully_split_sentence : list [split units (list of strings)]
        Sentence split into split Wapiti units
    sentence_length : int
        Sentence length
    '''
    def __init__(self, wapiti_sentence, label_handler=None):
        self.wapiti_sentence = wapiti_sentence
        if label_handler: self.label_type = label_handler
        else: self.label_type = LabelHandler()

        self.split_sentence = utils.text_to_line(self.wapiti_sentence)
        self.fully_split_sentence = [utils.line_to_word(line)
                                     for line in self.split_sentence]
        self.sentence_length = len(self.split_sentence)


    # Get elements from the Wapiti units
    def source_list(self):
        '''Get the list of source morphemes in the sentence.'''
        #split_sentence = [utils.line_to_word(line) for line in self.split_sentence]
        morpheme_list = [unit_list[0] for unit_list in self.fully_split_sentence]
        return morpheme_list

    # Generate a sentence for the reference file
    def generate_reference_sentence(self, label_type=None): #pos=False,
        '''Convert a Wapiti format sentence for Lost for the reference file.'''
        #split_sentence = utils.text_to_line(wapiti_sentence)
        #n = len(split_sentence)
        reference_list = []
        for i in range(self.sentence_length): #n):
            split_unit = utils.line_to_word(self.split_sentence[i])

            output_label_str = reference_label(split_unit, label_type)
            #if label_type.both: # With gold and aligned labels
                #output_label_str = reference_label(split_unit, LEN_FIVE_OUTPUTS)
            #elif pos: # With PoS tag and gram_lex label
                ###utils.check_equality(len(split_unit), LEN_THREE_OUTPUTS) #6) #7)
                ###pos_tag = split_unit.pop() # Third output label: PoS tag; last element in the list
                ###gram_lex = split_unit.pop() # Second output label: gram_lex label (gloss nature)
                ###label = split_unit.pop() # Main output label
                ###reference_list.append(
                ###f'{i}\t{i + 1}\t{"|".join(split_unit)}\t{label}|{gram_lex}|{pos_tag}')
                #output_label_str = reference_label(split_unit, LEN_THREE_OUTPUTS)
            #else: # Base label
                #utils.check_equality(len(split_unit), LEN_ONE_OUTPUT) #4) #5)
                #label = split_unit.pop() # Output label; last element in the list
                #reference_list.append(f'{i}\t{i + 1}\t{"|".join(split_unit)}\t{label}')
                #output_label_str = reference_label(split_unit, LEN_ONE_OUTPUT)
            reference_list.append(f'{i}\t{i + 1}\t{output_label_str}')
        reference_list.append(f'{i + 1}')
        #print(reference_list)
        return '\n'.join(reference_list)

    # List of all reference labels (for the training search space)
    def reference_pos_labels(self, mode='pos'):
        '''List of all the complex reference labels (for reachability or bugs).

        mode: pos (default: pos or morph), full, both.'''
        reference_list = []
        for i in range(self.sentence_length): #n):
            split_unit = utils.line_to_word(self.split_sentence[i])
            #if pos: # With PoS tag and gram_lex label
            #utils.check_equality(len(split_unit), LEN_THREE_OUTPUTS) #6) #7)
            assert len(split_unit) in \
            [LEN_THREE_OUTPUTS, LEN_MORPH, LEN_FULL, LEN_FIVE_OUTPUTS,
            LEN_DIST, LEN_COMP], \
                   f'There are {len(split_unit)} units'
            utils.check_equality(len(split_unit), self.label_type.unit_length)
            '''
            if mode == 'full':
                relative_diff = split_unit.pop() # Fifth output: relative difference
                translation_index = split_unit.pop() # Fourth output: translation index
            pos_tag = split_unit.pop() # Third output label: PoS tag; last element in the list
            gram_lex = split_unit.pop() # Second output label: gram_lex label (gloss nature)
            label = split_unit.pop() # Main output label
            #reference_list.append(f'{label}|{gram_lex}|{pos_tag}')
            '''

            if self.label_type.full: #mode == 'full':
                #reference_list.append('|'.join([label, gram_lex, pos_tag,
                #                            translation_index, relative_diff]))
                output_label_list = split_unit[(-self.label_type.output_length):] # 7 #5
            ##elif self.label_type.dist:
            ##    output_label_list = split_unit[(-self.label_type.output_length):]
            elif self.label_type.both: #mode == 'both':
                utils.check_equality(len(split_unit), LEN_FIVE_OUTPUTS)
                output_label_list = split_unit[(-self.label_type.output_length):] # 5
                #reference_list.append('|'.join(split_unit[-5:]))
            elif self.label_type.pos: #mode == 'pos': # Standard 'pos'
                #reference_list.append('|'.join([label, gram_lex, pos_tag]))
                #reference_list.append('|'.join(split_unit[-3:]))
                output_label_list = split_unit[(-self.label_type.output_length):] # 3
            else:
                raise ValueError(f'Unknown output label mode: {mode}')
            #    utils.check_equality(len(split_unit), LEN_ONE_OUTPUT) #4) #5)
            #    label = split_unit.pop() # Output label; last element in the list
            #    reference_list.append(f'{i}\t{i + 1}\t{"|".join(split_unit)}\t{label}')
            reference_list.append('|'.join(output_label_list))
        return reference_list

    '''
    def reference_both_labels(wapiti_sentence): ## TO DELETE
        ###List of all the reference (both) labels (for reachability or bugs).###
        sentence = SentenceHandler(wapiti_sentence)
        reference_list = []
        for i in range(sentence.sentence_length): #n):
            split_unit = utils.line_to_word(sentence.split_sentence[i])
            #if pos: # With PoS tag and gram_lex label
            utils.check_equality(len(split_unit), LEN_FIVE_OUTPUTS)
            reference_list.append('|'.join(split_unit[-5:]))
        return reference_list'''

    def labels_from_training_pos(self, dictionary, mode='pos', #both=False, full=False,
                                 punctuation=False, copy_index_list=[]): #label_type=None):
        '''Get some more possible labels based on the training dictionary.

        Modes:
        - pos: PoS case, three outputs
        - full: using all outputs (five outputs)
        - both: using both labels, five outputs
        In the both case, the label order is different:
            In the train dictionary, morpheme: (aligned, frequency, pos, reference).
        copy_index_list: list of all copied units
        '''
        label_list = []
        output_list = []
        #print(self.fully_split_sentence)
        for line in self.fully_split_sentence: #fully_split_wapiti_sentence:
            morpheme = line[0]
            if morpheme in dictionary:
                label_tuple = dictionary[morpheme]
                if punctuation and (morpheme in PUNCTUATION_LIST): # If punctuation
                    continue
                elif label_tuple[0].isupper(): # If grammatical label
                    continue
                else:
                    main_label = label_tuple[0]
                    if self.label_type.full: #mode == 'full': # Full labels
                        if len(label_tuple) == 3: # MORPH dictionary
                            main_label = label_tuple[0]
                        else: # FULL dictionary
                            corres = str(int(label_tuple[3] == label_tuple[0]))
                            main_label = label_tuple[3]
                        # label_tuple[3] if FULL dictionary, with MORPH: 0
                        output_list = [main_label, 'lex', label_tuple[2],
                                       #label_tuple[0], corres, '-1', 'D']
                                       '?', '-1', '-1', 'D']
                        #corres = '-1' # TEST
                        #output_list = [label_tuple[0], 'lex', label_tuple[2],
                        #               label_tuple[0], corres, '-1', 'D'
                        #               ] # TEST
                        #output_list = [label_tuple[0], 'lex', label_tuple[2],
                        #               '-1', 'D'] # '-1'
                    elif self.label_type.comp:
                        copy_trg = copy_in_sentence(main_label, self.source_list())
                        # if (copy_trg == '1') and copy_index_list:
                        #     copy_trg = morph_in_copy_indices_list(main_label,
                        #                                         copy_index_list)
                        #     copy_index_list[(copy_trg - 1)] = ('#DONE#', 0)
                        position_trg = '-1' # No target index from dictionary
                        output_list = [main_label, 'lex', label_tuple[2],
                                    str(copy_trg), position_trg] #, 'D']
                    elif self.label_type.dist: # Copy and position labels
                        #main_label = label_tuple[0]
                        copy_trg = copy_in_sentence(main_label, self.source_list())
                        position_trg = '-1' # No target index from dictionary
                        output_list = [main_label, 'lex', label_tuple[2],
                                    copy_trg, position_trg] #, 'D']
                    elif self.label_type.both: #mode == 'both': #label_type.both: # Both labels
                        corres = str(int(label_tuple[3] == label_tuple[0]))
                        output_list = [label_tuple[3], 'lex', label_tuple[2],
                                       label_tuple[0], corres]
                    elif self.label_type.pos: #mode == 'pos': # Standard PoS output label
                        output_list = [label_tuple[0], 'lex', label_tuple[2]]
                    else:
                        raise ValueError(f'Unknown output label mode: {mode}')
                    #print(morpheme, '|'.join(output_list))
                    #label_list.append(f'{label_tuple[0]}|lex|{label_tuple[2]}')
                label_list.append('|'.join(output_list))
        print(morpheme, '|'.join(output_list))
        return label_list

    def search_space_labels(self, possible_label_set): #, label_type):
        '''Generate all possible labels for the search space.'''
        sorted_possible_label = sorted(list(possible_label_set))
        #print(sorted_possible_label)
        search_space_list = []
        for i in range(self.sentence_length):
            split_unit = utils.line_to_word(self.split_sentence[i])
            split_input = split_unit[:self.label_type.label_position]
            utils.check_equality(len(split_input), self.label_type.label_position)
            input_label_str = "|".join(split_input)
            for label in sorted_possible_label:
                search_space_list.append(f'{i}\t{i + 1}\t{input_label_str}\t{label}')
        search_space_list.append(f'{i + 1}')
        return search_space_list


# Convert lexical label output into a complex label output
def unknown_complex_lex_output(translation_list, pos_tag_set): ## TO DELETE? unused
    '''Convert a list of translation words into a set of complex output labels.

    Former function for the search space.'''
    output_set = {f'{word}|lex|{pos_tag}' for word in translation_list
                  for pos_tag in pos_tag_set}
    print('unknown_complex_lex_output', output_set)
    return output_set

def complex_lex_output(raw_translation, pos_tag_set, label_type, #mode='pos',
                       capitalise=False, source_list=[], stop_word_set=None,
                       copy_index_list=[]): #translation_list
    '''Convert a raw translation sentence into a set of complex output labels.

    label_type: LabelHandler object
    Several modes of output labels:
    - pos: basic complex output: aligned gloss|binary category|PoS tag
    - both: with the translated words: aligned gloss|binary|PoS tag|aligned gloss
    - full: with full labels:
            aligned gloss|binary|PoS tag|translation index|relative difference
    pos_tag_sent parameter obsolete
    '''
    #label_type = LabelHandler(mode)
    trg_lang = label_type.language # Translation language
    capitalise = True
    #'''Convert a list of translation words into a set of complex output labels.'''
    pos_list = sp.sentence_pos_tag(raw_translation, trg_lang)[1]
    #print(raw_translation, pos_list)
    lemmatised_sentence = sp.lemmatise_sentence(raw_translation, trg_lang)[1]
    lemma_translation_list = lemmatised_sentence #utils.line_to_word(lemmatised_sentence)
    n = len(lemma_translation_list)
    utils.check_equality(n, len(pos_list))
    # Lemma AND PoS
    lemmatised_a_sentence = sp.lemmatise_sentence_for_alignment(raw_translation, trg_lang)[1]
    #print(raw_translation, lemmatised_a_sentence)
    #output_set = {f'{word}|lex|{pos_tag}' for word in translation_list for pos_tag in pos_tag_set}
    def duplicate_with_cap(lemma_list, pos_tag_list, n):
        '''Create a set with the same elements but with a capital letter.'''
        output_set = {f'{lemma_translation_list[i].title()}|lex|{pos_list[i][1]}'
                for i in range(n) #if len(lemma_translation_list[i]) > 5}
                if pos_list[i][1] == 'PROPN'} # Just for proper nouns
        return output_set

    output_list = []
    for i in range(n):
        translation_lemma = lemma_translation_list[i]
        if stop_word_set and (translation_lemma in stop_word_set):
            continue
        pos_tag = pos_list[i][1]
        utils.check_equality(lemmatised_a_sentence[i][0], translation_lemma)
        utils.check_equality(lemmatised_a_sentence[i][1], pos_tag)
        if capitalise and (pos_list[i][1] == 'PROPN'): # Capitalise proper nouns
            translation_lemma = translation_lemma.capitalize() #title()
        label_list = [translation_lemma, 'lex', pos_tag]

        if label_type.both: #mode == 'both':
            label_list.append(translation_lemma)
            output_list.append('|'.join(label_list))
        elif label_type.full: #mode == 'full':
            label_list.extend([translation_lemma, '1'])
            joined_label = '|'.join(label_list)
            count_in_trg = min(lemmatised_sentence.count(translation_lemma), 1)
            for j in ['-1', '0.0', 0.1, 0.2, '0.3+']:
            #for j in ['-1', '0.0', '0.1+']:
                output_list.append(f'{joined_label}|{j}|T{count_in_trg}')
            #output_list.append(f'{joined_label}|0|T{count_in_trg}') #'T'
        elif label_type.dist:
            #if translation_lemma.capitalize() == 'Stockholm':
            #    print(source_list, translation_lemma)
            copy_trg = copy_in_sentence(translation_lemma, source_list)
            # if label_type.comp and (copy_trg == '1'): # and copy_index_list:
            #     copy_trg = 1
                # Cancel copy update
                # copy_trg = morph_in_copy_indices_list(translation_lemma,
                #                                     copy_index_list)
                # print('dist: copy_trg', copy_trg, translation_lemma)
                # if copy_trg >= 0:
                #     copy_index_list[(copy_trg - 1)] = ('#DONE#', 0)
                # else: # copy_trg = -1
                #     pass
            # else: # label_type.dist
            #     copy_trg = 0
            label_list.append(str(copy_trg))
            relative_trg_pos = i / n
            label_list.append(cap_position_in_sentence(relative_trg_pos)) # position_trg
            count_in_trg = min(lemmatised_sentence.count(translation_lemma), 1)
            #if label_type.comp: # Add origin in the output
            #    label_list.append('T') #f'T{count_in_trg}')
            output_list.append(('|'.join(label_list)))
        elif label_type.pos: #mode == 'pos': # Base case
            output_list.append(('|'.join(label_list)))
        else:
            raise ValueError(f'Unknown output label mode: {label_type.label_type}')
            #mode}')

    '''
    if mode == 'both':
        output_set = {f'{lemma_translation_list[i]}|lex|{pos_list[i][1]}|'
                      f'{lemma_translation_list[i]}|1' for i in range(n)}
    elif mode == 'full':
        output_set = {f'{lemma_translation_list[i]}|lex|{pos_list[i][1]}|'
                      #f'{i}|{j}|T' for i in range(n) #{round(j, 1)
                      f'{lemma_translation_list[i]}|1|'
                      f'{j}|T' for i in range(n) #{round(j, 1)
                      #for j in np.arange(0.0, 1.0, 0.1)}
                      for j in ['0.0', 0.1, 0.2, '0.3+']}
    elif mode == 'pos': # Base case
        output_set = {f'{lemma_translation_list[i]}|lex|{pos_list[i][1]}'
                      for i in range(n)}
        #output_set = output_set.union(
        #        duplicate_with_cap(lemma_translation_list, pos_list, n))
        #print(f'{lemma_translation_list[n].title()}|lex|{pos_list[n][1]}')
    else:
        raise ValueError(f'Unknown output label mode: {mode}')
    '''
    output_set = set(output_list)
    #print(output_set)
    return output_set

# When using a training dictionary to have more search space labels
def labels_from_training(fully_split_wapiti_sentence, dictionary, gold=False,
                         punctuation=False):
    '''Get some more possible labels based on the training dictionary.'''
    label_list = []
    for line in fully_split_wapiti_sentence:
        morpheme = line[0]
        if morpheme in dictionary:
            #if type(dictionary[morpheme][0]) == str: # List of all possible glosses
            if gold: # List of all possible glosses
                label_list.extend(dictionary[morpheme])
            #elif type(dictionary[morpheme][0]) == tuple: # Gloss, frequency case
            else:
                #print(morpheme, dictionary[morpheme][0])
                label_list.append(dictionary[morpheme][0])
            #else:
            #    raise TypeError(f'Unknown dictionary type {dictionary[morpheme]}')
    return label_list


# Generate a sentence for the search space file
def generate_search_space_sentence(wapiti_sentence, translation, gram_labels_set,
                                   test=False, train_dict=None,
                                   punctuation=False, without_translation=False):
    '''Convert a Wapiti format sentence for Lost for the search space file.

    With lexemes in the ouput. Extract only the words in the translation and the output.
    If test is True, the output labels are not included in the search space.'''
    split_sentence = [utils.line_to_word(sent) for sent in utils.text_to_line(wapiti_sentence)]
    split_translation = utils.line_to_word(translation)
    n = len(split_sentence)

    # List of all possible labels for the sentence
    if test: # No output label possible
        additional_inference_label_set = {'and', 'be', 'become', 'get', 'let'}
        # Find some additional labels based on the training dictionary
        possible_label_set = gram_labels_set.union(split_translation)
        possible_label_set = possible_label_set.union({'?'})
        #possible_label_set = possible_label_set.union(additional_inference_label_set)

        if train_dict: # There is a training dictionary
            training_label = labels_from_training(split_sentence, train_dict,
                                punctuation=punctuation)
            possible_label_set = possible_label_set.union(set(training_label))
    else: # Training: include output label for reachability -> not needed anymore?
        possible_label_set = gram_labels_set.union(split_translation,
                                        [line[-1] for line in split_sentence])
    #possible_label_set = possible_label_set.union({'?'}) # Add the unknown label
    #print(possible_label_set) #, split_translation, [line[-1] for line in split_sentence])
    sorted_possible_label = sorted(list(possible_label_set))
    search_space_list = []
    for i in range(n):
        split_unit = split_sentence[i] #utils.line_to_word(split_sentence[i])
        utils.check_equality(len(split_unit), LEN_ONE_OUTPUT) #4) #5)
        split_unit.pop() # Output label; last element in the list
        #reference_list.append(f'{i}\t{i + 1}\t{"|".join(split_unit)}\t{label}')
        for label in sorted_possible_label: #possible_label_set:
            search_space_list.append(f'{i}\t{i + 1}\t{"|".join(split_unit)}\t{label}')
    search_space_list.append(f'{i + 1}')
    return '\n'.join(search_space_list)

def filter_lex_label(all_label_set):
    '''Keep lexical labels only.'''
    return {label for label in all_label_set if not label.isupper()}

# Generate a sentence for the search space file with a more complex output
def generate_search_space_sentence_pos(wapiti_sentence, raw_translation,
        gram_labels_set, pos_tag_set, label=None, test=False, train_dict=None,
        punctuation=False, without_translation=False, stop_word_set=None):
    '''Convert a Wapiti format sentence for Lost for the search space file with a richer output label.

    The input translation LIST must contain the pos tag for each word
    With lexemes in the ouput. Extract only the words in the translation and the output.
    If test is True, the output labels are not included in the search space.'''
    sentence = SentenceHandler(wapiti_sentence, label)
    split_sentence = sentence.fully_split_sentence
    #[utils.line_to_word(sent) #for sent in utils.text_to_line(wapiti_sentence)]
                        #for sent in sentence.split_sentence]
    #split_translation = utils.line_to_word(translation)

    ## List of all possible labels for the sentence ##
    # Grammatical labels
    output_gram_labels_set = {f'{gram_label}|gram|GRAM_GLOSS'
                              for gram_label in gram_labels_set}
    # Punctuation
    source_morpheme_list = sentence.source_list()
    if punctuation:
        for morph in source_morpheme_list:
            if morph in PUNCTUATION_LIST:
                #punctuation_list.append(morph])
                output_gram_labels_set.add(f'{morph}|gram|PUNCT')
    #print(output_gram_labels_set)
    # Lexical labels from the translation
    translation_label_set = complex_lex_output(raw_translation, pos_tag_set,
                                        label, source_list=source_morpheme_list,
                                        stop_word_set=stop_word_set)
    possible_label_set = output_gram_labels_set
    if not without_translation:
        possible_label_set = output_gram_labels_set.union(translation_label_set)

    if test: # No output label possible
        # Add more
        additional_inference_label_set = {'and|lex|CCONJ', 'be|lex|AUX',
                                'become|lex|VERB', 'get|lex|VERB', 'let|lex|VERB'}
        ##possible_label_set = gram_labels_set.union(split_translation)
        #possible_label_set = output_gram_labels_set.union(
        #    complex_lex_output(raw_translation, pos_tag_set,
        #                        source_list=source_morpheme_list))
        #possible_label_set = possible_label_set.union({'?|lex|?'})
        ##possible_label_set = possible_label_set.union(additional_inference_label_set)

        if train_dict: # There is a training dictionary
            #training_label = labels_from_training_pos(split_sentence, train_dict,
            training_label = sentence.labels_from_training_pos(train_dict,
                                                punctuation=punctuation)
            possible_label_set = possible_label_set.union(set(training_label))
    else: # Training: include output label for reachability -> not needed anymore?
        #possible_label_set = gram_labels_set.union(split_translation,
        #                        [line[-1] for line in split_sentence])
        #possible_label_set = output_gram_labels_set.union(
        #    complex_lex_output(split_translation, pos_tag_set),
        #   complex_lex_output(filter_lex_label([line[4]
        #    for line in split_sentence]), pos_tag_set)) # line[3]
        #possible_label_set = output_gram_labels_set.union(
        possible_label_set = possible_label_set.union(
            #complex_lex_output(raw_translation, pos_tag_set,
            #                source_list=source_morpheme_list),
                            set(sentence.reference_pos_labels(mode='pos'))
                            )
                #unknown_complex_lex_output(filter_lex_label(
                #[line[LEN_THREE_OUTPUTS - 3] for line in split_sentence]), pos_tag_set)) # line[3]
        # If some words are added from another source, put all known pos tags in the search space
    #print(possible_label_set) #, split_translation, [line[-1] for line in split_sentence])

    # Generate the search space labels
    search_space_list = sentence.search_space_labels(possible_label_set) #, label)
    return '\n'.join(search_space_list)

# Generate a sentence for the search space file with both labels in the output
def generate_search_space_sentence_both(wapiti_sentence, raw_translation,
            gram_labels_set, pos_tag_set, label=None, test=False, train_dict=None,
            punctuation=False, without_translation=False, stop_word_set=None):
    '''Convert a Wapiti format sentence for Lost for the search space file with a richer output label.

    The input translation LIST must contain the pos tag for each word
    With lexemes in the ouput. Extract only the words in the translation and the output.
    If test is True, the output labels are not included in the search space.'''
    sentence = SentenceHandler(wapiti_sentence, label)
    split_sentence = [utils.line_to_word(sent)
            for sent in sentence.split_sentence] #utils.text_to_line(wapiti_sentence)]
    #split_translation = utils.line_to_word(translation)
    n = len(split_sentence)

    # List of all possible labels for the sentence
    output_gram_labels_set = {f'{gram_label}|gram|GRAM_GLOSS|GRAM_GLOSS|-2' #{gram_label}'
                              for gram_label in gram_labels_set}
    if punctuation:
        output_punct_label = {f'{punct_label}|gram|PUNCT|PUNCT_GLOSS|-2'
                              for punct_label in PUNCTUATION_LIST}
        output_gram_labels_set.union(output_punct_label)
    #print(output_gram_labels_set)
    if test: # No output label possible
        # Add more
        #additional_inference_label_set = {}
        possible_label_set = output_gram_labels_set.union(
        #both_lex_output(raw_translation, pos_tag_set))
                complex_lex_output(raw_translation, pos_tag_set, label)) #mode='both'))
        possible_label_set = possible_label_set.union({'?|lex|?|?|-1'})
        #possible_label_set = possible_label_set.union(additional_inference_label_set)

        if train_dict: # There is a training dictionary
            training_label = sentence.labels_from_training_pos(train_dict,
                                            mode='both', punctuation=punctuation)
            possible_label_set = possible_label_set.union(set(training_label))
    else: # Training: include output label for reachability -> not needed anymore?
        possible_label_set = output_gram_labels_set.union(
                            #both_lex_output(raw_translation, pos_tag_set),
                complex_lex_output(raw_translation, pos_tag_set, label), #mode='both'),
                set(sentence.reference_pos_labels(mode='both')))
                #unknown_complex_lex_output(filter_lex_label(
                #[line[LEN_THREE_OUTPUTS - 3] for line in split_sentence]), pos_tag_set)) # line[3]
        # If some words are added from another source, put all known pos tags in the search space
    #print(possible_label_set) #, split_translation, [line[-1] for line in split_sentence])

    # Generate the search space labels
    search_space_list = sentence.search_space_labels(possible_label_set) #, label)
    return '\n'.join(search_space_list)

# Generate a sentence for the search space file with full output labels
def generate_search_space_sentence_full(wapiti_sentence, raw_translation,
                    gram_labels_set, pos_tag_set, label=None, test=False,
                    train_dict=None, punctuation=False,
                    without_translation=False, stop_word_set=None):
    '''Convert a Wapiti format sentence for Lost for the search space file with a full output label.

    The input translation LIST must contain the pos tag for each word
    With lexemes in the ouput. Extract only the words in the translation and the output.
    If test is True, the output labels are not included in the search space.'''
    sentence = SentenceHandler(wapiti_sentence, label)
    split_sentence = sentence.fully_split_sentence
    #[utils.line_to_word(sent) for sent in sentence.split_sentence]
    #split_translation = utils.line_to_word(translation)
    #n = len(split_sentence)
    #utils.check_equality(n, sentence.sentence_length) # TO REMOVE TEMPORARY

    ## List of all possible labels for the sentence ##
    # Grammatical labels
    output_gram_labels_set = {f'{gram_label}|gram|GRAM_GLOSS|GRAM_GLOSS|-2|-1|G'
                            #{f'{gram_label}|gram|GRAM_GLOSS|-1|-1|G'
                              for gram_label in gram_labels_set}
    # Punctuation
    if punctuation:
        output_punct_label = {f'{punct_label}|gram|PUNCT|PUNCT_GLOSS|-2|-1|G'
                            #{f'{punct_label}|gram|PUNCT|-1|-1|G'
                              for punct_label in PUNCTUATION_LIST}
        output_gram_labels_set.union(output_punct_label)
    #print(output_gram_labels_set)
    # Lexical labels from the translation
    translation_label_set = complex_lex_output(raw_translation, pos_tag_set,
                                               label, stop_word_set=stop_word_set) #mode='full')
    possible_label_set = output_gram_labels_set
    if not without_translation:
        possible_label_set = output_gram_labels_set.union(translation_label_set)
    if test: # No output label possible
        # Add more
        #additional_inference_label_set = {'and|lex|CCONJ', 'be|lex|AUX',
        #                        'become|lex|VERB', 'get|lex|VERB', 'let|lex|VERB'}
        #possible_label_set = output_gram_labels_set.union(
        #        complex_lex_output(raw_translation, pos_tag_set, mode='full'))
        #possible_label_set = possible_label_set.union({'?|lex|?|-1|-1|?'})
        possible_label_set = possible_label_set.union({'?|lex|?|?|-1|-1|?'})
        #possible_label_set = possible_label_set.union(additional_inference_label_set)

        if train_dict: # There is a training dictionary
            #training_label = labels_from_training_pos(split_sentence, train_dict,
            training_label = sentence.labels_from_training_pos(train_dict,
                                            mode='full', #full=True,
                                            punctuation=punctuation)
            filtered_dict_label_set = filter_full_labels(
                                    translation_label_set, set(training_label))
            #possible_label_set = possible_label_set.union(set(training_label))
            possible_label_set = possible_label_set.union(filtered_dict_label_set)
    else: # Training: include output label for reachability -> not needed anymore?
        ##possible_label_set = gram_labels_set.union(split_translation,
        ##                        [line[-1] for line in split_sentence])
        ##possible_label_set = output_gram_labels_set.union(
        ##    complex_lex_output(split_translation, pos_tag_set),
        ##   complex_lex_output(filter_lex_label([line[4]
        ##    for line in split_sentence]), pos_tag_set)) # line[3]
        #possible_label_set = output_gram_labels_set.union(
        #            complex_lex_output(raw_translation, pos_tag_set, mode='full'),
        #            set(sentence.reference_pos_labels(mode='full'))
        #                    )
        reference_label_set = set(sentence.reference_pos_labels(mode='full'))
        filtered_ref_label_set = filter_full_labels(
                                    translation_label_set, reference_label_set)
        possible_label_set = possible_label_set.union(reference_label_set)
                                                    #filtered_ref_label_set)

                #unknown_complex_lex_output(filter_lex_label(
                #[line[LEN_THREE_OUTPUTS - 3] for line in split_sentence]), pos_tag_set)) # line[3]
        # If some words are added from another source, put all known pos tags in the search space
    #print(possible_label_set) #, split_translation, [line[-1] for line in split_sentence])

    # Generate the search space labels
    search_space_list = sentence.search_space_labels(possible_label_set) #, label)
    return '\n'.join(search_space_list)

def filter_full_labels(translation_label_set, ref_dict_label_set, mode='full'):
    '''Filter the possible labels in the full setting.

    When the same label is available from the translation and the dictionary,
    use the translation label. NOT FINISHED'''
    label_handler = LabelHandler(mode)
    filtered_ref_dict_set = set()
    main_translation_label_set = {(re.split('[|]', unit)[0], re.split('[|]', unit)[2])
                                  for unit in translation_label_set}
    for label in ref_dict_label_set:
        split_label = re.split('[|]', label)
        if len(split_label) != label_handler.output_length: 
            print('filter_full_labels', split_label)
        utils.check_equality(len(split_label), label_handler.output_length) # 7
        # If already in the translation
        if (split_label[0], split_label[2]) in main_translation_label_set:
            #print(f'{split_label[0]}, {split_label[2]} already in the translation')
            pass
        else:
            filtered_ref_dict_set.add(label)
    #print(filtered_ref_dict_set)
    return filtered_ref_dict_set

# Generate a sentence for the search space file with full output labels
def generate_search_space_sentence_dist(wapiti_sentence, raw_translation,
                    gram_labels_set, pos_tag_set, label=None, test=False,
                    train_dict=None, punctuation=False,
                    without_translation=False, stop_word_set=None):
    '''Convert a Wapiti format sentence for Lost for the search space file with a dist output label.

    The input translation LIST must contain the pos tag for each word
    With lexemes in the ouput. Extract only the words in the translation and the output.
    If test is True, the output labels are not included in the search space.'''
    sentence = SentenceHandler(wapiti_sentence, label)
    split_sentence = sentence.fully_split_sentence

    ## List of all possible labels for the sentence ##
    # Grammatical labels
    output_gram_labels_set = {f'{gram_label}|gram|GRAM_GLOSS|-1|-2' #|G'
                              for gram_label in gram_labels_set}
    # Punctuation
    source_morpheme_list = sentence.source_list()
    if punctuation:
        for morph in source_morpheme_list:
            if morph in PUNCTUATION_LIST:
                #punctuation_list.append(morph])
                output_gram_labels_set.add(f'{morph}|gram|PUNCT|1|-2')
                #output_gram_labels_set.add(f'{morph}|gram|PUNCT|-1|-2|G')
        #output_punct_label = {f'{punct_label}|gram|PUNCT|-1|-2'
        #                      for punct_label in PUNCTUATION_LIST}
        #output_gram_labels_set.union(output_punct_label)
    #print(output_gram_labels_set)
    # Lexical labels from the translation
    translation_label_set = complex_lex_output(raw_translation, pos_tag_set,
                                label, #mode='dist',
                                source_list=source_morpheme_list,
                                stop_word_set=stop_word_set)
    possible_label_set = output_gram_labels_set
    if not without_translation:
        possible_label_set = output_gram_labels_set.union(translation_label_set)
    if test: # No output label possible
        # Add more
        #possible_label_set = possible_label_set.union({'?|lex|?|-1|-1'})

        if train_dict: # There is a training dictionary
            training_label = sentence.labels_from_training_pos(train_dict,
                                            mode='dist',
                                            punctuation=punctuation)
            filtered_dict_label_set = filter_full_labels(
                        translation_label_set, set(training_label), mode='dist')
            #possible_label_set = possible_label_set.union(set(training_label))
            possible_label_set = possible_label_set.union(filtered_dict_label_set)
    else: # Training: include output label for reachability -> not needed anymore?
        reference_label_set = set(sentence.reference_pos_labels(mode='dist'))
        #filtered_ref_label_set = filter_full_labels(
        #                            translation_label_set, reference_label_set)
        possible_label_set = possible_label_set.union(reference_label_set)

    # Generate the search space labels
    search_space_list = sentence.search_space_labels(possible_label_set)
    return '\n'.join(search_space_list)

def copy_indices(wapiti_sentence, label):
    '''Keep track of copy indices (for dist, comp).

    Format: [(morpheme, copy_index)]'''
    sentence = SentenceHandler(wapiti_sentence, label)
    copy_index_list = []
    check_index = 1
    for unit_list in sentence.fully_split_sentence:
        copy_index = int(unit_list[5]) # Source copy index (copy_src)
        if copy_index > 0:
            copy_index_list.append((unit_list[0], copy_index))
            print('copy_indices unit list', check_index, unit_list)
            utils.check_equality(check_index, copy_index)
            check_index += 1
    return copy_index_list

def morph_in_copy_indices_list(morpheme, copy_indices_list):
    '''Get the index of a morpheme in the copy indices list.'''
    copied_morphemes = [unit[0].lower() for unit in copy_indices_list]
    if morpheme.lower() in copied_morphemes:
        return copied_morphemes.index(morpheme.lower())
    else:
        print('morph_in_copy_indices_list', copied_morphemes, morpheme)
        return -1

# Generate a sentence for the search space file with comp output labels
def generate_search_space_sentence_comp(wapiti_sentence, raw_translation,
                    gram_labels_set, pos_tag_set, label=None, test=False,
                    train_dict=None, punctuation=False,
                    without_translation=False, stop_word_set=None):
    '''Convert a Wapiti format sentence for Lost for the search space file with a comp output label.

    The input translation LIST must contain the pos tag for each word
    With lexemes in the ouput. Extract only the words in the translation and the output.
    If test is True, the output labels are not included in the search space.'''
    sentence = SentenceHandler(wapiti_sentence, label)
    split_sentence = sentence.fully_split_sentence

    ## List of all possible labels for the sentence ##
    # Grammatical labels
    output_gram_labels_set = {f'{gram_label}|gram|GRAM_GLOSS|-1|-2'#|G'
                              for gram_label in gram_labels_set}

    copy_index_list = [] # copy_indices(wapiti_sentence, label) # Cancel copy update
    if copy_index_list != []: print('before processing copy', copy_index_list)
    # Punctuation
    source_morpheme_list = sentence.source_list()
    if punctuation:
        for morph in source_morpheme_list:
            if morph in PUNCTUATION_LIST:
                copy_src_i = morph_in_copy_indices_list(morph, copy_index_list)
                #output_gram_labels_set.add(f'{morph}|lex|?|1|-1')
                copy_src_i = 0 # Cancel copy update
                output_gram_labels_set.add(f'{morph}|lex|?|{copy_src_i + 1}|-1')
                # if copy_src_i >= 0:
                #     #print(copy_src_i, copy_index_list[copy_src_i])
                #     copy_index_list[(copy_src_i - 1)] = ('#DONE#', 0)
                ##output_gram_labels_set.add(f'{morph}|gram|PUNCT|-1|-2|G')
        #output_punct_label = {f'{punct_label}|gram|PUNCT|-1|-2'
        #                      for punct_label in PUNCTUATION_LIST}
        #output_gram_labels_set.union(output_punct_label)
    # Add numbers
    numbers = True
    if numbers:
        for morph in source_morpheme_list:
            if morph.isdigit():
                copy_src_i = morph_in_copy_indices_list(morph, copy_index_list)
                copy_src_i = 0 # Cancel copy update
                output_gram_labels_set.add(f'{morph}|lex|?|{copy_src_i + 1}|-2')
                # if copy_src_i >= 0:
                #     print('comp_numbers', copy_src_i, copy_index_list[copy_src_i])
                #     copy_index_list[(copy_src_i - 1)] = ('#DONE#', 0)
    #print(output_gram_labels_set)

    #if copy_index_list != []: print('after punct', copy_index_list)
    # Lexical labels from the translation
    translation_label_set = complex_lex_output(raw_translation, pos_tag_set,
                                label, #mode='comp',
                                source_list=source_morpheme_list,
                                stop_word_set=stop_word_set,
                                copy_index_list=copy_index_list)
    if copy_index_list != []: print('after done', copy_index_list)
    if test: print('TR COMP', translation_label_set)
    possible_label_set = output_gram_labels_set
    if not without_translation:
        possible_label_set = output_gram_labels_set.union(translation_label_set)
    if test: # No output label possible
        # Add more
        #possible_label_set = possible_label_set.union({'?|lex|?|-1|-1'})

        if train_dict: # There is a training dictionary
            training_label = sentence.labels_from_training_pos(train_dict,
                                            mode='comp',
                                            punctuation=punctuation,
                                            copy_index_list=copy_index_list)
            filtered_dict_label_set = filter_full_labels(
                        translation_label_set, set(training_label), mode='comp')
            #possible_label_set = possible_label_set.union(set(training_label))
            possible_label_set = possible_label_set.union(filtered_dict_label_set)
    else: # Training: include output label for reachability -> not needed anymore?
        reference_label_set = set(sentence.reference_pos_labels(mode='comp'))
        #filtered_ref_label_set = filter_full_labels(
        #                            translation_label_set, reference_label_set)
        possible_label_set = possible_label_set.union(reference_label_set)

    # Generate the search space labels
    search_space_list = sentence.search_space_labels(possible_label_set)
    return '\n'.join(search_space_list)

# Generate a sentence for the search space file with comp_both output labels
def generate_search_space_sentence_comp_both(wapiti_sentence, raw_translation,
                    gram_labels_set, pos_tag_set, label=None, test=False,
                    train_dict=None, punctuation=False, 
                    without_translation=False, stop_word_set=None):
    '''Convert a Wapiti format sentence for Lost for the search space file with a comp output label.

    The input translation LIST must contain the pos tag for each word
    With lexemes in the ouput. Extract only the words in the translation and the output.
    If test is True, the output labels are not included in the search space.'''
    sentence = SentenceHandler(wapiti_sentence, label)
    split_sentence = sentence.fully_split_sentence

    ## List of all possible labels for the sentence ##
    # Grammatical labels
    output_gram_labels_set = {f'{gram_label}|gram|GRAM_GLOSS|-1|-2'#|G'
                              for gram_label in gram_labels_set}
    # Punctuation
    source_morpheme_list = sentence.source_list()
    if punctuation:
        for morph in source_morpheme_list:
            if morph in PUNCTUATION_LIST:
                #punctuation_list.append(morph])
                output_gram_labels_set.add(f'{morph}|gram|PUNCT|1|-2')
                #output_gram_labels_set.add(f'{morph}|gram|PUNCT|-1|-2|G')
        #output_punct_label = {f'{punct_label}|gram|PUNCT|-1|-2'
        #                      for punct_label in PUNCTUATION_LIST}
        #output_gram_labels_set.union(output_punct_label)
    #print(output_gram_labels_set)
    # Lexical labels from the translation
    translation_label_set = complex_lex_output(raw_translation, pos_tag_set,
                                label, #mode='comp',
                                source_list=source_morpheme_list,
                                stop_word_set=stop_word_set)
    if not without_translation:
        possible_label_set = output_gram_labels_set.union(translation_label_set)
    if test: # No output label possible
        # Add more
        #possible_label_set = possible_label_set.union({'?|lex|?|-1|-1'})

        if train_dict: # There is a training dictionary
            training_label = sentence.labels_from_training_pos(train_dict,
                                            mode='comp',
                                            punctuation=punctuation)
            filtered_dict_label_set = filter_full_labels(
                        translation_label_set, set(training_label), mode='comp')
            #possible_label_set = possible_label_set.union(set(training_label))
            possible_label_set = possible_label_set.union(filtered_dict_label_set)
    else: # Training: include output label for reachability -> not needed anymore?
        reference_label_set = set(sentence.reference_pos_labels(mode='comp'))
        #filtered_ref_label_set = filter_full_labels(
        #                            translation_label_set, reference_label_set)
        possible_label_set = possible_label_set.union(reference_label_set)

    # Generate the search space labels
    search_space_list = sentence.search_space_labels(possible_label_set)
    return '\n'.join(search_space_list)

## Functions to split a text
# Split the dataset in two: train and test
def split_wapiti_file(wapiti_file, train_size, test_size, dev_size=0):
    '''Split a Wapiti file into a training (dev) and test dataset.'''
    split_file = re.split('\n\n', wapiti_file)
    train_list, dev_list, test_list = split_text_list(split_file, train_size,
                                                      test_size, dev_size)
    def wapiti_join(sent_list):
        return '\n\n'.join(sent_list)
    return map(wapiti_join, (train_list, dev_list, test_list))
    #'\n\n'.join(train_list), '\n\n'.join(dev_list), '\n\n'.join(test_list)

def split_file(file, train_size, test_size, dev_size=0):
    '''Split a standard file into a training (dev) and test dataset.'''
    split_file = utils.text_to_line(file)
    train_list, dev_list, test_list = split_text_list(split_file, train_size,
                                                      test_size, dev_size)
    return '\n'.join(train_list), '\n'.join(dev_list), '\n'.join(test_list)

def split_text_list(text_list, train_size, test_size, dev_size=0):
    '''Split a text (LIST format) into a LIST of train, dev, and test sentences.'''
    n = len(text_list)
    train_list = text_list[0:train_size]
    test_list = text_list[(n - test_size):]
    print(f'{len(train_list)} training and {len(test_list)} test sentences')
    # Development dataset
    #dev_list = text_list[train_size:(train_size + dev_size)]
    dev_list = text_list[-(test_size + dev_size):-test_size]
    if dev_list != []:
        print(f'{len(dev_list)} developement sentences')
    return train_list, dev_list, test_list

#def one_line_processing(train_wapiti, train_translation, test_wapiti,
#                        test_translation, save_path, exp_name, pos=False):
def one_line_processing(wapiti_file, translation, train_size, test_size,
                        save_path, exp_name, pos=False, label_type='base',
                        dev_size=0, train_dict=None, punctuation=False,
                        trg_language='en', without_translation=False,
                        custom_stop_words=None):
    '''Temporary function to run the whole process (one output label)'''
    # Indicate the label format
    label = LabelHandler(label_type, language=trg_language)
    label.print_setting()
    utils.check_equality(pos, label.pos) # Check -> to delete afterwards

    # Split the texts into train and test
    train_wapiti, dev_wapiti, test_wapiti = split_wapiti_file(
                                wapiti_file, train_size, test_size, dev_size)
    train_translation, dev_translation, test_translation = split_file(
                                translation, train_size, test_size, dev_size)

    # Defined custom stop words to remove from the translation
    if custom_stop_words: # Defined custom stop words to remove from the translation
        stop_word_set = set(utils.text_to_line(custom_stop_words))
    else:
        stop_word_set = set()

    # Process the split texts
    train_data = DataHandler(train_wapiti, train_translation, data_type='train',
                             pos=pos, label_type=label, train_dict=train_dict,
                             punctuation=punctuation,
                             without_translation=without_translation,
                             stop_word_set=stop_word_set)
    test_data = DataHandler(test_wapiti, test_translation, data_type='test',
                            pos=pos, label_type=label, train_dict=train_dict,
                            punctuation=punctuation,
                            without_translation=without_translation,
                            stop_word_set=stop_word_set)
    if (dev_wapiti == '') and (dev_translation == ''):
        dev = False
        print('No development dataset')
    else:
        dev = True
        dev_data = DataHandler(dev_wapiti, dev_translation, data_type='dev',
                        pos=pos, label_type=label, train_dict=train_dict,
                        punctuation=punctuation,
                        without_translation=without_translation,
                        stop_word_set=stop_word_set)
        print(f'Development dataset of size: {len(dev_data.split_text)}')
    gapl_train = train_data.get_all_possible_labels()
    if label.pos: # Complex output
        print('\tComplex output with three labels.')
        gapl_train_pos_tag = train_data.get_all_pos_labels() #train_wapiti)
        print('All pos labels', gapl_train_pos_tag)
    all_gram_label_train_set = filter_gram_label(gapl_train, punctuation)

    # Generate the texts
    lost_train_ref = train_data.generate_reference_file()

    if label.both: #label_type == 'both': # Both labels in output
        train_search_space = train_data.generate_search_space_file()
        test_search_space = test_data.generate_search_space_file(
                        all_gram_label_train_set, gapl_train_pos_tag, test=True)
        if dev: # Development data
            dev_search_space = dev_data.generate_search_space_file( #_both(
                        all_gram_label_train_set, gapl_train_pos_tag, test=True)
    elif label.pos: # Complex output
        train_search_space = train_data.generate_search_space_file_pos()
        test_search_space = test_data.generate_search_space_file_pos(
                        all_gram_label_train_set, gapl_train_pos_tag, test=True)
        if dev: # Development data
            dev_search_space = dev_data.generate_search_space_file_pos(
                        all_gram_label_train_set, gapl_train_pos_tag, test=True)
    else:
        train_search_space = train_data.generate_search_space_file()
        test_search_space = test_data.generate_search_space_file(
                                            all_gram_label_train_set, test=True)
        if dev: # Development data
            dev_search_space = dev_data.generate_search_space_file(
                                            all_gram_label_train_set, test=True)


    # Save the generated files
    utils.save_file(lost_train_ref, os.path.join(save_path, f'train_{exp_name}.ref'))

    utils.save_file(train_search_space, os.path.join(save_path, f'train_{exp_name}.spc'))
    utils.save_file(test_search_space, os.path.join(save_path, f'test_{exp_name}.spc'))

    # If there is a development data
    if dev:
        utils.save_file(dev_search_space, os.path.join(save_path, f'dev_{exp_name}.spc'))

    return 0


def main():
    args = parse_args()

    if args.label_type == 'both': # With the gold and aligned labels
        args.pos = True
        print('!!!!!!!!!Output with five labels.')
    elif args.label_type == 'full':
        args.pos = True
        print('!!!!!!!!!Six inputs and five output labels.')
    elif args.label_type == 'morph':
        args.pos = True
        print('!!!!!!!!!Five inputs and three output labels.')
    elif args.label_type == 'dist':
        args.pos = True
        print('!!!!!!!!!DIST, TO DELETE .')
    elif args.label_type == 'comp':
        args.pos = True
        print('!!! COMP')
        if not args.train_dict: # No training dictionary specified
            raise ValueError('A training dictionary must be specified')
    elif args.pos: # Complex output
        print('!!!!!!Output with three labels.')

    # Using punctuation
    if args.punctuation:
        print('Predicting punctuations')
    if args.trg_language != 'en':
        print(f'The target translation is in {args.trg_language}.')

    # Read the datasets
    wapiti_file = open(args.wapiti_filepath, 'r', encoding = 'utf8').read()
    translation = open(args.translation_filepath, 'r', encoding = 'utf8').read()

    # Using a training dictionary
    if args.train_dict:
        print(f'The search space will take into account a training dictionary.')
        with open(args.train_dict, 'rb') as f:
            train_dict = pickle.load(f)
        if args.pos or (args.label_type in ['both', 'morph', 'full', 'dist']): # temp
            print('Check the inference labels: code??') # temp
    else:
        train_dict = None

    # Using a custom list of stop words
    if args.custom_stop_words:
        custom_stop_words = open(
                        args.custom_stop_words, 'r', encoding = 'utf8').read()
    else:
        custom_stop_words = None

    # Process the files
    one_line_processing(wapiti_file, translation, args.train_size,
                        args.test_size, args.save_path, args.out_name,
                        pos=args.pos, label_type=args.label_type,
                        dev_size=args.dev_size, train_dict=train_dict,
                        punctuation=args.punctuation,
                        trg_language=args.trg_language,
                        without_translation=args.without_translation,
                        custom_stop_words=custom_stop_words)


if __name__ == "__main__":
    main()
