import argparse
import re

from collections import Counter
from tqdm import tqdm

import gloss_lost.utils as utils
import spacy_processing as sp

### Create a corpus in a Wapiti format, which will then be converted for Lost ###

# Constants for the lexical gloss and PoS tags
ALIGNED_LEMMA_POSITION = 2
POS_POSITION = ALIGNED_LEMMA_POSITION + 1
TRG_INDEX_POSITION = POS_POSITION + 1

# Number of units per Wapiti line
UNIT_QTY = 4
UNIT_QTY_POS = UNIT_QTY + 2 # 6
UNIT_QTY_MORPH = UNIT_QTY_POS + 2 # 8
UNIT_QTY_BOTH = UNIT_QTY_MORPH + 2 # 10
UNIT_QTY_FULL = UNIT_QTY_MORPH + 3 - 1 + 2 #+ 2 # 10 # 11
UNIT_QTY_DIST = UNIT_QTY_MORPH + 2 + 2 #+ 1 # 12 + 1
UNIT_QTY_DICT = {'base': UNIT_QTY, 'pos': UNIT_QTY_POS, 'morph': UNIT_QTY_MORPH,
                  'both': UNIT_QTY_BOTH, 'full': UNIT_QTY_FULL,
                  'dist': UNIT_QTY_DIST
                 }

# Punctuation list
PUNCTUATION_LIST = [',', r'.', '«', '»,', '»', '».', '?', '...', ':',
                    '?»', '.»', '"', ',»', '(', '…', ')', r'."', '?)',
                    '?"', '!', '???']


class LabelHandler:
    '''Indicate the label format (input and output).

    Parameters
    ----------
    label_type : string
        Label type (base, pos, gold, both, morph)
    use_gold : bool
        Using reference gloss as main output label
    language : string
        Language of the labels

    Attributes
    ----------
    pos : bool
        Using additional output labels: binary tag and PoS tag
    morph : bool
        Using morphological information (initial and final letters) as input.
    both : bool
        Using the reference label and a correspondence tag (if aligned = gold).
    full : bool
        Using all inputs and outputs (morph + position)
    dist : bool
        Latest method: using copy features and categorised position features
    unit_length : int
        Number of units (inputs & outputs)
    output_length : int
        Number of outputs
    label_position : int
        Position (index) of the main label to predict
    '''
    def __init__(self, label_type='base', use_gold=False, language='en'):
        self.label_type = label_type

        # Label types
        self.pos = bool(label_type == 'pos')
        # self.gold
        self.morph = bool(label_type == 'morph') # Using morphological features
        self.both = bool(label_type == 'both') # Both labels
        self.full = bool(label_type == 'full') # Using all inputs and outputs
        self.dist = bool(label_type == 'dist') # Using copy and position features

        self.use_gold = use_gold # Using gold reference as main output label
        self.language = language

        if self.full: self.both = True
        if self.full or self.both or self.dist: self.morph = True
        if self.morph: self.pos = True

        # Unit length
        if self.full:
            self.unit_length = UNIT_QTY_FULL # 10 + 2
        elif self.dist:
            self.unit_length = UNIT_QTY_DICT['dist'] # 12?
        elif self.both:
            self.unit_length = UNIT_QTY_BOTH # 10
        elif self.morph:
            self.unit_length = UNIT_QTY_MORPH # 8
        elif self.pos:
            self.unit_length = UNIT_QTY_POS # 6
        else:
            self.unit_length = UNIT_QTY # 4

        # Number of outputs
        #BASE_OUTPUT_LEN = 1
        #STRUCT_OUTPUT_LEN = BASE_OUTPUT_LEN + 2 # 3
        #BOTH_OUTPUT_LEN = STRUCT_OUTPUT_LEN + 2 # 5
        #FULL_OUTPUT_LEN = STRUCT_OUTPUT_LEN + 2 + 2 # 7
        if self.full:
            self.output_length = 7 # label_position = 5
        elif self.dist:
            self.output_length = 5 #+ 1 # label_position = 7
        elif self.both:
            self.output_length = 5 # label_position = 5
        elif self.pos :
            self.output_length = 3 # label_position = 3 (pos) or 5 (morph)
        else:
            self.output_length = 1 # label_position = 3

        # Label position
        self.label_position = self.unit_length - self.output_length



    def print_setting(self):
        '''Display configuration information.'''
        # Explain the label type
        if self.both: print(f'Using both aligned and gold lexical glosses')
        if self.dist: print(f'Using copy and position features')
        if self.morph: print(f'Using morphological features in input and gold labels')
        if self.full: print(f'Using all inputs and outputs')
        if self.use_gold: print('Using reference gloss as main output label')

        # Display number of outputs
        if self.both: print(f'There are four outputs.')
        if self.morph: print(f'There are five inputs and three outputs.')
        if self.full: print(f'There are five inputs and all outputs (CHANGE).')
        if self.dist: print(f'There are seven inputs and five outputs (CHANGE).')

        # Print unit length and label position
        print(f'{self.unit_length} units (label position: {self.label_position})')


# Process a corpus
class Corpus:
    '''Process a corpus made of sentences.

    Parameters
    ----------
    source : string
        Text in the studied language (segmented in words and morphemes)
    gloss : string
        Glossed text with the same number of units as the source
    translation : string
        Translation of the source text (raw, not tokenised)
    test : bool
        If it is a test data, there will be dummy outputs

    Attributes
    ----------
    split_source : list [sentences (string)]
        Source text split into sentences
    split_gloss : list [sentences (string)]
        Glossed text split into sentences
    split_translation : list [sentences (string)]
        Translation text split into sentences
    n_sent : int
        Number of sentences in the corpus
    sentence_list : list (Sentence)
        Processed sentence with the Sentence object
    label_type : LabelFormat object
        Label format for the input and output (generated with to_wapiti_format)
    '''
    def __init__(self, source, gloss, translation, test=False):
        self.source = source
        self.gloss = gloss
        self.translation = translation
        self.test = test

        self.split_source = utils.text_to_line(source)
        self.split_gloss = utils.text_to_line(gloss)
        self.split_translation = utils.text_to_line(translation)
        self.n_sent = len(self.split_source)
        utils.check_equality(self.n_sent, len(self.split_translation))
        if test: self.split_gloss = [''] * self.n_sent # No test gloss
        utils.check_equality(self.n_sent, len(self.split_gloss))
        print(f'The corpus has {self.n_sent} sentences.')

        # Process each sentence
        self.sentence_list = []
        for i in range(self.n_sent):
            self.sentence_list.append(Sentence(self.split_source[i],
                            self.split_gloss[i], self.split_translation[i],
                            self.test))


    def to_wapiti_format(self, alignment, expand=False, pos=False, verbose=False,
                         train_index=0, gold=False, label_format='base',
                         punctuation=False, use_gold=False, language='en',
                         hyphen=True):
        '''Convert a corpus into a Wapiti format file to use with Lost.

        label_type:
        gold: use the reference gloss label.
        both: use both the reference gloss label (main) and the aligned label.'''
        # Describe the corpus creation setting
        train = bool(train_index > 0) # Use of training dictionary?

        self.label_type = LabelHandler(label_format, use_gold, language)
        self.label_type.print_setting()
        pos = self.label_type.pos

        if self.label_type.morph: #abel_type.both or
            #print(f'Using both aligned and gold lexical glosses')
            pos = True
            if train:
                print(f' creating a dictionary with {train_index} sentences')
            gold = True
        elif gold:
            print(f'Using gold lexical glosses only')
            if train: print(f' creating a dictionary with {train_index} sentences')
        elif expand:
            print('Using expanded labels')
            if train: print(f' with {train_index} sentences')

        utils.check_equality(pos, self.label_type.pos) ## To remove at some point
        # Number of labels
        #if self.label_type.both: print(f'There are five outputs.')
        if self.label_type.both or self.label_type.morph \
            or self.label_type.full or self.label_type.dist: pass
        elif self.label_type.pos: print(f'There are three outputs.')
        else: print(f'There is one output.')

        # Process the alignment file
        if alignment: # Not None: standard case
            split_alignment = utils.text_to_line(alignment, empty=False)
            utils.check_equality(self.n_sent, len(split_alignment))
            test = False
        else: # For test (or development) files
            assert not train, 'An alignment file is needed for the training part.'
            test = True

        if expand and train: # Use a training dataset
            self.train_dict = self.create_train_dictionary(alignment, train_index,
                            verbose=verbose) #,
                            #label_type=self.label_type)
        elif self.label_type.both and train: # Use a training dataset with both labels
            self.train_dict = self.create_train_dictionary(alignment, train_index, #,
                            verbose=verbose) #,
                            #label_type=self.label_type)
        elif gold and train: # Create the gold dictionary
            print('Use gold training dictionary')
            self.train_dict = self.create_gold_dictionary(train_index)
        else:
            self.train_dict = None

        wapiti_sent_list = []
        for i in tqdm(range(self.n_sent)):
            #print(split_source[i], split_translation[i], split_gloss[i], split_alignment[i])
            sentence = self.sentence_list[i]
            #wapiti_sent_list.append(to_wapiti_sentence(split_source[i], split_translation[i],
            #                                           split_gloss[i], split_alignment[i], pos))
            if test or self.test: # No alignment file
                wapiti_line = sentence.to_wapiti_test_sentence(
                    self.label_type.pos, expand, verbose=verbose,
                    gold=gold, label_type=self.label_type)
            else:
                wapiti_line = sentence.to_wapiti_sentence(
                    split_alignment[i], pos=self.label_type.pos, expand=expand,
                    verbose=verbose, train_dict=self.train_dict, gold=gold,
                    label_type=self.label_type, punctuation=punctuation,
                    hyphen=hyphen)
            wapiti_sent_list.append(wapiti_line)

        # Gold labels with PoS tags
        if train and ((gold and self.label_type.pos) or self.label_type.morph) \
            and not (self.label_type.both or self.label_type.full): ##label_type.dist???
            print('Use a hybrid dictionary with projected PoS tags')
            self.train_dict = self.create_hybrid_pos_train_dictionary(
                        wapiti_sent_list, train_index)
        return '\n\n'.join(wapiti_sent_list)

    def align_lex_whole_text_with_source(self, alignment, expand_alignment=False,
                                         punctuation=False, verbose=False,
                                         hyphen=True):
        '''Align the lexical glosses of the whole corpus.

        The source morpheme is kept.
        This function is used to create the training dictionary.'''
        split_align = utils.text_to_line(alignment, empty=False)
        utils.check_equality(self.n_sent, len(split_align))

        aligned_lex_gloss_list = []
        for i in range(self.n_sent):
            #print(f'Processing sentence {i}')
            alignment_sent = split_align[i]
            sentence = self.sentence_list[i]
            reference_aligned_gloss = sentence.reference_and_aligned_gloss_with_source(
                        alignment_sent, expand_alignment=expand_alignment,
                        verbose=verbose, punctuation=punctuation,
                        language=self.label_type.language, hyphen=hyphen)
            aligned_lex_gloss_list.append(reference_aligned_gloss)
        return aligned_lex_gloss_list

    def create_train_dictionary(self, alignment, train_index=200,
                                use_reference=False, verbose=False): #, label_type=None):
        '''Create a dictionary of label based on a training dataset.

        Format of the output dictionary (standard; both=False):
            source_morpheme: (most_frequent_label, frequency, pos).
        Format of the output dictionary (both=True):
            source_morpheme: (most_frequent_label, frequency, pos, reference_gloss).
        '''
        # First pass on the text
        aligned_gloss_list = self.align_lex_whole_text_with_source(alignment,
                            expand_alignment=True, verbose=verbose, hyphen=True)

        # Process the training part of the dataset
        FREQ_POS = 1 # Position of frequency in the dictionary
        if self.label_type.both: # Using both labels # &full
            print(f'Using a train dataset of {train_index} sentences (both labels).')
        else: # Standard case
            print(f'Using a train dataset of {train_index} sentences.')
        if use_reference: print(' + using references when no alignment found')

        train = aligned_gloss_list[:train_index]
        train_dict = dict()
        # Constitute a dictionary from the training part
        flat_train = [(unit[0], unit[1], unit[2], unit[3])
                      for unit in utils.flatten_2D(train)]
        count_train = Counter(flat_train)
        #print(count_train)
        for (morpheme, reference, aligned, pos), frequency in count_train.items():
        #    for (morpheme, reference, aligned) in sentence:
            if morpheme in train_dict: # Already seen
                # 1st case: Get all the possibilities
                #train_dict[(morpheme, reference)].append((aligned, frequency))
                # 2nd case: Use the majority label
                current_freq = train_dict[morpheme][FREQ_POS]
                # A more frequent label is found
                if (current_freq < frequency) and (aligned not in ['', '?']):
                    if self.label_type.both: # &full
                        train_dict[morpheme] = (aligned, frequency, pos, reference)
                    else:
                        train_dict[morpheme] = (aligned, frequency, pos)
            else: # New morpheme
                if aligned not in ['', '?']: #!= '':
                    if self.label_type.both: # &full
                        train_dict[morpheme] = (aligned, frequency, pos, reference)
                    else:
                        train_dict[morpheme] = (aligned, frequency, pos)
                else: # Morpheme that has not been aligned yet => use reference?
                    #print(f'Using the reference for {morpheme}')
                    if self.label_type.use_gold or use_reference:
                        if self.label_type.both: # &full
                            train_dict[morpheme] = (reference, frequency, pos, reference)
                        else:
                            train_dict[morpheme] = (reference, frequency, pos)
        return train_dict

    def create_gold_dictionary(self, train_index=-1):
        '''Create a dictionary made of all seen gold glosses.'''
        morph_gloss_list = []
        for i in range(train_index):
            source_sent = re.split('[ -]', self.split_source[i])
            gloss_sent = re.split('[ -]', self.split_gloss[i])
            m = len(source_sent)
            utils.check_equality(m, len(gloss_sent))

            pair_list = [(source_sent[j], gloss_sent[j]) for j in range(m)]
            morph_gloss_list.extend(pair_list)

        # Count the morpheme-gloss pairs
        morph_gloss_count = Counter(morph_gloss_list)
        train_dict = dict()
        for (morpheme, gloss), frequency in morph_gloss_count.items():
            if morpheme in train_dict: # Already seen morpheme
                # Gloss cannot be already seen in this case (unique pair)
                train_dict[morpheme].append(gloss)
            else: # New morpheme
                train_dict[morpheme] = [gloss]
        return train_dict

    def create_hybrid_pos_train_dictionary(self, split_wapiti_file, train_index=200):
        '''Create a dictionary of gold labels with PoS tags based on a training dataset.

        Format of the output dictionary:
            source_morpheme: (most_frequent_label, frequency, pos).'''
        #split_file = re.split('\n\n', wapiti_file)

        # Process the training part of the dataset
        print(f'Using a train dataset of {train_index} sentences.')

        split_train = [utils.text_to_line(sentence)
                       for sentence in split_wapiti_file[:train_index]]
        train_dict = dict()
        # Constitute a dictionary from the training part
        flat_train = utils.flatten_2D(split_train)

        label_postition = self.label_type.label_position
        pos_position = label_postition + 2
        new_unit_list = []
        for unit in flat_train:
            split_unit = utils.line_to_word(unit)
            utils.check_equality(len(split_unit), self.label_type.unit_length)
            #assert len(split_unit) in [6, 8, 11], f'There are {len(split_unit)} units'
            #new_unit_list.append((split_unit[0], split_unit[3], split_unit[5]))
            new_unit_list.append((split_unit[0], split_unit[label_postition],
                                      split_unit[pos_position]))
            #new_unit_list.append((split_unit[0], split_unit[-3], split_unit[-1]))

        count_train = Counter(new_unit_list)
        #print(count_train)
        for (morpheme, gloss, pos), frequency in count_train.items():
            if gloss.isupper(): # Grammatical gloss
                pass #continue
            #else:
            if morpheme in train_dict: # Already seen LEXICAL gloss
                # 1st case: Get all the possibilities
                #train_dict[(morpheme, reference)].append((aligned, frequency))
                # 2nd case: Use the majority label
                current_freq = train_dict[morpheme][1]
                # A more frequent label is found
                if (current_freq < frequency) and (gloss not in ['', '?']):
                    train_dict[morpheme] = (gloss, frequency, pos)
            else: # New morpheme
                if gloss not in ['', '?']: #!= '':
                    train_dict[morpheme] = (gloss, frequency, pos)
                else: # Morpheme that has not been aligned yet => use reference?
                    pass
                    #if use_reference:
                    #    train_dict[morpheme] = (reference, frequency, pos)
        return train_dict


# Process sentences
class Sentence:
    '''Object to handle one sentence and its annotations.

    Parameters
    ----------
    source : string
        Sentence in the studied language (segmentation in words and morphemes)
    gloss : string
        Glossed sentence with the same number of units as the source
    translation : string
        Translation of the source sentence (raw, not tokenised)
    test : bool
        If it is a test data, there will be dummy outputs

    Attributes
    ----------
    split_source : list [morphemes (string)]
        Source sentence split into morphemes
    split_gloss : list [glosses (string)]
        Glosses split at the morpheme level
    split_translation : list [words (string)]
        Translation split into a list of words
    n_morph : int
        Number of morphemes in the source sentence (or number of glosses)
    '''
    def __init__(self, source, gloss, translation, test=False):
        self.source = source
        self.gloss = gloss
        self.translation = translation

        self.split_source = re.split('[ -]', self.source)
        self.split_gloss = re.split('[ -]', self.gloss)
        self.split_translation = utils.line_to_word(self.translation)
        self.n_morph = len(self.split_source)
        if not test: utils.check_equality(self.n_morph, len(self.split_gloss))


    def morpheme_position_list(self, biof=False):
        '''Create a list containing the position of a morpheme in a word.

        Format (biof=False): [(source_morpheme, position_index)]
        Format (biof=True): [(source_morpheme, BIOF_position_tag)]
        '''
        split_source_with_hyphen = word_to_morpheme_decomp(self.source)
        #print(split_source_with_hyphen)
        position_list = []
        position_counter = 0
        for i in range(len(split_source_with_hyphen) - 1):
            morpheme = split_source_with_hyphen[i]
            if morpheme == '-':
                position_counter += 1
                continue
            position_list.append((morpheme, position_counter))
            if split_source_with_hyphen[(i + 1)] == '-': # Word with several morphemes
                continue
            else: # End of word
                position_counter = 0
        # Last element
        if (len(split_source_with_hyphen) > 1) and (split_source_with_hyphen[-2] == '-'): # Still in a word
            position_list.append((split_source_with_hyphen[-1], position_counter))
        else: # New word
            position_list.append((split_source_with_hyphen[-1], 0))
        utils.check_equality(self.n_morph, len(position_list))
        #print('position: ', position_list)

        # Use BIOF tags instead of numbers
        # B: begining of word, I: internal morpheme, O: end of word,
        # and F: free morpheme
        if biof:
            new_position_list = []
            i = -1 # In case there is only one morpheme
            for i in range(self.n_morph - 1): # = len(position_list)
                morpheme = position_list[i][0]
                if position_list[i][1] == 0:
                    if position_list[(i + 1)][1] == 0: # Free morpheme F
                        new_position_list.append((morpheme, 'F'))
                    else: # Initial morpheme B
                        new_position_list.append((morpheme, position_list[i][1])) #'B'))
                else:
                    # Internal morpheme I
                    if position_list[(i + 1)][1] == (position_list[i][1] + 1):
                        new_position_list.append((morpheme, position_list[i][1]))#'I'))
                    else: # Final morpheme O
                        new_position_list.append((morpheme, position_list[i][1]))#'O'))
            final_pair = position_list[(i + 1)]
            if final_pair[1] == 0: # Free morpheme F
                new_position_list.append((final_pair[0], 'F'))
            else: # Final morpheme O
                new_position_list.append((final_pair[0], final_pair[1]))
                #position_list[i][1])) #'O'))
            return new_position_list
        else:
            return position_list

    # Create the Wapiti unit for a sentence
    def to_wapiti_sentence(self, alignment, pos=False, expand=False,
                           verbose=False, train_dict=None, gold=False,
                           label_type=None, punctuation=False, hyphen=True):
        '''Create from the four inputs for a sentence its corresponding Wapiti unit.

        Output for each morpheme:
            new version: morpheme morpheme_position_in_word morpheme_length output_label(s)
        label_type (LabelHandler object):
        - base: basic entry (source, position, and length) + gloss label.
        - pos: include the PoS tag of the translated word in the output label.
            + an output label for the category of the grammatical and lexical glosses (gram_lex).
        - both: include the reference label and a correspondence tag (aligned = gold).

        expand: boolean parameter to add more 'alignments'
                based on the presence in the translation.
        train_dict: dictionary to complete the label of some morphemes
                    when already seen in the training dataset
        gold: use gold label for lexical glosses
        both: use gold label + aligned label as output labels
        '''
        utils.check_equality(pos, label_type.pos) ## To remove at some point
        # Prepare the source and the gloss which have the same number of morphemes/units
        # For the grammatical glosses
        # For a Wapiti unit with morpheme boundaries
        split_source_with_hyphen = word_to_morpheme_decomp(self.source)
        #print(split_source_with_hyphen)
        # Use BIO morpheme position tags
        if label_type.full or label_type.both or label_type.morph: # label_type.dist
            morpheme_position = self.morpheme_position_list(biof=True)
        else:
            morpheme_position = self.morpheme_position_list(biof=False)

        # Prepare the other part: lexical glosses, the translation and the alignment
        #lex_gloss = self.only_lex_gloss_sent()
        #split_alignment = string_to_pair_list(alignment)
        aligned_lex_gloss = self.reference_and_aligned_gloss_with_source(
                        alignment, expand_alignment=expand, verbose=verbose,
                        train_dict=train_dict, punctuation=punctuation,
                        language=label_type.language, hyphen=hyphen)
        if verbose: print(aligned_lex_gloss)
        #if expand:
        #    print('Find some more gloss alignments using the training dataset.')
        #    aligned_lex_gloss = complete_with_seen_morpheme(aligned_lex_gloss,
        #            train_index=train_index, use_reference=False, verbose=False)
        #lex_gloss_aligned(self.translation, self.gloss, alignment, pos)
        self.split_translation = utils.line_to_word(self.translation)
        translation_length = len(self.split_translation)
        #print(self.split_source, '\n', self,split_translation, '\n',
        #self.split_gloss, '\n', lex_gloss, '\n', split_alignment,
        #      '\n', aligned_lex_gloss)
        # To get the correct relative difference value
        tokenised_translation, lemmatised_translation = \
        sp.lemmatise_sentence_for_alignment(self.translation,
                                            label_type.language, hyphen)
        translation_length = len(tokenised_translation)

        wapiti_unit_list = []
        index = 0 # Index for all morphemes (i.e. without hyphen)
        j = 0 # Index for lexical gloss

        for i in range(len(split_source_with_hyphen)):
            morpheme = split_source_with_hyphen[i] # Source morpheme
            if morpheme == '-':
                pass
            else:
                length = len(morpheme) # Morpheme length
                if label_type.morph:
                    morph_init, morph_end = extract_letters(morpheme, length)
                cap_length = cap_morpheme_length(length, cap_val=5)

                gloss = self.split_gloss[index]
                position_pair = morpheme_position[index]
                utils.check_equality(morpheme, position_pair[0]) # Check that the morphemes do match
                position = position_pair[1]
                if gloss.isupper(): # If it is a grammatical unit
                    label = gloss
                    # If three output labels
                    if label_type.pos: # With PoS tag
                        gram_lex = 'gram'
                        aligned_pos = 'GRAM_GLOSS'
                    if label_type.both: # With the reference label + correspondence tag
                        reference = label
                        label = 'GRAM_GLOSS' # Trial
                        correspondence = '-2' #'-1' #'1'
                    origin = 'G'
                    if label_type.full: # With full inputs and outputs
                        output_index_str = '-1'
                        relative_diff = '-1'
                        #origin = 'G'
                    if label_type.dist: # Copy and position feature
                        copy_trg = '-1'
                        position_trg = '-2'
                elif punctuation and (morpheme in PUNCTUATION_LIST):
                    label = gloss
                    gram_lex = 'gram'
                    aligned_pos = 'PUNCT'
                    if label_type.both: # With the reference label + correspondence tag
                        reference = label
                        label = 'PUNCT_GLOSS' # Trial
                        correspondence = '-2'
                    origin = 'G'
                    if label_type.full: # With full inputs and outputs
                        output_index_str = '-1'
                        relative_diff = '-1'
                        #origin = 'G' # 'P'
                    if label_type.dist: # Copy and position feature
                        copy_trg = '1' #'-1'
                        position_trg = '-1'
                else: # Lexical unit (or hybrid unit)
                    #align_index = aligned_lex_gloss[j][1]
                    #aligned_lemma = aligned_lex_gloss[j][2]
                    label = aligned_lex_gloss[j][ALIGNED_LEMMA_POSITION]
                    # Target (aligned) position
                    #if label in lemmatised_translation:
                    #    trg_index = lemmatised_translation.index(label) # For dist labels
                    #else: # Not aligned
                    #    trg_index = -1
                    trg_index = aligned_lex_gloss[j][TRG_INDEX_POSITION]
                    # Use gold lexical gloss
                    if (gold or label_type.use_gold) and not label_type.both:
                        label = gloss
                    # More complex output label
                    if label_type.pos: # With PoS tag
                        gram_lex = 'lex'
                        aligned_pos = aligned_lex_gloss[j][POS_POSITION]
                    if label_type.both: #With the reference label + correspondence tag
                        reference = gloss
                        correspondence = f'{int(label == gloss)}'
                    if label_type.full:
                        # Aligned index
                        output_index_str = str(aligned_lex_gloss[j][-1])
                        relative_diff = relative_position_difference(
                                index, self.n_morph, int(output_index_str),
                                translation_length)
                        relative_diff = cap_relative_difference(relative_diff) #
                        #relative_diff = str(round(relative_diff, 1)) #2))
                        if label == gloss: # != '?':
                            count_in_trg = min(lemmatised_translation.count(label), 1)
                            origin = f'T{count_in_trg}' #'T'
                        else:
                            origin = 'D'
                    if label_type.dist: # Copy and position feature
                        #copy_bool = int(bool(label.title() == morpheme.title()))
                        copy_trg = copy_in_sentence(label, split_source_with_hyphen)
                        if punctuation and (morpheme in PUNCTUATION_LIST):
                            copy_trg = str(1)
                        #trg_index = lemmatised_translation.index(label)
                        relative_trg_pos = max(trg_index / translation_length, -1)
                        position_trg = cap_position_in_sentence(relative_trg_pos)
                        if label == gloss: # != '?':
                            count_in_trg = min(lemmatised_translation.count(label), 1)
                            origin = f'T{count_in_trg}' #'T'
                        else:
                            origin = 'D'
                    j += 1

                # Basic inputs
                unit_list = [morpheme, str(position), cap_length] #str(length)]
                if label_type.full:
                    unit_list.extend([morph_init, morph_end,
                    #unit_list = [morpheme, str(position), cap_length,
                                 #str(index),
                                 reference, #label,
                                 gram_lex, aligned_pos, label, correspondence,
                                 #output_index_str,
                                 relative_diff, origin]) #])
                elif label_type.dist:
                    copy_src = copy_in_sentence(morpheme, tokenised_translation)
                    relative_src_pos = index / self.n_morph
                    position_src = cap_position_in_sentence(relative_src_pos)
                    seen = 0 # Change: seen in dict?
                    unit_list.extend([morph_init, morph_end,
                                 copy_src, position_src,
                                 label, gram_lex, aligned_pos,
                                 copy_trg, position_trg]) #, origin])
                elif label_type.both: # With the reference label
                    #unit_list = [morpheme, str(position), str(length),
                    unit_list.extend([morph_init, morph_end,
                            reference, gram_lex, aligned_pos, label,
                            correspondence])
                elif label_type.morph:
                    #unit_list = [morpheme, str(position), cap_length, #str(length),
                    unit_list.extend([morph_init, morph_end,
                                 label, gram_lex, aligned_pos])
                    #utils.check_equality(len(unit_list), UNIT_QTY_BOTH)
                elif label_type.pos: # With PoS tag
                    unit_list = [morpheme, str(position), str(length),
                                 label, gram_lex, aligned_pos]
                    #utils.check_equality(len(unit_list), UNIT_QTY_POS)
                else:
                    #unit_list.append(label)
                    unit_list = [morpheme, str(position), str(length), label]
                    #utils.check_equality(len(utils.line_to_word(unit_string)), 4)
                    utils.check_equality(len(unit_list), UNIT_QTY)
                utils.check_equality(len(unit_list), label_type.unit_length)
                wapiti_unit_list.append(' '.join(unit_list))
                index += 1

        return '\n'.join(wapiti_unit_list)

    # Create the Wapiti unit for a TEST sentence
    def to_wapiti_test_sentence(self, pos=False, expand=False,
                           verbose=False, gold=False, label_type=None):
        '''Create from the inputs for a sentence its corresponding Wapiti TEST unit.

        Output for each morpheme:
            new version: morpheme morpheme_position_in_word morpheme_length
        No output labels!
        '''
        # Prepare the source and the gloss which have the same number of morphemes/units
        # For the grammatical glosses
        # For a Wapiti unit with morpheme boundaries
        split_source_with_hyphen = word_to_morpheme_decomp(self.source)
        #print(split_source_with_hyphen)

        # Use BIO morpheme position tags
        if label_type.full or label_type.both or label_type.morph:
            # or label_type.dist:
            morpheme_position = self.morpheme_position_list(biof=True)
        else:
            morpheme_position = self.morpheme_position_list(biof=False)

        # Prepare the other part: lexical glosses, the translation and the alignment
        #aligned_lex_gloss = self.reference_and_aligned_gloss_with_source(
        #                alignment, expand_alignment=expand, verbose=verbose,
        #                train_dict=train_dict)
        #self.split_translation = utils.line_to_word(self.translation)
        #translation_length = len(self.split_translation)
        # To get the correct relative difference value
        tokenised_translation, lemmatised_translation = \
                        sp.lemmatise_sentence_for_alignment(self.translation)
        #translation_length = len(tokenised_translation)
        wapiti_unit_list = []
        index = 0 # Index for all morphemes (i.e. without hyphen)
        j = 0 # Index for lexical gloss

        for i in range(len(split_source_with_hyphen)):
            morpheme = split_source_with_hyphen[i] # Source morpheme
            if morpheme == '-':
                pass
            else:
                length = len(morpheme) # Morpheme length
                if label_type.morph:
                    morph_init, morph_end = extract_letters(morpheme, length)
                cap_length = cap_morpheme_length(length, cap_val=5)

                #gloss = self.split_gloss[index]
                position_pair = morpheme_position[index]
                utils.check_equality(morpheme, position_pair[0]) # Check that the morphemes do match
                position = position_pair[1]

                # Basic inputs
                unit_list = [morpheme, str(position), cap_length] #str(length)]
                if label_type.dist:
                    copy_src = copy_in_sentence(morpheme, tokenised_translation)
                    relative_src_pos = index / self.n_morph
                    position_src = cap_position_in_sentence(relative_src_pos)
                    unit_list.extend([morph_init, morph_end,
                                 copy_src, position_src])
                elif label_type.both or label_type.full or label_type.morph: # Five inputs
                    unit_list.extend([morph_init, morph_end])
                #elif label_type.full:
                #    unit_list.extend([morph_init, morph_end]) #, str(index)]) #,
                #elif label_type.morph:
                #    unit_list.extend([morph_init, morph_end]) #,
                elif label_type.pos: # With PoS tag
                    unit_list = [morpheme, str(position), str(length)] #,
                                 #label, gram_lex, aligned_pos]
                    #utils.check_equality(len(unit_list), UNIT_QTY_POS)
                else:
                    unit_list = [morpheme, str(position), str(length)] #, label]
                    #utils.check_equality(len(utils.line_to_word(unit_string)), 4)
                    #utils.check_equality(len(unit_list), UNIT_QTY)
                utils.check_equality(len(unit_list), label_type.label_position)
                wapiti_unit_list.append(' '.join(unit_list))
                index += 1

        return '\n'.join(wapiti_unit_list)

    # Functions about lexical glosses
    def only_lex_gloss_sent(self, punctuation=False, gloss_sep=r'.'):
        '''Extract lexical glosses only.

        # New warning: outputs a list (and not a string)!
        Composed lexical glosses with dots are also separated.
        punctuation: if there are punctuation marks (=True), remove them.'''
        sent_list = []

        # Define how to split glosses
        #if gloss_sep == r'.':
        #    split_marker = r'[.]'
        #else:
        #    split_marker = gloss_sep

        for gloss in self.split_gloss:
            if gloss.isupper(): # Grammatical gloss
                continue
            elif punctuation and (gloss in PUNCTUATION_LIST):
                continue
            else: # Lexical gloss
                if gloss_sep in gloss: # Composed gloss
                    split_gloss = split_composed_gloss(gloss, gloss_sep)
                    #re.split(split_marker, gloss)
                    #print(split_gloss)
                    for g in split_gloss:
                        if g.isupper(): # Grammatical gloss
                            pass
                        else:
                            sent_list.append(g)
                else: # Simple gloss
                    sent_list.append(gloss)
        return sent_list #' '.join(sent_list)

    def original_lex_gloss_sent(self, punctuation=False, gloss_sep=r'.'):
        '''Extract lexical glosses as they are (no separation with the dots).

        # New warning: outputs a list (and not a string)!'''

        sent_list = []
        for gloss in self.split_gloss:
            if gloss.isupper(): # Grammatical gloss
                continue
            elif punctuation and (gloss in PUNCTUATION_LIST):
                continue
            else:
                if gloss_sep in gloss: # Composed gloss
                    split_gloss = split_composed_gloss(gloss, gloss_sep)
                    #re.split(r'[.]', gloss)
                    #print(split_gloss)
                    new_split_gloss = []
                    for g in split_gloss:
                        if g.isupper():
                            pass
                        else:
                            new_split_gloss.append(g)
                    sent_list.append(gloss_sep.join(new_split_gloss)) # '.'
                else: # Simple gloss
                    sent_list.append(gloss)
        return sent_list #' '.join(sent_list)

    def reference_and_aligned_gloss_with_source(self, alignment,
                        expand_alignment=False, verbose=False, train_dict=None,
                        punctuation=False, gloss_sep=r'.', language='en',
                        hyphen=True):
        '''Get the reference and aligned glosses while keeping the source morpheme as information.

        There is an additional output at each position.
        The PoS tag (if existing) is always given.
        Output format: [(corresponding_source_morpheme, reference_original_lex_gloss,
                         aligned_gloss (if relevant), corresponding_pos (if relevant),
                         alignment_index (if relevant))].
        expand_alignment: boolean parameter to get more alignments
                          based on the presence in the translation.
        verbose: boolean parameter to print all messages.
        train_dict: dictionary to complete some missing alignments
                    for morphemes seen during training.'''
        #print(self.source, self.gloss, self.translation)
        #sentence = tw.Sentence(source, gloss, translation)
        split_alignment = string_to_pair_list(alignment) #utils.line_to_word(alignment)
        split_lex_gloss = self.only_lex_gloss_sent(punctuation=punctuation)
        split_original_gloss = self.original_lex_gloss_sent(punctuation=punctuation)
        standardised_full_gloss = remove_gram_from_mixed_gloss(self.split_gloss)
        if verbose: print(f'Standardised full gloss: {standardised_full_gloss}')
        if verbose: print(f'Find an alignment for {split_original_gloss}\n'
                          f'({len(split_original_gloss)} lexical glosses)\n')

        #split_translation = utils.line_to_word(translation)
        # With the aligned word, its PoS tag, and its position
        tokenised_translation, lemmatised_translation = \
                sp.lemmatise_sentence_for_alignment(self.translation, language,
                                                    hyphen)
        split_translation = [unit[0] for unit in lemmatised_translation]
        split_pos = [unit[1] for unit in lemmatised_translation]
        utils.check_equality(len(split_translation), len(split_pos))

        if expand_alignment:
            raw_split_translation = tokenised_translation

        #if verbose: print(test(only_lex_gloss_sent(gloss), translation, alignment))
        alignment_n = len(split_alignment)
        if alignment_n == 0: # No alignments
            if verbose: print('No alignment pair')
            return []

        lexical_gloss_match_list = []
        lex_gloss_index = 0
        alignment_index = 0
        original_index = 0
        for i in range(len(split_original_gloss)):
            original_gloss = split_original_gloss[i]
            original_index = find_source_index(standardised_full_gloss,
                                               original_gloss, original_index)
            source_morph = self.split_source[original_index]
            if verbose: print(f'Processing {original_gloss} ({i}, at the '
                        f'{original_index} position originally {source_morph})')

            if alignment_index >= alignment_n: # If there are no more alignment pairs
                lexical_gloss_match_list.append(
                        (source_morph, original_gloss, '?', '?', -1))
                continue

            # There are some unprocessed alignment pairs left
            lex_gloss = split_lex_gloss[lex_gloss_index]
            if verbose: print(original_gloss, lex_gloss, lex_gloss_index,
                              split_alignment[alignment_index])
            if original_gloss == lex_gloss: # Same lexical gloss
                # The lexical gloss is aligned
                if lex_gloss_index == split_alignment[alignment_index][0]:
                    translation_index = split_alignment[alignment_index][1]
                    lexical_gloss_match_list.append((source_morph, original_gloss,
                                    split_translation[translation_index],
                                    lemmatised_translation[translation_index][1],
                                    translation_index))
                    alignment_index += 1
                else: # No aligned lexical gloss: align it with an empty string
                    if expand_alignment:
                        # Using the training dictionary
                        if (train_dict) and (source_morph in train_dict):
                            pos_tag = train_dict[source_morph][2]
                            lexical_gloss_match_list.append(
                                (source_morph, original_gloss,
                                train_dict[source_morph][0], pos_tag, -1))
                        # No aligned word in the translation ...
                        # ... but the lexical gloss is in the translation
                        elif (original_gloss in split_translation) or \
                            (original_gloss in raw_split_translation):
                            lexical_gloss_match_list.append(
                            (source_morph, original_gloss, original_gloss, '?', -1))
                        else: # Unknown
                            lexical_gloss_match_list.append(
                            (source_morph, original_gloss, '?', '?', -1))
                    else:
                        lexical_gloss_match_list.append(
                        (source_morph, original_gloss, '?', '?', -1))
                lex_gloss_index += 1

            else: # Not the same lexical gloss between the original and aligned lexical glosses
                # Split the composed gloss to handle each gloss separately
                split_comp_gloss = split_composed_gloss(original_gloss, gloss_sep)
                #re.split(r'[.]', original_gloss)
                if verbose: print(split_comp_gloss)
                composed_gloss = [] #[lex_gloss]
                aligned_indices = [] #[lex_gloss_index]
                lex_gloss_index_temp = lex_gloss_index
                while split_comp_gloss != composed_gloss:
                    composed_gloss.append(split_lex_gloss[lex_gloss_index_temp])
                    aligned_indices.append(lex_gloss_index_temp)
                    lex_gloss_index_temp += 1

                if verbose: print(f'Aligned indices: {aligned_indices}')
                aligned_word_list = []
                for index in aligned_indices:
                    # If the lexical gloss is aligned
                    if (alignment_index < alignment_n) and \
                        (index == split_alignment[alignment_index][0]):
                        translation_index = split_alignment[alignment_index][1]
                        aligned_word_list.append(split_translation[translation_index])
                        alignment_index += 1
                    else: # No aligned lexical gloss: align it with an empty string
                        pass
                if verbose: print(f'Found aligned words: {aligned_word_list}')
                aligned_word = gloss_sep.join(aligned_word_list)

                # Assign the found word as a label; if nothing is found, use the dictionary?
                if aligned_word != '': # Non empty label
                    lexical_gloss_match_list.append(
                        (source_morph, original_gloss, aligned_word, '?', -1))
                else:
                    if expand_alignment:
                        # Using the training dictionary
                        if (train_dict) and (source_morph in train_dict):
                            pos_tag = train_dict[source_morph][2]
                            lexical_gloss_match_list.append(
                                (source_morph, original_gloss,
                                train_dict[source_morph][0], pos_tag, -1))
                        else: # Unknown
                            lexical_gloss_match_list.append(
                            (source_morph, original_gloss, '?', '?', -1))
                    else:
                        lexical_gloss_match_list.append(
                                (source_morph, original_gloss, '?', '?', -1))
                lex_gloss_index = lex_gloss_index_temp

        utils.check_equality(split_original_gloss, [pair[1]
                                        for pair in lexical_gloss_match_list])
        return lexical_gloss_match_list


# Some utility functions
def word_to_morpheme_decomp(word):
    '''Decompose into morphemes while keeping morpheme information.'''
    return re.split(' ', re.sub('-', ' - ', word))

def find_source_index(source_list, gloss, last_index=0):
    '''Find the index of the corresponding source morpheme/lexical gloss.'''
    return source_list.index(gloss, last_index)

def split_composed_gloss(gloss, gloss_sep=r'.'):
    '''Split a composed gloss according to a separator.'''
    # Define how to split glosses
    if gloss_sep == r'.':
        split_marker = r'[.]'
    else:
        split_marker = gloss_sep
    return re.split(split_marker, gloss)

def remove_gram_from_mixed_gloss(gloss_list, gloss_sep=r'.'):
    '''Remove the grammatical gloss part from hybrid glosses.

    (grammatical and lexical glosses fused).'''
    new_gloss_list = []
    for gloss in gloss_list:
        if gloss_sep in gloss: # Composed gloss
            split_gloss = split_composed_gloss(gloss, gloss_sep)
            #re.split(r'[.]', gloss)
            #print(split_gloss)
            new_split_gloss = []
            for g in split_gloss:
                if g.isupper():
                    pass
                else:
                    new_split_gloss.append(g)
            new_gloss_list.append(gloss_sep.join(new_split_gloss)) # '.'
        else: # Simple gloss
            new_gloss_list.append(gloss)
    return new_gloss_list

# New inputs
def extract_letters(string, length):
    '''Extract the initial and final letters of a morpheme (at most 3).'''
    if length >= 3:
        morph_init = string[0:3]
        morph_end = string[-3:]
    else: # Morpheme of 1 or 2 letters
        morph_init = string
        morph_end = string
    return morph_init, morph_end

def cap_morpheme_length(length, cap_val=5):
    '''Cap morpheme length as input (and return a string).'''
    if length > cap_val:
        return '6+'
    else:
        return str(length)

def relative_position_difference(i, I, j, J):
    '''Compute the relative position difference (no cap).'''
    if j < 0: # Not aligned
        return -1
    else:
        return abs((i / I) - (j / J))

def raise_value_error(rd):
    '''Raise value error for a relative difference.'''
    raise ValueError(f'The relative difference is not in range [0,1]: {rd}')

def cap_relative_difference(rd):
    '''Cap the relative difference (and return a string).'''
    # All conditions
    if rd < 0: # Not aligned
        if rd != -1:
            raise_value_error(rd)
        cap_rd = -1
    elif rd < 0.1: # 0 - 0.1
        cap_rd = 0.0
    elif rd < 0.2: # 0.1 - 0.2
        cap_rd = 0.1
    elif rd < 0.3: # 0.2 - 0.3
        cap_rd = 0.2
    elif rd <= 1.0: # 0.3 - 1.0
        cap_rd = '0.3+'
    #elif rd <= 1.0: # 0.1 - 1.0 # NEW version
    #    cap_rd = '0.1+'
    else:
        raise_value_error()
    return str(cap_rd)
    #return str(round(rd, 1)) #2))

# Compute whether a string is in a sentence or not
def copy_in_sentence(string, tokenised_sentence):
    '''Compute whether a string is in a sentence or not (copy).

    src: the source morpheme is in the translation
    trg: the translation word is in the source sentence'''
    if (string.capitalize() in tokenised_sentence) \
        or (string.lower() in tokenised_sentence):
        return str(1)
    else:
        return str(0)

# Compute position in sentence
def cap_position_in_sentence(relative_pos):
    '''Compute a categorised relative position in sentence.'''
    #string_position = tokenised_sentence.index(string)
    #relative_pos = string_position / len(tokenised_sentence)
    #print(string, tokenised_sentence, string_position, relative_pos)
    if relative_pos < 0: # Not aligned
        return str(-1)
    elif relative_pos < 0.25:
        return '1/4'
    elif relative_pos < 0.5: # (relative_pos >= 0) and
        return '2/4' #str(0)
    elif relative_pos < 0.75:
        return '3/4'
    elif relative_pos <= 1.0:
        return '4/4' #str(1)
    else:
        raise_value_error()

# Process alignments
def string_to_pair_list(alignment_string):
    '''Convert an alignment string into a list of alignment pairs.'''
    alignment_list = []
    if alignment_string == '': # No alignment
        pass
    else:
        split_string = utils.line_to_word(alignment_string)
        for pair_str in split_string:
            split_pair = pair_str.split('-')
            utils.check_equality(len(split_pair), 2)
            alignment_list.append((int(split_pair[0]), int(split_pair[1])))
    return alignment_list
