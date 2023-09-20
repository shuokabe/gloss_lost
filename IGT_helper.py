import os
import re

import gloss_lost.to_wapiti as tw
import gloss_lost.utils as utils


### Preprocessing functions ###
def preprocess_translation(translation_sentence, tilde=False, lower=True):
    '''Preprocess the translation sentence.'''
    new_sentence = translation_sentence.strip()
    if lower: new_sentence = new_sentence.lower()
    new_sentence = re.sub(r'[!?\[\];:"/.(),]', ' ', new_sentence)
    new_sentence = re.sub(r'[=|$¿¡]', ' ', new_sentence) # New
    if tilde: new_sentence = re.sub(r'~', ' ', new_sentence)
    new_sentence = re.sub('--', ' ', new_sentence) # New
    new_sentence = re.sub(' - ', ' ', new_sentence) # New
    # Deal with '
    new_sentence = re.sub(" ' ", ' ', new_sentence)
    new_sentence = re.sub(" '", ' ', new_sentence)
    # Deal with other special characters
    new_sentence = re.sub(r'[\u200e“”\u200b…<>]', ' ', new_sentence) # New
    new_sentence = re.sub(' +', ' ', new_sentence)
    return new_sentence.strip()

def split_uncovered(uncovered_file, save_path, file_name, covered=False,
                    tilde=False, lower=True):
    '''Convert the Shared task file into three separate files.

    4 tiers:
    \t: raw source sentence
    \m: morpheme segmented sentence
    \g: gloss
    \l: translation
    If covered, there is no gloss output'''
    split_file = re.split('\n\n', uncovered_file)
    n = len(split_file)
    print(f'There are {n} sentences.')

    source_list, gloss_list, translation_list = [], [], []
    for sentence in split_file:
        split_sentence = utils.text_to_line(sentence)
        if len(split_sentence) == 5: # 5 tiers
            #print(split_sentence)
            field_dict = {'m': 1, 'g': 3, 'l': 4}
        elif len(split_sentence) == 4:
            field_dict = {'m': 1, 'g': 2, 'l': 3}
        else:
            print(f'Number of tiers {len(split_sentence)} for {split_sentence}')
        #utils.check_equality(len(split_sentence), 4) # 4 tiers

        # Source sentence
        source = split_sentence[field_dict['m']]
        utils.check_equality(source[0:3], '\m ')

        # Gloss sentence
        gloss = split_sentence[field_dict['g']]
        utils.check_equality(gloss[0:3], '\g ')

        # Translation
        translation = split_sentence[field_dict['l']]
        utils.check_equality(translation[0:3], '\l ')

        # Add to lists
        source_list.append(source[3:])
        gloss_list.append(gloss[3:])
        translation_list.append(
            preprocess_translation(translation[3:], tilde=tilde, lower=lower))


    #print(source_list[0])
    #print(gloss_list[0])
    #print(translation_list[0])
    # Save files
    utils.save_file('\n'.join(source_list), os.path.join(save_path, f'{file_name}_src.txt'))
    if not covered: utils.save_file('\n'.join(gloss_list), os.path.join(save_path, f'{file_name}_glo.txt'))
    utils.save_file('\n'.join(translation_list), os.path.join(save_path, f'{file_name}_trg.txt'))


### Language-specific preprocessing ###
def preprocess_lezgi_gloss(gloss_sentence):
    '''Preprocess Lezgi gloss sentence (grammatical glosses).'''
    cap_dict = {'1sg.abs': '1SG.ABS', '1sg.ERG': '1SG.ERG', '2sg': '2SG',
                '1sg.gen': '1SG.GEN', '2sg.abs': '2SG.ABS', '2sg.gen': '2SG.GEN',
                '1pl': '1PL', '1pl.abs': '1PL.ABS', '1pl.erg': '1PL.ERG', '1pl.gen': '1PL.GEN',
                '2pl': '2PL', '2pl.abs': '2PL.ABS', '2pl.dat': '2PL.DAT', '2sg.erg': '2SG.ERG'}
    split_gloss = utils.line_to_word(re.sub('-', ' - ', gloss_sentence))
    new_sent_list = []
    for morpheme in split_gloss:
        if morpheme in cap_dict:
            new_sent_list.append(cap_dict[morpheme])
            #print(morpheme)
        else:
            new_sent_list.append(morpheme)
    return re.sub(' - ', '-', ' '.join(new_sent_list))


### Functions to help create the Wapiti corpus ###
def create_corpus(path, file_start, test=False):
    '''Create the language Corpus object from a path.'''
    src_data = open(os.path.join(path, f'{file_start}_src.txt'), 'r').read()
    if test:
        glo_data = ''
    else:
        glo_data = open(os.path.join(path, f'{file_start}_glo.txt'), 'r').read()
    trg_data = open(os.path.join(path, f'{file_start}_trg.txt'), 'r').read()
    return tw.Corpus(src_data, glo_data, trg_data, test=test)

# Create lex only file
def lex_only_file(corpus, gloss_sep=r'.'):
    '''Create a file with lexical glosses only.'''
    lex_only_list = []
    for sentence in corpus.sentence_list:
        gloss_sent = sentence.only_lex_gloss_sent(gloss_sep=gloss_sep) # Sentence object
        lex_only_list.append(re.sub(' +', ' ', ' '.join(gloss_sent)).strip())
    return lex_only_list

# Create dummy alignment for the dev and test datasets
def create_dummy_alignment(lex_gloss, translation):
    '''Create dummy alignment to generate the test and development datasets.'''
    split_lex_gloss = utils.text_to_line(lex_gloss, empty=False)
    if split_lex_gloss[-1] == '': split_lex_gloss = split_lex_gloss[:-1]
    split_translation = utils.text_to_line(translation, empty=False)
    n = len(split_lex_gloss)
    utils.check_equality(n, len(split_translation))
    alignment_list = []
    for i in range(n):
        lex_gloss_sent = utils.line_to_word(split_lex_gloss[i])
        translation_sent = utils.line_to_word(split_translation[i])
        m = min(len(lex_gloss_sent), len(translation_sent))
        alignment_line = [f'{j}-{j}' for j in range(m)]
        alignment_list.append(' '.join(alignment_line))
    return '\n'.join(alignment_list)

def display_sentence_alignment(path, file_start, i, alignment):
    '''Display the sentence with its alignments (test = True).'''
    corpus = create_corpus(path, file_start, test=True)
    split_alignment = utils.text_to_line(alignment)
    sentence = corpus.sentence_list[i]
    #tw.Sentence(corpus.split_src[i], gloss, translation)
    return (sentence.reference_and_aligned_gloss_with_source(split_alignment[i]),
    sentence)

### Functions to help create the Lost corpus ###
def concatenate_files(wapiti_path, translation_path, language, method,
                      train_size, test=False):
    '''Concatenate the train and development Wapiti and translation files.'''
    # Wapiti files
    train_wapiti = open(os.path.join(wapiti_path,
    f'wapiti_for_lost_{language}_train_match_{train_size}_{method}.txt'), 'r').read()
    dev_wapiti = open(os.path.join(wapiti_path,
        f'wapiti_for_lost_{language}_dev_match_{method}.txt'), 'r').read()
    test_wapiti = open(os.path.join(wapiti_path,
        f'wapiti_for_lost_{language}_test_match_{method}.txt'), 'r').read()

    if test: conc_wapiti = '\n\n'.join([train_wapiti, dev_wapiti, test_wapiti])
    else: conc_wapiti = '\n\n'.join([train_wapiti, dev_wapiti])

    # Translation files
    train_trg = open(os.path.join(translation_path,
                                f'raw_{language}_train_trg.txt'), 'r').read()
    dev_trg = open(os.path.join(translation_path,
                                    f'raw_{language}_dev_trg.txt'), 'r').read()
    test_trg = open(os.path.join(translation_path,
                                    f'raw_{language}_test_trg.txt'), 'r').read()

    if test: conc_trg = '\n'.join([train_trg, dev_trg, test_trg])
    else: conc_trg = '\n'.join([train_trg, dev_trg])

    utils.save_file(conc_wapiti, os.path.join(wapiti_path,
            f'wapiti_for_lost_{language}_conc_match_{train_size}_{method}.txt'))
    utils.save_file(conc_trg,
                os.path.join(translation_path, f'raw_{language}_conc_trg.txt'))


### Analysis ###
def convert_lost_to_IGT(split_source, split_target, lost_results,
                        postprocess=None, sep_at=False, label='morph'):
    '''Convert a Lost result file into an IGT format (Shared Task).'''
    label_type = tw.LabelHandler('comp') #label)
    #split_source = utils.text_to_line(source)
    #split_target = utils.text_to_line(target)
    split_lost_results = utils.text_to_line(lost_results)
    n = len(split_source)
    utils.check_equality(n, len(split_target))
    utils.check_equality(n, len(split_lost_results))
    print(f'There are {n} sentences')

    new_file_list = []
    for i in range(n):
        if i == 0: print(split_lost_results[i])
        source_sent = split_source[i]
        # Handle hyphen as punctuation mark
        if (' - ' in source_sent):
            source_sent = source_sent.replace(' - ', ' $$$ ')
        if (source_sent[0:2] == '- '):
            source_sent = '$$$' + source_sent[1:]

        morpheme_only_list = re.split('[ -]', source_sent) #split_source[i])
        morpheme_list = re.split('[ ]', re.sub('-', ' - ', source_sent)) #split_source[i]))
        split_lost_line = utils.line_to_word(split_lost_results[i])[:-1]
        # -1 because the last element is an empty string
        m = len(morpheme_list)
        #print(morpheme_only_list, split_lost_line)
        if len(morpheme_only_list) != len(split_lost_line):
            print(morpheme_only_list, split_lost_line)
        utils.check_equality(len(morpheme_only_list), len(split_lost_line))
        label_list = []
        j = 0 # Lost line index
        for morpheme in morpheme_list:
            if morpheme == '-':
                label_list.append('-')
            else:
                split_unit = re.split('@', split_lost_line[j])
                if not sep_at:
                    utils.check_equality(len(split_unit), 2) # Verify that there are only to units
                    #print(split_unit)
                    split_input = re.split('\|', split_unit[0])
                    split_output = re.split('\|', split_unit[1])
                else: # if sep_at, join the remaining elements
                    #split_unit[1] = '@'.join(split_unit[1:])
                    split_unit = process_at_output(label_type, split_lost_line[j])
                    split_input = split_unit # NOT TRUE but the check function works
                    split_output = split_unit[label_type.label_position:]

                #print(split_input, split_output)
                utils.check_equality(split_input[0], morpheme) # Verify that the source morphemes match
                label_list.append(split_output[0])
                j += 1

        reconst_gloss = re.sub(' - ', '-', ' '.join(label_list))
        if postprocess: # Postprocess gloss line
            reconst_gloss = postprocess(reconst_gloss)
        s_line = f'\\m {split_source[i]}'
        g_line = f'\\g {reconst_gloss}'
        l_line = f'\\l {split_target[i]}'
        if i == 0:
            print('\n'.join([s_line, g_line, l_line]))
        new_file_list.append('\n'.join([s_line, g_line, l_line]))
    return '\n\n'.join(new_file_list)

def process_at_output(label, string):
    '''Process Lost predictions with @ symbol (Uspanteko).'''
    #label = LabelHandler(label_type)
    first_pass = re.split('[|]', string)
    #print(first_pass)
    pred_unit = first_pass[0:(label.label_position - 1)] # Inputs without last (because of @)
    #print(pred_unit)
    to_process = first_pass[(label.label_position - 1):]
    #print(to_process)
    utils.check_equality(len(to_process), label.output_length)
    processed = re.split('@', to_process[0])
    pred_unit.append(processed[0]) # Last input
    #print(pred_unit)
    if len(processed) > 1:
        pred_unit.append('@'.join(processed[1:]))
    #print(pred_unit)
    pred_unit.extend(to_process[-(label.output_length - 1):]) # 2
    return pred_unit

def extract_source_gloss_target(split_IGT_unit):
    '''Extract the source, gloss and target fields from an IGT unit.'''
    for j in range(len(split_IGT_unit)):
        line = split_IGT_unit[j]
        #print(line, line[:3], len(line[:3]))
        if line[:3] == '\\m ': # Source
            source = line[3:]
        elif line[:3] == '\\l ': # Translation
            translation = line[3:]
        elif line[:3] == '\\g ': # Gloss
            gloss = line[3:]

    #print(f'Fields: source: {source}\ngloss: {gloss}\ntranslation: {translation}')
    return source, gloss, translation

def fill_covered(covered, src, trg, prediction, postprocess=None, sep_at=False):
    '''Use the same format as the covered IGT file.'''
    converted_prediction = convert_lost_to_IGT(src, trg, prediction,
                                        postprocess=postprocess, sep_at=sep_at)
    split_covered = re.split('\n\n', covered)
    split_pred = re.split('\n\n', converted_prediction)
    n = len(split_covered)
    #print(split_pred, split_covered)
    utils.check_equality(n, len(split_pred))

    filled_list = []
    for i in range(n):
        covered_unit = utils.text_to_line(split_covered[i])
        pred_unit = utils.text_to_line(split_pred[i])
        # Extract the covered source and target sentence, and the gloss line index
        cov_src, cov_glo, cov_trg = extract_source_gloss_target(covered_unit)
        gloss_index = covered_unit.index('\\g ')
        assert gloss_index >= 0, 'No gloss field found'

        # Extract the lines in the prediction
        pred_src, pred_glo, pred_trg = extract_source_gloss_target(pred_unit)
        utils.check_equality(cov_src, pred_src)
        #utils.check_equality(cov_trg, pred_trg)

        filled_unit = covered_unit
        filled_unit[gloss_index] = f'\\g {pred_glo}'

        filled_list.append('\n'.join(filled_unit))
    return '\n\n'.join(filled_list)

def create_submission_file(test_covered, test_path, output_path, language,
                           output_suffix, postprocess=None, sep_at=False):
    '''Create the prediction file in an IGT format (for submission).

    Change dev into test in create_pred_file function.'''
    test_src = open(os.path.join(test_path, f'raw_{language}_test_src.txt'), 'r').read()
    #test_glo = open(os.path.join(test_path, f'raw_{language}_test_glo.txt'), 'r').read()
    test_trg = open(os.path.join(test_path, f'raw_{language}_test_trg.txt'), 'r').read()

    sp_test_src = utils.text_to_line(test_src)#[-200:]
    #sp_test_glo = utils.text_to_line(test_glo)#[-200:]
    sp_test_trg = utils.text_to_line(test_trg)#[-200:]

    test_output = open(os.path.join(output_path, f'output_{language}_IGT_{output_suffix}.out'), 'r').read()

    # Reference
    #cgtigt_dev = convert_gold_to_IGT(sp_dev_src, sp_dev_trg, sp_dev_glo)

    # Prediction
    #cltigt_dev = convert_lost_to_IGT(sp_dev_src, sp_dev_trg, dev_output, sep_at=sep_at)
    cltigt_test = fill_covered(test_covered, sp_test_src, sp_test_trg,
                               test_output, postprocess=postprocess, sep_at=sep_at)

    print(cltigt_test[0:100])
    return cltigt_test


### Preprocessing & postprocessing function for specific languages
def lezgi_inverse_gloss(gloss_sentence):
    '''Revert preprocessing of Lezgi gloss (grammatical glosses).'''
    cap_dict = {'1sg.abs': '1SG.ABS', '1sg.ERG': '1SG.ERG', '2sg': '2SG',
                '1sg.gen': '1SG.GEN', '2sg.abs': '2SG.ABS', '2sg.gen': '2SG.GEN',
                '1pl': '1PL', '1pl.abs': '1PL.ABS', '1pl.erg': '1PL.ERG', '1pl.gen': '1PL.GEN',
                '2pl': '2PL', '2pl.abs': '2PL.ABS', '2pl.dat': '2PL.DAT', '2sg.erg': '2SG.ERG'}
    inv_cap_dict = {value: key for key, value in cap_dict.items()}
    split_gloss = utils.line_to_word(re.sub('-', ' - ', gloss_sentence))
    new_sent_list = []
    for morpheme in split_gloss:
        if morpheme in inv_cap_dict:
            new_sent_list.append(inv_cap_dict[morpheme])
            #print(morpheme)
        else:
            new_sent_list.append(morpheme)
    return re.sub(' - ', '-', ' '.join(new_sent_list))
