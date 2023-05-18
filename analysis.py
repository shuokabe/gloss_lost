import matplotlib.pyplot as plt
import os
import re

from gloss_lost.to_wapiti import LabelHandler
import gloss_lost.utils as utils

REF_LABEL_POSITION = 3
LABEL_POSITION = 3

# Evaluation of the output of Lost
def extract_labels(ref_file, lost_file, label_type='base', source=False,
                   control=True):
    '''Extract the reference and predicted labels.

    The reference is a Wapiti format file, while the prediction is a lost file.
    This function keeps the sentence structure:
    each sentence is a list in the output.
    source: boolean parameter to include the source morpheme in the output or not.
    control: check the unit size
    '''
    label = LabelHandler(label_type)
    print(f'Using a {label_type} label ({label.unit_length} units)')

    split_ref = re.split('\n\n', ref_file)
    split_pred = utils.text_to_line(lost_file)
    n = len(split_ref)
    utils.check_equality(n, len(split_pred))
    print(f'Evaluating {n} sentences')

    label_list = []
    for i in range(n): # Iterate through all sentences
        ref_sentence = utils.text_to_line(split_ref[i])
        pred_sentence = utils.line_to_word(split_pred[i][:-1])
        # -1 because there is a final whitespace in the output
        #print(i)
        m = len(ref_sentence)
        utils.check_equality(m, len(pred_sentence))

        sentence_label_list = []
        for j in range(m): # Iterate through all morphemes
            ref_unit = utils.line_to_word(ref_sentence[j])
            pred_unit = re.split('[|@]', pred_sentence[j])
            utils.check_equality(len(ref_unit), 4) #5)
            if control: utils.check_equality(len(pred_unit), label.unit_length) #5)
            utils.check_equality(ref_unit[0], pred_unit[0])
            #else:
                #sentence_label_list.append((ref_unit[4], pred_unit[4]))
            if source: # Include source morpheme in output
                sentence_label_list.append((ref_unit[0],
                ref_unit[REF_LABEL_POSITION], pred_unit[label.label_position]))
            else: # Keep the labels only
                sentence_label_list.append(
                (ref_unit[REF_LABEL_POSITION], pred_unit[label.label_position]))
        label_list.append(sentence_label_list)
    return label_list

def wapiti_field(data, field_n=3):
    '''Extract one specific field from a dataset in the Wapiti format.

    The default value is 3 for the main label (4th position).'''
    split_data = utils.text_to_line(data)
    label_list = []
    for line in split_data:
        if (line == '') or (line == '- - - -'): # Blank line or hyphen
            continue
        else:
            split_line = re.split(' ', line)
            utils.check_equality(len(split_line), 4) #2)
            #label_list.append((split_line[0], split_line[3])) # Morpheme and label
            label_list.append(split_line[field_n]) # Only the label (default)
    return label_list

# Accuracy
def glossing_accuracy(ref_wapiti, pred_lost, label_type='base', verbose=True,
                      control=True):
    '''Compute the accuracy with a reference (Wapiti format) and a Lost output.

    Only when there is one output.
    verbose parameter to print the differentiated accuracy.'''
    extracted_labels = extract_labels(ref_wapiti, pred_lost,
                                      label_type=label_type, control=control)
    flat_labels = utils.flatten_2D(extracted_labels)
    utils.check_equality(len(flat_labels[0]), 2) # Check pair
    accuracy = utils.compute_accuracy(extracted_labels) #, flat=True)
    if verbose: print(f'General accuracy: {(accuracy * 100):.3f}')
    if verbose: print(accuracy_per_gram_lex(flat_labels))
    #return utils.compute_accuracy(extracted_labels) #, flat=True)
    return accuracy

# Compute the separate accuracy for glosses, either grammatical or lexical
def accuracy_per_gram_lex(extracted_labels):
    '''Compare the accuracy for the grammatical and lexical glosses.

    The glosses are categorised according to the reference.'''
    gram_label_list, lex_label_list = [], []
    for label_pair in extracted_labels:
        if label_pair[0].isupper(): # Grammatical label (reference)
            gram_label_list.append(label_pair)
        else: # Lexical label (reference)
            lex_label_list.append(label_pair)
    gram_acc = utils.compute_accuracy(gram_label_list, flat=True)
    lex_acc = utils.compute_accuracy(lex_label_list, flat=True)
    print(f'Accuracy for the grammatical glosses: {(gram_acc * 100):.3f}')
    print(f'Accuracy for the lexical glosses: {(lex_acc * 100):.3f}')
    return gram_acc, lex_acc

# Compute the accuracy for UNSEEN lexical glosses
def unseen_lex_gloss_accuracy(ref_wapiti, pred_lost, gold_dict, label_type,
                              verbose=0):
    '''Compute the accuracy for UNSEEN lexical glosses based on a dictionary.

    The gold training dataset is in a Wapiti format.'''
    ##seen_words = set(dictionary.keys())]
    # Set of all seen source morphemes
    ##seen_morphemes = set(wapiti_field(gold_train, field_n=0))
    #seen_morph_label = [(key, label) for key, value in gold_dict.items()
    #                    for label in value]
    seen_morph_label = set(gold_dict.keys())
    flat_extracted_labels = utils.flatten_2D(extract_labels(ref_wapiti,
                                            pred_lost, label_type, source=True))
    n_token = len(flat_extracted_labels)
    if (verbose >= 1): print(f'There are {n_token} morphemes in total (tokens).')
    # Filter source morphemes only seen during the inference
    lex_labels = [unit for unit in flat_extracted_labels
                if not unit[1].isupper()] # Reference lexical labels only
    unseen_source_labels = [(unit[1], unit[2]) for unit in lex_labels
                ##[(unit[1], unit[2]) for unit in flat_extracted_labels
                ##if (unit[0] not in seen_morphemes) and (not unit[1].isupper())]
                #if ((unit[0], unit[1]) not in seen_morph_label)]
                if (unit[0] not in seen_morph_label)]
                ##and (not unit[1].isupper()))]
    # Compute the proportion of unseen morphemes
    n_unseen = len(unseen_source_labels)
    test_morph_list = wapiti_field(ref_wapiti, field_n=3)
    test_lex_label_list = [label for label in test_morph_list if not label.isupper()]
    n_test_lex = len(test_lex_label_list)
    utils.check_equality(n_test_lex, len(lex_labels))
    if (verbose >= 1): print(f'{n_test_lex} lexical labels in test dataset.')
    # Display
    print(f'{n_unseen} lexical morpheme tokens of the test dataset '
    #print(f'{n_unseen} lexical morph-label pairs (tokens) of the test dataset '
          f'are unseen during training ({(n_unseen / n_test_lex) * 100:.3}%).')
    if verbose == 2: print(unseen_source_labels)
    return utils.compute_accuracy(unseen_source_labels, flat=True)

# Plot the accuracy on the developement dataset
def dev_accuracy_evolution(reference, folder_path, iteration=15, label_type='base'):
    '''Observe the evolution of accuracy on the dev dataset.'''
    file_number = 0
    accuracy_list = []
    for file_name in os.listdir(folder_path):
        # Open only .txt files
        if file_name[-4:] == '.txt':
            pred_file = open(os.path.join(folder_path, file_name), 'r').read()
            find_iteration = re.findall(r"\d+", file_name)
            #print(find_iteration)
            # Change value below: depending on the file naming format
            #utils.check_equality(len(find_iteration), 2) # 3) #2) #3) #2)#1)
            assert len(find_iteration) in [2, 3], \
                            f'File name issue: {len(find_iteration)} numbers'
            iteration_number = int(find_iteration[0])
            print(f'File: {file_name}; iteration {iteration_number}')
            file_number += 1
            #acc = glossing_accuracy_morph(reference, pred_file)
            extracted_labels = extract_labels(reference, pred_file, label_type)
            acc = utils.compute_accuracy(extracted_labels)
            accuracy_list.append((iteration_number, acc))
            #print(analyse_tl_file_name(file_name, key_seed))
        else:
            pass
        #print('\n')
    print(f'Processed {file_number} files')
    utils.check_equality(file_number, iteration)

    # Sort the accuracy list according to the iteration number
    sorted_list = sorted(accuracy_list, key = lambda x:(x[0], x[1]))
    #print(sorted_list)
    # Check that the iteration numbers are correct
    check_list = [pair[0] for pair in sorted_list]
    utils.check_equality(check_list, list(range(1, (file_number + 1))))

    sorted_acc_list = [pair[1] for pair in sorted_list]
    max_acc = max(sorted_acc_list) #max(accuracy_list)
    print(f'Maximum accuracy {max_acc:.4f} '
          f'reached at iteration {(sorted_acc_list.index(max_acc) + 1)}')

    plt.plot(range(1, (file_number + 1)), sorted_acc_list)
    return sorted_acc_list
