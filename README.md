# gloss_lost: Lost for automatic gloss generation

## Lost
The `lost` folder contains the original Lost (Lavergne et al., 2011, [2013][1]). Lost can be compiled with the `make` command in the folder.

The `run.sh` file contains all the parameters to run Lost.

[1]: https://aclanthology.org/F13-1033.pdf

## Lost for the glossing task
Disclaimer: this is a work in progress and contains some unpractical implementation parameters or constraints. We strive to remove them gradually.

Lost requires the data to be in a specific format, that can be obtained by following the step-by-step guide below. 
The `data/` directory contains example files for the Gitksan corpus of the SIGMORPHON Shared Task.

### 0. Initial data format
The programme needs the following data:
- a source file (`src_data`)
- a gloss file (`glo_data`)
- a translation file (`trg_data`).

Each sentence should be contained in one line and should correspond with each other.
Preprocessing must also be done at this step; for the Shared Task data, the IGT format is split into three with the `split_uncovered` function in the `IGT_helper.py` file.
An example is given at `data/raw_corpus/`.

### 1. Creating an alignment file
We used SimAlign [(Jalili Sabet et al., 2020)][2] to create an alignment between the lexical glosses and the translation words for the training data. 
We get an alignment file (`alignment`; cf. `data/raw_corpus/git_train_aligned_0.txt`), where each line contains pairs of indices `i-j`, associating a lexical gloss (`i`) with the target word (`j`). 

The `IGT_helper.py` file contains some utility functions:
- `lex_only_file` creates a corpus with lexical glosses only (which identifies fully-lowercased glosses)
- An alignment must be provided for every data, so `create_dummy_alignment` creates a dummy alignment to fill this constraint. It will not be considered by other functions. 

[2]: https://aclanthology.org/2020.findings-emnlp.147.pdf

### 2. Converting to the Wapiti format
From those three files, the data must first be converted to the Wapiti [(Lavergne et al., 2010)][3] format (CRF-compatible). 
To this end, the `to_wapiti.py` file can be used.

A `Corpus` object is used to process the data:
```
corpus = Corpus(src_data, glo_data, trg_data)
```
The optional parameter `test` indicates whether it is a training or test (includes development) data.

Converting the data to the Wapiti format can be done as follows:
```
wapiti_text = corpus.to_wapiti_format(alignment, expand=True, verbose=False, train_index=train_index_dict, pos=True, label_format=method)
```
Here, `train_index_dict` indicates the number of sentences used for training and `method` the desired input and output configuration. See `to_wapiti.py` for more details.
An example of Wapiti format file can be seen in `data/wapiti_format/`.

The training dictionary that might be used in the next step can be obtained under `corpus.train_dict`, which should be saved with the `pickle` package (see `data/wapiti_format/dictionary_gitksan_train_match_31.pkl`). 

Note: if the original data is already split into training, development, and test data (e.g., IGT Shared Task), this step should be done separately for each of them, and all three must be combined (using `concatenate_files` from `IGT_helper.py`, for instance) for the following step, keeping the train, dev, test order. The next function splits one combined file into three parts.

[3]: https://aclanthology.org/P10-1052.pdf

### 3. Converting to the Lost format
Lost requires a specific format, which can be obtained with the `wapiti_to_lost.py` file:

```
gloss_lost/wapiti_to_lost.py 'path_to_wapiti_file' 'path_to_trg_data'
--train_size training_size --test_size test_size --save_path 'save_path' -o output_file_name --train_dict 'path_to_train_dict.pkl' --dev_size development_size --pos --label_type method 
```
The function needs to access the Wapiti format file generated at the previous step (`path_to_wapiti_file`), the original translation file (`path_to_trg_data`), and the training dictionary (`path_to_train_dict.pkl`). The training, development, and test data sizes must be specified. 
Finally, `label_type` is the same as the one used in step 2.

Additional parameters can be specified, which are consistent with the previous step:
- `--punctuation` should be used when punctuation marks should be predicted (e.g., Natugu or Lezgi)
- `--trg_language` to indicate the target language (e.g., Uspanteko).

See `wapiti_to_lost.py` for more details.

This function returns four files, as in `data/lost_format/`: one reference file for the train set (`.ref`) and three search space files (`.spc`), if a development dataset is used.

### 4. Running Lost
In the run files, the data paths should be modified: 
- `train-spc`, `train-ref`: training search space and reference file paths
- `devel-spc`, `devel-out`: development search space and output paths (optional)
- `test-spc`, `test-out`: test search space and output paths
- `mdl-save`, `str-save`: paths to save the model.

Launching the model is done from the terminal with `./run.sh`. 
The output using the `dist` label can be seen at `data/output/output_gitksan_IGT_conc_match_dist_31.out`.

### 5. Evaluating the output
Using the `convert_lost_to_IGT` function from the `IGT_helper.py` file, the output from Lost can be converted back to the IGT format. The evaluation can then be carried out with the [Shared Task evaluation code][4] or a custom evaluation function.

[4]: https://github.com/sigmorphon/2023glossingST/tree/main/baseline

## Settings
Settings used in TALN:
`base` (simple in the experiments) and `pos` (struct.)

Settings used in Shared Task:
`morph` and `dist`

Settings used in ...:
`comp`

## Citation
- For Lost
```
@inproceedings{lavergne-etal-2013-fully,
    title = "A fully discriminative training framework for Statistical Machine Translation (Un cadre d{'}apprentissage int{\'e}gralement discriminant pour la traduction statistique) [in {F}rench]",
    author = "Lavergne, Thomas  and
      Allauzen, Alexandre  and
      Yvon, Fran{\c{c}}ois",
    booktitle = "Proceedings of TALN 2013 (Volume 1: Long Papers)",
    month = jun,
    year = "2013",
    address = "Les Sables d{'}Olonne, France",
    publisher = "ATALA",
    url = "https://aclanthology.org/F13-1033",
    pages = "450--463",
}
```

- For the Shared Task setting (`morph` and `dist`)
```
@inproceedings{okabe-yvon-2023-lisn,
    title = "{LISN} @ {SIGMORPHON} 2023 Shared Task on Interlinear Glossing",
    author = "Okabe, Shu  and
      Yvon, Fran{\c{c}}ois",
    booktitle = "Proceedings of the 20th SIGMORPHON workshop on Computational Research in Phonetics, Phonology, and Morphology",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.sigmorphon-1.21",
    doi = "10.18653/v1/2023.sigmorphon-1.21",
    pages = "202--208",
}
```
