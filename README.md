# Word Alignment

This program implements the IBM Model 1 for word alignment between a pair of parallel corpora. The input corpus should be in the form of a gzipped TSV file, where each line represents a sentence pair in the following format:

```
<English sentence>\t<Czech sentence>\t<sure word alignment>\t<possible word alignment>
```

For example:

```
This is a sentence.    Toto je věta.    0-0 1-1 2-2 3-3    0-0 1-1 2-2 3-3 
```

## Installation

To install the required packages, run:

```
pip install -r requirements.txt
```

## Usage

To run the word alignment process, use the following command:

```
python word_alignment.py --file_path <path_to_input_file> --iterations <number_of_iterations> --number_of_sentences <number_of_sentences> --top <number_of_top_translations> --threshold <threshold> --output_file <path_to_output_file> --lowercase
```

The command-line arguments are as follows:

- `--file_path`: Path to the input corpus file (default: input/czenali.gz).
- `--iterations`: Number of iterations to perform for the EM algorithm (default: 5).
- `--number_of_sentences`: Limit on the number of sentences to be processed (default 2000).
- `--top`: Number of top translations to consider for each Czech word (default: 3).
- `--threshold`: Threshold for translation probability (default: 0.0).
- `--output_file`: Path to the output file where the best translation dictionary will be written (default: output/best_dictionary.txt).
- `--lowercase`: Convert all words to lowercase (default: False).

To find the best parameters for word alignment, use the following command:

```
python word_alignment.py --find_best
```

This will perform an exhaustive search over a pre-defined set of parameter values, and output the best parameter values along with the evaluation score.

## Evaluation

To evaluate the quality of the alignment produced by the program, the `eval()` method can be called on the `WordAlignment` object. This method returns the percentage of Czech words for which at least one correct English translation was identified by the model.

To evaluate the alignment produced by the `--find_best` command, the evaluation score is printed to the console along with the best parameters.

## Results

Best params found are: iterations: 8, number_of_sentences: 3000, threshold: 0.0, lowercase: False.\
Score of eval function: 53.06%.
