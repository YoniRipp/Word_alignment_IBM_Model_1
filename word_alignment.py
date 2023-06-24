import itertools
from collections import defaultdict, Counter
import numpy as np
import string
import argparse
import gzip
import math
class WordAlignment:

    def __init__(self, file_path, lowercase, threshold, n_sentences, n_iterations, output_file):
        # Initialize class attributes
        self.file_path = file_path
        self.n_sentences = n_sentences
        self.n_iterations = n_iterations
        self.output_file = output_file
        self.sentence_pairs = []  
        self.t = defaultdict(Counter)  
        self.czech_word_order = [] 
        self.threshold = threshold
        self.lowercase = lowercase
        self.dic = {}
        self.Newdic = {}
        self.Bestdic = {}

    def read_sentence_pairs(self):
    # Read sentence pairs from file
        with gzip.open(self.file_path, "rt", encoding="utf-8") as file:
            for line in file:
                line = line.strip().split('\t')
                english_tokens = line[0].split()
                czech_tokens = line[1].split()
                self.sentence_pairs.append((english_tokens, czech_tokens))
                self.czech_word_order.extend(czech_tokens)
                sure_alignments = line[2].split()
                for i in range(len(sure_alignments)):
                    english_cord, czech_cord = sure_alignments[i].split('-')
                    eng_token = english_tokens[int(english_cord)-1]
                    cze_token = czech_tokens[int(czech_cord)-1]
                    if cze_token in self.dic and eng_token not in self.dic[cze_token]:
                        self.dic[cze_token].append(eng_token)
                    else:
                        self.dic[cze_token] = [eng_token]

    def write_dictionary(self, output_file):
        with open(output_file, "w", encoding="utf-8") as file:
            for key, values in self.dic.items():
                file.write(f"{key}: {', '.join(values)}\n")

    def preprocess(self):
        # Limit number of sentences to be processed
        if self.n_sentences != math.inf:
            self.sentence_pairs = list(itertools.islice(self.sentence_pairs, self.n_sentences))
        # Pre-process each sentence pair
        processed_sentence_pairs = []
        self.czech_word_order = []  # Clear the original czech_word_order
        for en, cz in self.sentence_pairs:
            
            if self.lowercase:
                en = [token.lower() for token in en]
                cz = [token.lower() for token in cz]
            
            processed_sentence_pairs.append((en, cz))
            self.czech_word_order.extend(cz)  
        self.sentence_pairs = processed_sentence_pairs

    def initialize_translation_prob(self):
        # Initialize translation probabilities
        for en, cz in self.sentence_pairs:
            for e in en:
                for c in cz:
                    self.t[e][c] = 1.0

        # Normalize translation probabilities
        for e in self.t:
            total = sum(self.t[e].values())
            for c in self.t[e]:
                self.t[e][c] /= total

    def expectation_maximization(self):
        # Perform expectation maximization algorithm
        for _ in range(self.n_iterations):
            counts = defaultdict(Counter)
            total = defaultdict(float)

            for en, cz in self.sentence_pairs:
                for e in en:
                    z = sum(self.t[e][c] for c in cz)
                    for c in cz:
                        c_e = self.t[e][c] / z
                        counts[e][c] += c_e
                        total[c] += c_e

            for e in self.t:
                for c in self.t[e]:
                    self.t[e][c] = counts[e][c] / total[c]

    def invert_translation_prob(self):
        # Invert translation probabilities
        inverted_t = defaultdict(Counter)
        for e in self.t:
            for c in self.t[e]:
                inverted_t[c][e] = self.t[e][c]

        # Normalize inverted probabilities
        for c in inverted_t:
            total = sum(inverted_t[c].values())
            for e in inverted_t[c]:
                inverted_t[c][e] /= total

        return inverted_t

    def best_translation_dictionary(self):
        # Find the best translation for each Czech word
        inverted_t = self.invert_translation_prob()
        with open(self.output_file, "w", encoding="utf-8") as file:
            printed_czech_words = set()
            for c in self.czech_word_order:
                if c in inverted_t and c != "" and c not in printed_czech_words:
                    # Sort the English words by their probabilities and keep only the top 3
                    top_pairs = sorted(inverted_t[c].items(), key=lambda x: -x[1])[:3]
                    # Filter pairs based on the threshold
                    filtered_pairs = [(e, p) for e, p in top_pairs if p >= self.threshold]
                    # Format the English words and their probabilities as a string
                    top_english_words = "  ".join(f"{e}" for e,_ in filtered_pairs if e)
                    self.Newdic[c] = [e for e,_ in filtered_pairs if e]
                    # Write the Czech word and its best translations to the output file
                    file.write(f"{c}: {top_english_words}\n")
                    printed_czech_words.add(c)


    def run(self):
        # Run the word alignment process
        self.read_sentence_pairs()
        self.preprocess()
        self.initialize_translation_prob()
        self.expectation_maximization()
        self.best_translation_dictionary()

    def eval(self):
        
        c = 0
        for key in self.dic.keys():
            if key in self.Newdic:
                for value in self.Newdic[key]:
                    if value in self.dic[key]:
                        c += 1
                        break

        return (c/len(self.dic.keys())) * 100

    def run_word_alignment(args):
        # Create a new WordAlignment object with the given parameters
        word_alignment = WordAlignment(
            file_path=args.file_path,
            n_iterations=args.iterations,
            n_sentences=args.number_of_sentences,
            threshold=args.threshold,
            output_file=args.output_file,
            lowercase=args.lowercase
        )
        word_alignment.run()
        word_alignment.write_dictionary("output/true_dictionary.txt")
        score = round(word_alignment.eval(), 2)
        print(f"Parameters: {vars(args)}\nScore: {score}%\n")

    def run_find_best(args):
        print("Finding best params")
        bscore = 0
        bparams = {}
        # Define a list of all the parameters to check
        params_to_check = [
            {"name": "--file_path", "values": ["input/czenali.gz"]},
            {"name": "--iterations", "values": [3,5,8]},
            {"name": "--number_of_sentences", "values": [2000,3000,5000,math.inf]},
            {"name": "--top", "values": [3]},
            {"name": "--threshold", "values": [0.0,0.2]},
            {"name": "--output_file", "values": ["result_best_dictionary/best_dictionary.txt"]},
            {"name": "--lowercase", "values": [True, False]},
        ]

        # Use argparse to define all the parameters to check
        parser = argparse.ArgumentParser()
        for param in params_to_check:
            parser.add_argument(param["name"], type=type(param["values"][0]), choices=param["values"])

        # Iterate over all possible combinations of parameter values and run program
        for param_values in itertools.product(*[param["values"] for param in params_to_check]):
            params = {}
            for i, param in enumerate(params_to_check):
                params[param["name"]] = param_values[i]
            # Create a new word_alignment object with the given parameter values
            word_alignment = WordAlignment(
                file_path=params["--file_path"],
                n_iterations=params["--iterations"],
                n_sentences=params["--number_of_sentences"],
                threshold=params["--threshold"],
                output_file=params["--output_file"],
                lowercase=params["--lowercase"],
            )
            
            # Run the word alignment process, write the dictionary to file, and evaluate the results
            word_alignment.run()
            word_alignment.best_translation_dictionary()
            score = round(word_alignment.eval(), 2)
            if(bscore < score):
                bscore = score
                bparams = params 
                word_alignment.Bestdic = word_alignment.Newdic
            print("-------------------\n")
            print(f"Parameters: {params}\nScore: {score}%\n")
        print("-------------------")  
        print(f"Best Parameters: {bparams}\nBest score: {bscore}%\n")
        # Create a new word_alignment object with the best parameters values 
        word_alignment = WordAlignment(
                file_path=bparams["--file_path"],
                n_iterations=bparams["--iterations"],
                n_sentences=bparams["--number_of_sentences"],
                threshold=bparams["--threshold"],
                output_file=bparams["--output_file"],
                lowercase=bparams["--lowercase"],

        )
        word_alignment.run()
        word_alignment.best_translation_dictionary()

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--find_best", action="store_true", help="Find the best parameters for word alignment")
    parser.add_argument("--file_path", type=str, default="input/czenali.gz", help="Path to input file")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations for EM algorithm")
    parser.add_argument("--number_of_sentences", type=int, default=2000, help="Limit on number of sentences to be processed")
    parser.add_argument("--top", type=int, default=3, help="Number of top translations to consider for each Czech word")
    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold for translation probability")
    parser.add_argument("--output_file", type=str, default="output/res_dictionary.txt", help="Path to output file")
    parser.add_argument("--lowercase", action="store_true", help="Convert all words to lowercase")
    args = parser.parse_args()

    if args.find_best:
        WordAlignment.run_find_best(args)
    else:
        WordAlignment.run_word_alignment(args)
if __name__ == "__main__":
    main()