PYTHON = python
ARGS = --file_path input/czenali.gz --iterations 5 --number_of_sentences 2000 --threshold 0.0 --top 3 --output_file output/res_dictionary.txt

all: run

run:
	$(PYTHON) word_alignment.py $(ARGS)

clean:
	del output\true_dictionary.txt output\res_dictionary.txt 

.PHONY: all run clean
