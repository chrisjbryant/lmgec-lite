import argparse
import kenlm
import os
import re
import spacy
from hunspell import Hunspell

'''
Grammatical Error Correction (GEC) with a language model (LM) and minimal resources.
Corrects: non-word errors, morphological errors, some determiners and prepositions.
Does not correct: missing word errors, everything else.
'''

def main(args):
	# Load various useful resources
	res_dict = loadResources(args)
	# Create output m2 file
	out_sents = open(args.out, "w")

	# Process each tokenized sentence
	with open(args.input_sents) as sents:
		for sent in sents:
			# Keep track of all upper case sentences and lower case them for the LM.
			upper = False
			if sent.isupper(): 
				sent = sent.lower()
				upper = True
			# Strip whitespace and split sent into tokens.
			sent = sent.strip().split()
			# If the line is empty, preserve the newline in output and continue
			if not sent: 
				out_sents.write("\n")
				continue
			# Search for and make corrections while has_errors is true
			has_errors = True
			# Iteratively correct errors one at a time until there are no more.
			while has_errors:
				sent, has_errors = processSent(sent, res_dict, args)
			# Join all the tokens back together and upper case first char
			sent = " ".join(sent)
			sent = sent[0].upper()+sent[1:]
			# Convert all upper case sents back to all upper case.
			if upper: sent = sent.upper()
			# Write the corrected sentence and a newline.
			out_sents.write(sent+"\n")

# Input: Command line args.
# Output: A dictionary of useful resources.
def loadResources(args):
	# Get base working directory.
	basename = os.path.dirname(os.path.realpath(__file__))
	# Language model built by KenLM: https://github.com/kpu/kenlm
	lm = kenlm.Model(args.model)
	# Load spaCy
	nlp = spacy.load("en")
	# Hunspell spellchecker: https://pypi.python.org/pypi/CyHunspell
	# CyHunspell seems to be more accurate than Aspell in PyEnchant, but a bit slower.
	gb = Hunspell("en_GB-large", hunspell_data_dir=basename+'/resources/spelling/')
	# Inflection forms: http://wordlist.aspell.net/other/
	gb_infl = loadWordFormDict(basename+"/resources/agid-2016.01.19/infl.txt")
	# List of common determiners
	det = {"", "the", "a", "an"}
	# List of common prepositions
	prep = {"", "about", "at", "by", "for", "from", "in", "of", "on", "to", "with"}
	# Save the above in a dictionary:
	res_dict = {"lm": lm,
				"nlp": nlp,
				"gb": gb,
				"gb_infl": gb_infl,
				"det": det,
				"prep": prep}
	return res_dict

# Input: Path to Automatically Generated Inflection Database (AGID)
# Output: A dictionary; key is lemma, value is a set of word forms for that lemma
def loadWordFormDict(path):
	entries = open(path).read().strip().split("\n")
	form_dict = {}
	for entry in entries:
		entry = entry.split(": ")
		key = entry[0].split()
		forms = entry[1]
		# The word lemma
		word = key[0]
		# Ignore some of the extra markup in the forms
		forms = re.sub("[012~<,_!\?\.\|]+", "", forms)
		forms = re.sub("\{.*?\}", "", forms).split()
		# Save the lemma and unique forms in the form dict
		form_dict[word] = set([word]+forms)
	return form_dict

# Input 1: The sentence we want to correct; i.e. a list of token strings
# Input 2: A dictionary of useful resources
# Input 3: Command line args
# Output 1: The input sentence or a corrected sentence
# Output 2: Boolean whether the sentence needs more corrections
def processSent(sent, res_dict, args):
	# Process sent with spacy
	proc_sent = processWithSpacy(sent, res_dict["nlp"])
	# Calculate avg token prob of the sent so far.
	orig_prob = res_dict["lm"].score(proc_sent.text, bos=True, eos=True)/len(proc_sent)
	# Store all the candidate corrected sentences here
	cand_dict = {}
	# Process each token.
	for tok in proc_sent:
		# SPELLCHECKING
		# Spell check: tok must be alphabetical and not a real word.
		if tok.text.isalpha() and not res_dict["gb"].spell(tok.text):
			cands = res_dict["gb"].suggest(tok.text)
			# Generate correction candidates
			if cands: cand_dict.update(generateCands(tok.i, cands, sent, args.threshold))
		# MORPHOLOGY
		if tok.lemma_ in res_dict["gb_infl"]:
			cands = res_dict["gb_infl"][tok.lemma_]
			cand_dict.update(generateCands(tok.i, cands, sent, args.threshold))
		# DETERMINERS
		if tok.text in res_dict["det"]:
			cand_dict.update(generateCands(tok.i, res_dict["det"], sent, args.threshold))
		# PREPOSITIONS
		if tok.text in res_dict["prep"]:
			cand_dict.update(generateCands(tok.i, res_dict["prep"], sent, args.threshold))

	# Keep track of the best sent if any
	best_prob = float("-inf")
	best_sent = []
	# Loop through the candidate edits; edit[-1] is the error type weight
	for edit, cand_sent in cand_dict.items():
		# Score the candidate sentence
		cand_prob = res_dict["lm"].score(" ".join(cand_sent), bos=True, eos=True)/len(cand_sent)
		# Compare cand_prob against weighted orig_prob and best_prob
		if cand_prob > edit[-1]*orig_prob and cand_prob > best_prob:
			best_prob = cand_prob
			best_sent = cand_sent
	# Return the best sentence and a boolean whether to search for more errors
	if best_sent: return best_sent, True
	else: return sent, False

# Input 1: A list of token strings.
# Input 2: A spacy processing object
# Output: A spacy processed sentence.
def processWithSpacy(sent, nlp):
	proc_sent = nlp.tokenizer.tokens_from_list(sent)
	nlp.tagger(proc_sent)
	return proc_sent
	
# Input 1: A token index indicating the target of the correction.
# Input 2: A list of candidate corrections for that token.
# Input 3: The current sentence as a list of token strings.
# Input 4: An error type weight
# Output: A dictionary. Key is a tuple: (tok_id, cand, weight),
# value is a list of strings forming a candidate corrected sentence.
def generateCands(tok_id, cands, sent, weight):
	# Save candidates here.
	edit_dict = {}
	# Loop through the input alternative candidates
	for cand in cands:
		# Copy the input sentence
		new_sent = sent[:]
		# Change the target token with the current cand
		new_sent[tok_id] = cand
		# Remove empty strings from the list (for deletions)
		new_sent = list(filter(None, new_sent))
		# Give the edit a unique identifier
		edit_id = (tok_id, cand, weight)
		# Save non-empty sentences
		if new_sent: edit_dict[edit_id] = new_sent
	return edit_dict

if __name__ == "__main__":
	# Define and parse program input
	parser = argparse.ArgumentParser()
	parser.add_argument("input_sents", help="A text file containing 1 tokenized sentence per line.")
	parser.add_argument("-mdl", "--model", help="The path to a KenLM model file.", required=True)	
	parser.add_argument("-o", "--out", help="The output correct text file, 1 tokenized sentence per line.", required=True)
	parser.add_argument("-th", "--threshold", help="LM percent improvement threshold. Default: 0.96 requires scores to be at least 4% higher than the original.", type=float, default=0.96)
	args = parser.parse_args()
	# Run the program.
	main(args)
