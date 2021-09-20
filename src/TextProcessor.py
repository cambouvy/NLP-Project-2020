from nltk.stem import PorterStemmer
import numpy as np
import re
from nltk.tokenize import word_tokenize
import string
from collections import Counter


class TextProcessor():

    def __init__(self):
        """ Baseline text processor"""

    def process(self,text):
        """ Input: List of words [w1,w2,w3,....]"""


        """ Remove all characters that aren't alphabet, space or apostrophe"""
        text = re.sub(r'[^a-zA-Z\s\x27]', '', text)

        """ Tokenize """
        tokens = word_tokenize(text)


        """ Output: List of token"""
        return tokens


    def textToInt(self, text):

        # The unique characters in the file
        vocab = sorted(set(text))

        """Create a mapping from unique characters to indices"""
        char_to_index = {u: i for i, u in enumerate(vocab)}
        index_to_char = np.array(vocab)

        """Convert the text in indices"""
        text_as_index = np.array([char_to_index[c] for c in text])

        return index_to_char, char_to_index, text_as_index