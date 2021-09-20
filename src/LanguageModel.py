import math

class LanguageModel:
    """ Base class for a language model """
    def __init__(self,size):
        self.size = size

    def train(self,tokens):
        """ Train the language model on the list of tokens in tokens"""
        return

    def calcLogProb(self,ngram):
        """ Calculate probability of the ngram list of tokens p(ngram[-1]|ngram[:-1))"""
        return 0

    def  getSize(self):
        """ Return size of the language model; e.g. 2-gram language model, return 2"""
        return self.size


    def getPPL(self,text):
        text = ["<s>"]*(self.size-1)+text
        sum = 0
        count = 0
        for i in range(self.size,len(text)):
            ngram = text[i-self.size:i]
            logProb =self.calcLogProb(ngram)
            count += 1
            sum += logProb
        ppl = math.pow(2,-1.0/count*sum)
        print("PPL:",ppl)