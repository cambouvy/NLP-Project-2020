import math
import random
from LanguageModel import LanguageModel
import scipy
import numpy as np
import bisect
import nltk

class NGramLanguageModel(LanguageModel):


    def __init__(self, size):
        self.size = size
        self.n_1gram = {}
        self.ngram = {}
        self.goodTuringProbabilities = {}
        self.totalNgramCounts = 0

    def train(self,tokens):
        """ Train the language model on the list of tokens in tokens"""

        self.types = {}


        if self.size != 1:
            '#Keeps track of the previously encountered words to avoid inner loops'
            words = ''

            for i in tokens[self.size-1:]:

                '#Compute the number of types in tokens'
                if i in self.types:
                    self.types[i] += 1
                else:
                    self.types[i] = 1

                '#String composed of the n-1 words'
                seriesN_1 = ''
                '#String composed of the n-1 words + the current one'
                seriesN = ''


                '#Only loop backwards the first n-1 times'
                if len(words.split()) < self.size-1:
                    '#Dict of n-1 tokens series with their frequency count'
                    for j in range(self.size-1, 0, -1):
                        seriesN_1 += tokens[tokens.index(i)-j] + ' '
                    '# Dict of n tokens series with their frequency count'
                    for j in range(self.size - 1, -1, -1):
                        seriesN += tokens[tokens.index(i) - j] + ' '
                    if len(words) is not 0:
                        words += ' '
                    words += i

                else:
                    seriesN_1 = words
                    words += ' ' +i
                    seriesN = words
                    newWords = words.split(' ')
                    words = ' '.join(newWords[1:])



                if seriesN_1 in self.n_1gram:
                    self.n_1gram[seriesN_1] += 1
                else:
                    self.n_1gram[seriesN_1] = 1



                if seriesN in self.ngram:
                    self.ngram[seriesN] += 1
                else:
                    self.ngram[seriesN] = 1


        else:
            for i in tokens:
                if i in self.types:
                    self.types[i] += 1
                else:
                    self.types[i] = 1
            self.ngram = self.types


        self.vocSize = len(self.types)
        return


    def calcLogProb(self,ngram):
        """ Calculate probability of the ngram list of tokens p(ngram[-1]|ngram[:-1))"""


        ngramString = ' '.join(ngram)
        n_1gramString = ' '.join(ngram[:-1])

        nCount = 0
        n_1Count = 0

        if self.size is 1:
            if ngramString in self.ngram:
                nCount = self.ngram[ngramString]
            n_1Count = self.totalNgramCounts

        else:
            if ngramString in self.ngram:
                nCount = self.ngram[ngramString]
                n_1Count = self.n_1gram[n_1gramString]

            else:
                if n_1gramString in self.n_1gram:
                    n_1Count = self.n_1gram[n_1gramString]


        '# Laplace Smoothing'
        nCount += 1
        n_1Count += 1*self.vocSize

        '#If there is no smoothing'
        if nCount is 0:
            return 0

        prob = math.log2(nCount / n_1Count)

        return prob



    def calcProb(self,ngram):
        """ Calculate probability of the ngram list of tokens p(ngram[-1]|ngram[:-1))"""

        ngramString = ' '.join(ngram)
        n_1gramString = ' '.join(ngram[:-1])

        nCount = 0
        n_1Count = 0

        if self.size is 1:
            if ngramString in self.ngram:
                nCount = self.ngram[ngramString]
            n_1Count = self.totalNgramCounts

        else:
            if ngramString in self.ngram:
                nCount = self.ngram[ngramString]
                n_1Count = self.n_1gram[n_1gramString]
            else:
                if n_1gramString in self.n_1gram:
                    n_1Count = self.n_1gram[n_1gramString]


        '# Laplace Smoothing'
        nCount += 1
        n_1Count += 1*self.vocSize

        '#If there is no smoothing'
        if nCount is 0:
            return 0
        prob = nCount/n_1Count

        return prob




    def predictSong(self, initialWord, size):
        song = initialWord + ' '
        k = 20
        p = 0.95
        generatedNgram = []
        for i in range(size):
            maxProb = 0
            if i is 0:
                currentWord = initialWord
                allProbs = {}
                sumOfProbs = 0
                for seq in self.n_1gram:
                    seq = seq.strip()
                    ngram = currentWord + ' ' + seq
                    ngram = ngram.split(" ")
                    prob = self.calcProb(ngram)
                    allProbs[seq] =  prob
                    sumOfProbs += prob

                #for unigrams
                if self.size == 1:
                    for token in self.ngram:
                        allProbs[token] = self.calcProb(token)

                """ Greedy """
                #bestN_1gram = self.greedy(allProbs)


                """ Select randomly a word within the k most probable ones """
                #bestN_1gram = self.selectKMostProbable(allProbs,k)

                """Sample from the distribution"""
                #bestN_1gram = self.sampleFromDistribution(allProbs, sumOfProbs)

                """Top-p sampling"""
                #bestN_1gram = self.nucleusSampling(allProbs, sumOfProbs,p)

                """Top k-sampling"""
                bestN_1gram = self.topKSampling(allProbs, sumOfProbs, k)

                song = song + bestN_1gram
                currentN_gram = bestN_1gram.split(' ')
                generatedNgram.append(currentN_gram)

            else:
                allProbs = {}
                sumOfProbs = 0
                for word in self.types:
                    ngram = currentN_gram.copy()
                    ngram.append(word)
                    prob = self.calcProb(ngram)
                    allProbs[word] = prob
                    sumOfProbs += prob


                """ Greedy """
                #bestWord = self.greedy(allProbs)

                """ Select randomly a word within the k most probable ones """
                #bestWord = self.selectKMostProbable(allProbs,k)

                """Sample from the distribution"""
                #bestWord = self.sampleFromDistribution(allProbs, sumOfProbs)

                """Top-p sample"""
                #bestWord = self.nucleusSampling(allProbs, sumOfProbs,p)

                """Top-k sample"""
                bestWord = self.topKSampling(allProbs, sumOfProbs,k)


                song = song + ' ' + bestWord
                currentN_gram.append(bestWord)
                generatedNgram.append(currentN_gram)
                currentN_gram = currentN_gram[1:]

        print('BLEUscore:',self.selfBleuScore(song, 5))
        return song


    def greedy(self, allProbs):
        """ Greedy: pick the word with highest probability"""
        return max(allProbs, key=allProbs.get)


    def selectKMostProbable(self,allProbs,k):
        """ Select randomly a word within the k most probable n_1grams """
        rand = random.randint(1, k)
        sortedProbs = {ngram: prob for ngram, prob in sorted(allProbs.items(), key=lambda item: item[1], reverse=True)}
        best = list(sortedProbs)[rand]
        return best

    def sampleFromDistribution(self, allProbs, sumOfProbs):
        # Normalize the MLE probabilities
        for seq in allProbs:
            allProbs[seq] = allProbs[seq] / sumOfProbs
        # Sort the probabilities
        sortedProbs = {ngram: prob for ngram, prob in sorted(allProbs.items(), key=lambda item: item[1], reverse=False)}
        # Create the cdf
        cumulProbs = []
        cumul = 0;
        for x in sortedProbs.values():
            cumul = cumul + x;
            cumulProbs.append(cumul);
        # Renormalize
        for i in range(len(cumulProbs)):
            cumulProbs[i] = cumulProbs[i] / cumul
        rand = random.random()
        index = bisect.bisect(cumulProbs, rand)
        best = list(sortedProbs)[index]
        return best


    def nucleusSampling(self, allProbs, sumOfProbs,p):
        """ Sample from the n_1grams having CDF > p"""
        # Normalize the MLE probabilities
        for seq in allProbs:
            allProbs[seq] = allProbs[seq] / sumOfProbs
        # Sort the probabilities in descending order
        sortedProbs = {ngram: prob for ngram, prob in sorted(allProbs.items(), key=lambda item: item[1], reverse=False)}

        # Create the cdf
        cumulProbs = []
        cumul = 0;
        for x in sortedProbs.values():
            cumul = cumul + x
            cumulProbs.append(cumul)
        # Renormalize
        cumulProbs = [cumulProb/cumul for cumulProb in cumulProbs]



        # Cutoff the CDF to have only the probs > p
        cnt = 0
        for cumulProb in cumulProbs:
            if cumulProb >= p:
                cnt +=1
        ngramsCutoff = list(sortedProbs)[-cnt:]
        probsCutoff = list(sortedProbs.values())[-cnt:]

        # Recreate the CDF
        cumulProbsCutoff = []
        cumul = 0;
        for x in probsCutoff:
            cumul = cumul + x
            cumulProbsCutoff.append(cumul)
        # Renormalize
        cumulProbsCutoff = [cumulProb / cumul for cumulProb in cumulProbsCutoff]

        # Sample from the new distribution
        rand = random.random()
        index = bisect.bisect(cumulProbsCutoff, rand)
        best = ngramsCutoff[index]
        return best


    def topKSampling(self, allProbs, sumOfProbs,k):
        """ Sample from the k ngrams having highest distribution"""
        # Normalize the MLE probabilities
        for seq in allProbs:
            allProbs[seq] = allProbs[seq] / sumOfProbs
        # Sort the probabilities in descending order
        sortedProbs = {ngram: prob for ngram, prob in sorted(allProbs.items(), key=lambda item: item[1], reverse=True)}

        topNgrams = list(sortedProbs)[:k]
        topProbs = list(sortedProbs.values())[:k]

        # Create the cdf
        cumulProbs = []
        cumul = 0;
        for x in topProbs:
            cumul = cumul + x
            cumulProbs.append(cumul)
        # Renormalize
        cumulProbs = [cumulProb/cumul for cumulProb in cumulProbs]


        # Sample from the new distribution
        rand = random.random()
        index = bisect.bisect(cumulProbs, rand)
        best = topNgrams[index]
        return best


    def selfBleuScore(self,generatedSong, seqLength):
        generatedNgrams = []
        generatedSong = generatedSong.split(' ')
        for i in range(seqLength, len(generatedSong)):
            generatedNgrams.append(generatedSong[i - seqLength:i])
        bleuScores = []
        f = nltk.translate.bleu_score.SmoothingFunction()
        for genNgram in generatedNgrams:
            for genNgram2 in generatedNgrams:
                if genNgram != genNgram2:
                    bScore = nltk.translate.bleu_score.sentence_bleu([genNgram], genNgram2, smoothing_function=f.method1)
                    bleuScores.append(bScore)
        return sum(bleuScores) / len(bleuScores)
