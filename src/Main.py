import sys, argparse
from TextProcessor import TextProcessor
from NGramLanguageModel import NGramLanguageModel
from RNNLanguageModel import RNNLanguageModel


def main():
    parser = argparse.ArgumentParser(description='Script to train a language model')
    parser.add_argument("--song_data", default="../data/processed/popSongs.txt", type=str, help="text file containing the training data")

    args = parser.parse_args()
    processor = TextProcessor()


    f = open(args.song_data, encoding="utf-8")

    lyrics = f.read(100000)

    """N-gram model"""
    ngram_model = NGramLanguageModel(4)
    tokens = processor.process(lyrics)
    print("Train language model ....")
    ngram_model.train(tokens)
    print("Language model trained")
    print("Generating song ....")
    predictedSong = ngram_model.predictSong("I", 50)
    print("Predicted song: ")
    print(predictedSong)
    ngram_model.getPPL(predictedSong.split())





if __name__ == "__main__":
    main()