import nltk 

book1 = "Data/Pride_and_Prejudice.txt"
book2 = "Data/Mansfield_Park.txt"
book3 = "Data/Sense_and_Sensibility.txt"

trainCorp = nltk.corpus.reader.plaintext.PlaintextCorpusReader(".", book1)
valCorp = nltk.corpus.reader.plaintext.PlaintextCorpusReader(".", book2)
testCorp = nltk.corpus.reader.plaintext.PlaintextCorpusReader(".", book3)
