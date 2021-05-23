## sentencizer GENERAL from row to nested sentences format
def importpackages():
    import csv
    import re
    import nltk
    import operator
    import unicodedata
    import unidecode
    from nltk import FreqDist
    from nltk.probability import ConditionalFreqDist
    from collections import Counter
    from collections import defaultdict
    import sklearn_crfsuite
    from sklearn_crfsuite import scorers
    from sklearn_crfsuite import metrics


def fileload(file_a = None, file_b = None, file_c = None, file_d = None):
        A = file_a
        B = file_b
        C = file_c
        D = file_d
        E = file_d

        return([A,B,C,D,E])


def tosentences(twords):
    sent = []
    for tw in twords:
        if tw[1] == '<SB':
            yield sent
            sent = []
        else:
            sent.append(tw)

#wordizer
def towords(sentences):
    d = []
    for sen in sentences:
        for tw in sen:
            d.append(tw)
        d.append(('#','<SB'))
    return d

##word list sentencizer
def wordtosentence(wordlist):
    sent = []
    for word in wordlist:
        if word == '#':
            yield sent
            sent = []
        else:
            sent.append(word)

#gloss list sentencizer
def glosstosentence(glosslist):
    sent = []
    for gloss in glosslist:
        if gloss == '<SB':
            yield sent
            sent = []
        else:
            sent.append(gloss)

def word2features(sent, i, dict, operator, unicodedata):
    word = sent[i][0]
    defaultdict = {'NA': 1, 'NA': 0}

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.strip()': str(unicodedata.normalize('NFD', word).encode('ascii', 'ignore').decode("utf-8")),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.istitle()': word.istitle(),
        'word.pos()': list((dict.get(word,defaultdict))),
        'word.poslast()': max(dict.get(word,defaultdict)),
        'word.posfirst()':list((dict.get(word,defaultdict)).keys())[0],
        'word.posmostcommon()': max((dict.get(word,defaultdict)).items(), key=operator.itemgetter(1))[0]
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.strip()': str(unicodedata.normalize('NFD', word1).encode('ascii', 'ignore').decode("utf-8")),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.pos()': list((dict.get(word1,defaultdict))),
            '-1:word.poslast()': max(dict.get(word1,defaultdict)),
            '-1:word.posfirst()':list((dict.get(word1,defaultdict)).keys())[0],
            '-1:word.posmostcommon()': max((dict.get(word1,defaultdict)).items(), key=operator.itemgetter(1))[0]
        })
    else:
        features['BOS'] = True


    if i > 1:
        word1 = sent[i-2][0]
        features.update({
            '-2:word.lower()': word1.lower(),
            '-2:word.strip()': str(unicodedata.normalize('NFD', word1).encode('ascii', 'ignore').decode("utf-8")),
            '-2:word.istitle()': word1.istitle(),
            '-2:word.pos()': list((dict.get(word1,defaultdict))),
            '-2:word.poslast()': max(dict.get(word1,defaultdict)),
            '-2:word.posfirst()':list((dict.get(word1,defaultdict)).keys())[0],
            '-2:word.posmostcommon()': max((dict.get(word1,defaultdict)).items(), key=operator.itemgetter(1))[0]
        })

    else:
        features['BOS'] = True


    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.strip()': str(unicodedata.normalize('NFD', word1).encode('ascii', 'ignore').decode("utf-8")),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.pos()': list((dict.get(word1,defaultdict))),
            '+1:word.poslast()': max(dict.get(word1,defaultdict)),
            '+1:word.posfirst()':list((dict.get(word1,defaultdict)).keys())[0],
            '+1:word.posmostcommon()': max((dict.get(word1,defaultdict)).items(), key=operator.itemgetter(1))[0]
        })
    else:
        features['EOS'] = True

    if i < len(sent)-2:
        word1 = sent[i+2][0]
        features.update({
            '+2:word.lower()': word1.lower(),
            '+2:word.strip()': str(unicodedata.normalize('NFD', word1).encode('ascii', 'ignore').decode("utf-8")),
            '+2:word.istitle()': word1.istitle(),
            '+2:word.pos()': list((dict.get(word1,defaultdict))),
            '+2:word.poslast()': max(dict.get(word1,defaultdict)),
            '+2:word.posfirst()':list((dict.get(word1,defaultdict)).keys())[0],
            '+2:word.posmostcommon()': max((dict.get(word1,defaultdict)).items(), key=operator.itemgetter(1))[0]
        })


    else:
        features['EOS'] = True

    return features


def sent2features(sent, dict, operator,unicodedata):
    return [word2features(sent, i, dict, operator,unicodedata) for i in range(len(sent))]

def sent2gloss(sent, dict, operator,unicodedata):
    return [gloss for token, gloss in sent]

def sent2token(sent, dict, operator,unicodedata):
    return [token for token,gloss in sent]

def train_crf(X_train, y_train):
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    return crf.fit(X_train, y_train)


####stripped

def strip_accents(text,unicodedata):

    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3
        pass

    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")

    return str(text)



def findrep (string, dictionary,re):
    # sort keys by length, in reverse order
    for item in sorted(dictionary.keys(), key = len, reverse = True):
        string = re.sub(item, dictionary[item], string)
    return string

### listmaker

def openaslist(file = None):
    if file != None:
        file = str(file) + '.csv'

        with open(file, 'r',encoding='utf-8-sig') as csvFile:
            reader = csv.reader(csvFile)
            data = [row for row in csv.reader(csvFile)]
        return data
    else:
        file = str(input('''Enter the file name of your tagged corpus that is in CSV format (no need to type the extension): ''')) + '.csv'

        with open(file, 'r',encoding='utf-8-sig') as csvFile:
            reader = csv.reader(csvFile)
            data = [row for row in csv.reader(csvFile)]
        return data

def permtag(file = None):
    if file != None:
        file = str(file) + '.csv'

        with open(file, 'r',encoding='utf-8-sig') as csvFile:
            reader = csv.reader(csvFile)
            puncdata= [row for row in csv.reader(csvFile)]

        return puncdata
    else:
        file = str(input('''Enter the file name of your tagged corpus that is in CSV format (no need to type the extension): ''')) + '.csv'

        with open(file, 'r',encoding='utf-8-sig') as csvFile:
            reader = csv.reader(csvFile)
            puncdata= [row for row in csv.reader(csvFile)]

        return puncdata


def dictrep(file = None):
    if file != None:
        file = str(file) + '.csv'

        with open(file, 'r',encoding='utf-8-sig') as csvFile:
            reader = csv.reader(csvFile)
            repdict= dict([row for row in csv.reader(csvFile)])

        return repdict
    else:
        file = str(input('''Enter the file name of your tagged corpus that is in CSV format (no need to type the extension): ''')) + '.csv'

        with open(file, 'r',encoding='utf-8-sig') as csvFile:
            reader = csv.reader(csvFile)
            repdict= dict([row for row in csv.reader(csvFile)])

        return repdict


#load and split
def dictsplitterload( file = None):
    if file != None:
        file = str(file) + '.csv'

        with open(file, 'r', encoding='utf-8-sig') as csvFile:
            reader = csv.reader(csvFile)
            dictsplit= [tuple(row) for row in csv.reader(csvFile)]

        return dictsplit
    else:
        file = str(input('''Enter the file name of your Dictionary of Words for Splitting CSV file (no need to type the extension): ''')) + '.csv'

        with open(file, 'r', encoding='utf-8-sig') as csvFile:
            reader = csv.reader(csvFile)
            dictsplit= [tuple(row) for row in csv.reader(csvFile)]

        return dictsplit

def dictcombinerload( file = None):
    if file != None:
        file = str(file) + '.csv'

        with open(file, 'r', encoding='utf-8-sig') as csvFile:
            reader = csv.reader(csvFile)
            dictcombine= [tuple(row) for row in csv.reader(csvFile)]

        return dictcombine
    else:
        file = str(input('''Enter the file name of your CSV file (no need to type the extension): ''')) + '.csv'

        with open(file, 'r', encoding='utf-8-sig') as csvFile:
            reader = csv.reader(csvFile)
            dictcombine= [tuple(row) for row in csv.reader(csvFile)]

        return dictcombine

def openprocessraw( file = None):
    if file != None:
        file = str(file) + '.csv'

        #For Processing
        with open(file, 'r',encoding='utf-8-sig') as csvFile:
            reader = csv.reader(csvFile)
            overall = []
            for row in csv.reader(csvFile):
                overall.append(list(tuple(row)))
        return overall

    else:
        file = str(input('''Enter the file name of your CSV file - the raw unseen data - (no need to type the extension): ''')) + '.csv'

        #For Processing
        with open(file, 'r',encoding='utf-8-sig') as csvFile:
            reader = csv.reader(csvFile)
            overall = []
            for row in csv.reader(csvFile):
                overall.append(list(tuple(row)))
        return overall

def openprocessraworig( file = None):
    if file != None:
        file = str(file) + '.csv'

        #Duplicate
        with open(file, 'r',encoding='utf-8-sig') as csvFile:
            reader = csv.reader(csvFile)
            overallorig = []
            for row in csv.reader(csvFile):
                overallorig.append(list(tuple(row)))
        return overallorig


    else:
        file = str(input('''Enter the file name of your CSV file - the raw unseen data - AGAIN (no need to type the extension): ''')) + '.csv'

        #Duplicate
        with open(file, 'r',encoding='utf-8-sig') as csvFile:
            reader = csv.reader(csvFile)
            overallorig = []
            for row in csv.reader(csvFile):
                overallorig.append(list(tuple(row)))
        return overallorig
