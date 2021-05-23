import csv, re, nltk, operator, unicodedata, unidecode, sklearn_crfsuite, subprocess, os, sys, argparse, random, warnings, pickle, statistics
from nltk import FreqDist
from collections import Counter
from os import path
from nltk.probability import ConditionalFreqDist
from nltk.corpus import words
from collections import Counter, defaultdict
from sklearn_crfsuite import scorers, metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from tqdm import tqdm
import string as stringg
import tkinter as tk
import numpy as np
import pandas as pd

##GUI
root=tk.Tk()
root.geometry("700x350") # setting the windows size
corpus_training_var=tk.StringVar() # declaring string variable
corpus_unseen_var=tk.StringVar()
splitdict_var = tk.StringVar()
stringrep_var = tk.StringVar()
sourcelang_var = tk.StringVar()
modelsav_var = tk.StringVar()
usemodelsav_var = tk.StringVar()
filetype_var = tk.StringVar()
splitproportion_var = tk.StringVar()
crossvalbin_var = tk.StringVar()
evalonly_var = tk.StringVar()

def submit():
    global corpus_training
    global corpus_unseen
    global splitdict
    global stringrep
    global sourcelang
    global modelsav
    global splitproportion
    global filetype
    global usemodelsav
    global crossvalbin
    global evalonly


    corpus_training = corpus_training_var.get()
    corpus_unseen = corpus_unseen_var.get()
    splitdict = splitdict_var.get()
    stringrep = stringrep_var.get()
    sourcelang = sourcelang_var.get()
    modelsav = modelsav_var.get()
    usemodelsav = usemodelsav_var.get()
    splitproportion = splitproportion_var.get()
    crossvalbin = crossvalbin_var.get()
    evalonly = evalonly_var.get()
    filetype = [filetype_entry.get(i) for i in filetype_entry.curselection()]

    newlinefiletype = "".join([str(idx+1) + '.' + "\t" + item.replace(")","").replace("(","") + '\n' for idx,item in enumerate(filetype)])

    root.destroy()
    print("\n")
    print("------------------------------")
    print("INPUT SUMMARY")
    print("------------------------------")
    print("Training data (tagged data) file: " + corpus_training)
    print("Unseen data (data to be tagged) file: " + corpus_unseen)
    print("Splitting dictionary: " +  splitdict)
    print("String replacement dictionary: " +  stringrep)
    print("\n")
    print("Tag source language (English/Tagalog/Hokkien-Others): " +  sourcelang)
    print("Save model: " +  modelsav)
    print("Use existing saved model: " +  usemodelsav)
    print("Files to be generated:\n" + newlinefiletype)
    print("\n")
    print("Split proportion: " +  splitproportion)
    print("Cross-validation: " +  crossvalbin)
    print("Evaluate only and do not tag: " +  evalonly)
    print("------------------------------")
    print("\n")

root.title('General POS Tagger/ Trainer and Source Language Identifier') #set title

corpus_training_label = tk.Label(root, text = 'Tagged Corpus CSV File', font=('calibre',10, 'bold'))
corpus_training_entry = tk.Entry(root,textvariable = corpus_training_var, font=('calibre',10,'normal'), width  = 40)
corpus_training_var.set("annotateddata") # set default values

corpus_unseen_label = tk.Label(root, text = 'Untagged Corpus CSV File (For Tagging)', font = ('calibre',10,'bold'))
corpus_unseen_entry=tk.Entry(root, textvariable = corpus_unseen_var, font = ('calibre',10,'normal'), width  = 40)
corpus_unseen_var.set("untaggeddata")

splitdict_label1 = tk.Label(root, text = 'Splitting Dictionary File', font = ('calibre',10,'bold'))
splitdict_label2 = tk.Label(root, text = '(Leave blank if you do not have or if you do not want to split.)', font = ('calibre',10))
splitdict_entry=tk.Entry(root, textvariable = splitdict_var, font = ('calibre',10,'normal'), width  = 40)
splitdict_var.set("dictsplit") # default value

stringrep_label1 = tk.Label(root, text = 'String Replacement Dictionary File', font = ('calibre',10,'bold'))
stringrep_label2 = tk.Label(root, text = '(Leave blank if you do not have or if you do not want to replace anything.)', font = ('calibre',10))
stringrep_entry=tk.Entry(root, textvariable = stringrep_var, font = ('calibre',10,'normal'), width  = 40)
stringrep_var.set("stringreplace") # default value

sourcelang_label = tk.Label(root, text = 'I want to include the source language tag (Eng/Tag/Hok-Oth) in the output.', font = ('calibre',10,'bold'))
sourcelang_entry = tk.OptionMenu(root, sourcelang_var, "yes", "no")
sourcelang_var.set("yes") # default value

modelsav_label = tk.Label(root, text = 'I want to save the model.', font = ('calibre',10,'bold'))
modelsav_entry = tk.OptionMenu(root, modelsav_var, "yes", "no")
modelsav_var.set("yes") # default value

usemodelsav_label = tk.Label(root, text = 'I have an existing model and want to use it.', font = ('calibre',10,'bold'))
usemodelsav_entry = tk.OptionMenu(root, usemodelsav_var, "yes", "no")
usemodelsav_var.set("no") # default value

filetype_label = tk.Label(root, text = 'Generate the following files (multiple select):', font = ('calibre',10,'bold'))
filetype_entry = tk.Listbox(root, listvariable=filetype_var, selectmode = "multiple", font = ('calibre',10,'normal'), width  = 40, height = 4)

list_items = ["Plain tagged corpus - word as row with diacritics", "Plain tagged corpus - sentence as row with diacritics", "Plain tagged corpus - sentence as row without diacritics", "Full tagged corpus - sentence as row"]
for item in list_items:
    filetype_entry.insert('end', item)


splitproportion_label = tk.Label(root, text = 'Proportion (Training)', font = ('calibre',10,'bold'))
splitproportion_entry=tk.Entry(root, textvariable = splitproportion_var, font = ('calibre',10,'normal'), width  = 40)
splitproportion_var.set("0.8")

crossvalbin_label = tk.Label(root, text = 'I want to cross-validate my model. (Takes longer)', font = ('calibre',10,'bold'))
crossvalbin_entry = tk.OptionMenu(root, crossvalbin_var, "yes", "no")
crossvalbin_var.set("no") # default value

evalonly_label = tk.Label(root, text = 'I only want to evaluate my model.', font = ('calibre',10,'bold'))
evalonly_entry = tk.OptionMenu(root, evalonly_var, "yes", "no")
evalonly_var.set("no") # default value

sub_btn=tk.Button(root,text = 'Submit', command = submit) # creating a button using the widget button that will call the submit function

corpus_training_label.grid(row=0,column=0, sticky = 'W') # placing the label and entry in the required position using grid method
corpus_training_entry.grid(row=0,column=1, sticky = 'W')
corpus_unseen_label.grid(row=1,column=0, sticky = 'W')
corpus_unseen_entry.grid(row=1,column=1, sticky = 'W')
splitdict_label1.grid(row=2,column=0, sticky = 'W')
splitdict_entry.grid(row=2,column=1, sticky = 'W')
splitdict_label2.grid(row=3,column=0, sticky = 'W')
stringrep_label1.grid(row=4,column=0, sticky = 'W')
stringrep_entry.grid(row=4,column=1, sticky = 'W')
stringrep_label2.grid(row=5,column=0, sticky = 'W')
sourcelang_label.grid(row=6,column=0, sticky = 'W')
sourcelang_entry.grid(row=6,column=1, sticky = 'W')
modelsav_label.grid(row=7,column=0, sticky = 'W')
modelsav_entry.grid(row=7,column=1, sticky = 'W')
usemodelsav_label.grid(row=8,column=0, sticky = 'W')
usemodelsav_entry.grid(row=8,column=1, sticky = 'W')
filetype_label.grid(row=9,column=0, sticky = 'W')
filetype_entry.grid(row=9,column=1, sticky = 'W')

splitproportion_label.grid(row=14,column=0, sticky = 'W')
splitproportion_entry.grid(row=14,column=1, sticky = 'W')
crossvalbin_label.grid(row=15,column=0, sticky = 'W')
crossvalbin_entry.grid(row=15,column=1, sticky = 'W')
evalonly_label.grid(row=16,column=0, sticky = 'W')
evalonly_entry.grid(row=16,column=1, sticky = 'W')

sub_btn.grid(row=16,column=1, sticky = 'S')

root.mainloop() # performing an infinite loop for the window to display


## Loading
current =  os.getcwd()
os.chdir(current)
exec(open("functions_pack.py").read())
files = fileload(stringrep,corpus_training,splitdict,corpus_unseen)


if sourcelang == "yes":
    word_list = words.words() #English Dictionary (from nltk package)
    eng_dic = {}
    for index,word in enumerate(word_list):
        eng_dic[word] = index
    with open('tagalog_dict.txt', 'r',encoding='utf-8-sig') as txtFile: #Tagalog Dictionary from text file in folder
        contents = txtFile.read()
        tagwordlist = contents.split('\n')
    tag_dic = {}
    for index,word in enumerate(tagwordlist):
        tag_dic[word] = index


if stringrep == "":
    repdict = dictrep('stringreplace_default')
else:
    repdict = dictrep(files[0])


##Loading Tagged Corpus
puncdata = permtag('permtag')
data = openaslist(files[1])
dat = []
for row in data:
    newrow = []
    replacedword = findrep(row[0], repdict, re)
    newrow.append(replacedword)
    newrow.append(row[1])
    dat.append(newrow)
data = dat
joineddata = puncdata + data
data = joineddata

#Flattening diacritics, if there are any, for model Trainingfor row in data:
for row in data:
    unacc_word = strip_accents(row[0],unicodedata)
    row.remove(row[0])
    row.append(unacc_word)
    row.reverse()
data = [tuple(row) for row in data]

##Creating frequency dictionaries
freqdict = {}
for (word, pos) in data:
    if word not in freqdict:
        freqdict[word] = {}
    if pos not in freqdict[word]:
        freqdict[word][pos] = 1
    else:
        freqdict[word][pos] += 1
glossdict = {}
for (word, pos) in data:
    if pos not in glossdict:
        glossdict[pos] = {}

##loading the splitter dictionary
if splitdict == "":
    dictsplit = dictsplitterload('dictsplit_default')
else:
    dictsplit = dictsplitterload(files[2])
dictionarysplit = {}
for k,v in dictsplit:
    dictionarysplit[k]=v

##Model Training
sentences = list(tosentences(data)) #Sentencize the tagged corpus data.
print("\n")
print("------------------------------")
print("ANNOTATED DATA SUMMARY")
print("------------------------------")
print("Sentences:", len(sentences), "Words:", len(data)) #Check out number of words and sentences.
print("------------------------------")
print("\n")

#Split the model data into two parts: training, and dev
n = int(len(sentences)*float(splitproportion))

random.seed(222)
shuffled_sentences = sentences
random.shuffle(shuffled_sentences)
training = shuffled_sentences[:n]
dev = shuffled_sentences[n:]


## Vectorize

x_train = [sent2features(s,freqdict, operator, unicodedata) for s in training]
y_train = [sent2gloss(s,freqdict, operator, unicodedata) for s in training]
x_dev = [sent2features(s,freqdict, operator, unicodedata) for s in dev]
y_dev = [sent2gloss(s,freqdict, operator, unicodedata) for s in dev]
x_all = [sent2features(s,freqdict, operator, unicodedata) for s in shuffled_sentences]
y_all = [sent2gloss(s,freqdict, operator, unicodedata) for s in shuffled_sentences]

# Comparison
if ((path.exists('model.sav')) and (usemodelsav == "yes")):
# load the model from disk
    print("Using saved model.")
    loaded_model = pickle.load(open('model.sav', 'rb'))
    crf = loaded_model
    print("Saved model loaded.")

    labels = list(crf.classes_)
    warnings.filterwarnings('ignore')

else:

    ## Training Proper
    import scipy
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    params_space = {
    'c1': scipy.stats.expon(scale=0.1),
    'c2': scipy.stats.expon(scale=0.1),
}


    print("Fitting a model on the training data...")
    # Prediction
    crf.fit(x_train, y_train)


    labels = list(crf.classes_)

    #warnings.filterwarnings('ignore')

    if modelsav == "yes":
        filename = 'model.sav'
        pickle.dump(crf, open(filename, 'wb'))
        print("Model saved.")

# save the model to disk
print("Predicting using model.")
y_pred = crf.predict(x_dev)

print("------------------------------")
print("Sample predictions:\n\n")
print("Actual glosses:")
print(y_dev[0])
print("Predicted glosses:")
print(y_pred[0])
print("\nActual glosses:")
print(y_dev[12])
print("Predicted glosses:")
print(y_pred[12])
print("------------------------------")

print(metrics.flat_classification_report(
    y_dev, y_pred, labels=sorted(labels), digits=3
))


if crossvalbin == 'yes':
    print("Predictions complete. Evaluating and cross-validating model...")
    print("\n")
    print("\n")
else:
    print("Predictions complete. Evaluating model... (Please wait)")
    print("\n")
    print("\n")


metric_f1 = metrics.flat_f1_score(y_dev, y_pred, average='macro', labels=np.unique(labels))
metric_accuracy = metrics.flat_accuracy_score(y_dev, y_pred)
metric_precision = metrics.flat_precision_score(y_dev, y_pred, average='macro', labels=np.unique(labels))
metric_recall = metrics.flat_recall_score(y_dev, y_pred, average='macro', labels=np.unique(labels))


print("------------------------------")
print("MODEL EVALUATION RESULTS (Development Data)")
print("------------------------------")
print("Proportion: ", float(splitproportion)*100,"-",100-float(splitproportion)*100, "\nSentences trained from:", len(training), "\nSentences used for evaluation:", len(dev))
print("\n")
print("F1: \t\t", round(float(metric_f1),4))
print("Accuracy: \t", round(float(metric_accuracy),4))
print("Precision: \t", round(float(metric_precision),4))
print("Recall: \t", round(float(metric_recall),4))
print("------------------------------")
print("\n")
print("Evaluation complete.")


if crossvalbin == 'yes':
    print("Cross-validating now... This will take a while...")
    import fractions
    warnings.filterwarnings('ignore')
    n = int(float(str(fractions.Fraction(splitproportion).limit_denominator()).split('/')[1]))

    f1_scorer = make_scorer(metrics.flat_f1_score, average = 'macro', labels=np.unique(labels))
    accuracy_scorer = make_scorer(metrics.flat_accuracy_score)
    precision_scorer = make_scorer(metrics.flat_precision_score, average = 'macro', labels=np.unique(labels))
    recall_scorer = make_scorer(metrics.flat_recall_score, average = 'macro', labels=np.unique(labels))


    cv_f1 = cross_validate(crf, x_all, y_all, cv = n, scoring = f1_scorer)
    print("Unaveraged F1 scores:\t\t", cv_f1['test_score'])
    cv_accuracy = cross_validate(crf, x_all, y_all, cv = n, scoring = accuracy_scorer)
    print("Unaveraged accuracy scores:\t", cv_accuracy['test_score'])
    cv_precision = cross_validate(crf, x_all, y_all, cv = n, scoring = precision_scorer)
    print("Unaveraged precision scores:\t", cv_precision['test_score'])
    cv_recall = cross_validate(crf, x_all, y_all, cv = n, scoring = recall_scorer)
    print("Unaveraged recall scores:\t", cv_recall['test_score'])

    print("--------------------------------------------------------------------")
    print("                     CROSS-VALIDATION RESULTS")
    print("--------------------------------------------------------------------")
    print("Proportion: ", float(splitproportion)*100,"-",100-float(splitproportion)*100, "\nSentences trained from:", len(training), "\nSentences used for evaluation:", len(dev))
    print("Number of folds: ", n)
    print("\n")
    print("F1 (cross-val): \t", round(statistics.mean(list(cv_f1['test_score'])),5), "\tSD: \t",round(statistics.stdev(list(cv_f1['test_score'])),5))
    print("Accuracy (cross-val): \t", round(statistics.mean(list(cv_accuracy['test_score'])),5), "\tSD: \t",round(statistics.stdev(list(cv_accuracy['test_score'])),5))
    print("Precision (cross-val): \t", round(statistics.mean(list(cv_precision['test_score'])),5), "\tSD: \t",round(statistics.stdev(list(cv_precision['test_score'])),5))
    print("Recall (cross-val): \t", round(statistics.mean(list(cv_recall['test_score'])),5), "\tSD: \t",round(statistics.stdev(list(cv_recall['test_score'])),5))
    print("--------------------------------------------------------------------")
    print("\n")
    print("\n")


    print("--------------------------------------------------------------------")
    print("                           COMPARISON")
    print("--------------------------------------------------------------------")

    print("metric\t\ttrained model\tave. k-fold\tdifference")
    print("--------------------------------------------------------------------")
    print("F1 score\t",round(float(metric_f1),4), "\t", round(statistics.mean(list(cv_f1['test_score'])),5), "\t",round(round(float(metric_f1),4)-round(statistics.mean(list(cv_f1['test_score'])),5),5))
    print("Accuracy score\t",round(float(metric_accuracy),4), "\t", round(statistics.mean(list(cv_accuracy['test_score'])),5), "\t",round(round(float(metric_accuracy),4)-round(statistics.mean(list(cv_accuracy['test_score'])),5),5))
    print("Precision score\t",round(float(metric_precision),4), "\t", round(statistics.mean(list(cv_precision['test_score'])),5), "\t",round(round(float(metric_precision),4)-round(statistics.mean(list(cv_precision['test_score'])),5),5))
    print("Recall score\t",round(float(metric_recall),4), "\t", round(statistics.mean(list(cv_recall['test_score'])),5), "\t",round(round(float(metric_recall),4)-round(statistics.mean(list(cv_recall['test_score'])),5),5))

    print("--------------------------------------------------------------------")


    print("Cross-validation complete.")

    print("\n")
    print("\n")

if evalonly == "no":


    #### Converting Predictions to Nested Sentence Format

    predsentcor = [] # With corrected data beside
    for i, s in enumerate(x_dev):
        sentence = []
        for j, w in enumerate(s):
            sentence.append((w['word.lower()'],y_pred[i][j],y_dev[i][j]))
        predsentcor.append(sentence)

    predsent = [] #> Without corrected data beside
    for i, s in enumerate(x_dev):
        sentence = []
        for j, w in enumerate(s):
            sentence.append((w['word.lower()'],y_pred[i][j]))
        predsent.append(sentence)

    dat = towords(predsent) # Convert NSF to CSV format (if needed)
    with open('output.csv','w') as out:
        for w,pos in dat:
            out.write(w+','+pos+'\n')

    ### Loading the New Data (Unseen Data)
    overallorig = openprocessraworig(files[3])
    lastrow = len(overallorig[0])-1

    newoverall=[]
    for row in overallorig:
        newrow = []
        for i,item in enumerate(row):
            if i < lastrow:
                newrow.append(row[i])
            elif i == lastrow:
                newword = findrep(row[lastrow], repdict, re)
                newrow.append(newword)
            else:
                newrow.append(row[i])
        newoverall.append(newrow)
    overallorig = newoverall


    overall = openprocessraw(files[4])

    newoverall=[]
    for row in overall:
        newrow = []
        stripz = []
        for i,item in enumerate(row):
            if i < lastrow:
                newrow.append(row[i])
            elif i == lastrow:
                newword = findrep(row[lastrow], repdict, re)
                newrow.append(newword)

                stripped = strip_accents(item,unicodedata)
                stripz.append(stripped)

                newrow.append(''.join(stripz))

            else:
                newrow.append(row[i])
        newoverall.append(newrow)

    overall = newoverall

    # Original: sentencesonly
    sentencesonly = []
    for sentence in overall:
        sentencesonly.append(sentence[lastrow])

    # Duplicate: sentencesonly2
    sentencesonly2 = []
    for sentence in overallorig:
        sentencesonly2.append(sentence[lastrow])


    #### Token Extractor
    ##Tokenizing and converting to splittable format
    #Original: This is the original file that we want to put our content back to later
    toksenorig = []
    for sentence in sentencesonly2:
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            tokentag = []
            tokentag.append(token)
            tokentag.append('?')
            tokentag = tuple(tokentag)
            toksenorig.append(tokentag)
        toksenorig.append(('#','<SB'))

    #Duplicate/Working Token File
    toksen = []
    for sentence in sentencesonly:
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            #unaccented_token = strip_accents(token,unicodedata) #remove if you want accent
            tokentag = []
            tokentag.append(token)
            tokentag.append('?')
            tokentag = tuple(tokentag)
            toksen.append(tokentag)
        toksen.append(('#','<SB'))

    ### Splitting merged tokens using the Word Split Dictionary

    dictionarysplit = {}
    for k,v in dictsplit:
        dictionarysplit[k]=v

    toksencorr = []
    for w in toksen:
        if w[0] in dictionarysplit.keys():
            tokens = dictionarysplit[w[0]].split(' ')
            for t in tokens:
                giz = t,w[1]
                tuple(giz)
                toksencorr.append(giz)
        else:
            toksencorr.append(w)

    toksenorigcorr = []
    for w in toksenorig:
        if w[0] in dictionarysplit.keys():
            tokens = dictionarysplit[w[0]].split(' ')
            for t in tokens:
                giz = t,w[1]
                tuple(giz)
                toksenorigcorr.append(giz)
        else:
            toksenorigcorr.append(w)

    #### Stripping the diacritics
    corrtoksencorr = []
    for word in toksencorr:
            stripped = strip_accents(word[0],unicodedata)
            worded = []
            worded.append(stripped)
            worded.append(word[1])
            worded = tuple(worded)
            corrtoksencorr.append(worded)

    #### Use corrected tokens and sentencize them (the pre-tagging data)
    predata = list(tosentences(corrtoksencorr))

    #### Check length of sentences and words of the pre-tagging data
    print("\n")
    print("------------------------------")
    print("DATA FOR TAGGING - CORPUS SUMMARY")
    print("------------------------------")
    print("Sentences:", len(predata), "Words:", len(corrtoksencorr))
    print("------------------------------")
    print("\n")

    #### Extract features from the pre-tagging data
    predatafeatx = [sent2features(s,freqdict, operator, unicodedata) for s in predata]
    predatafeaty = [sent2gloss(s,freqdict, operator, unicodedata) for s in predata]

    #### Use if you want to use the development set in predicting as well (not recommended)
    #crf = sklearn_crfsuite.CRF(
    #    algorithm='lbfgs',
    #    c1=0.1,
    #    c2=0.1,
    #    max_iterations=100,
    #    all_possible_transitions=True
    #)

    #x_total = [sent2features(s,freqdict, operator, unicodedata) for s in sentences]
    #y_total = [sent2gloss(s,freqdict, operator, unicodedata) for s in sentences]

    #crf.fit(x_total, y_total)
    #labels = list(crf.classes_)

    print("Predicting POS based on trained model...")
    postdata = crf.predict(predatafeatx)
    print("Predictions generated.")

    #### Combine the output with unaccented token and autotagged gloss
    predsentcor = []
    for i, s in enumerate(predatafeatx):
        sentence = []
        for j, w in enumerate(s):
            sentence.append((w['word.lower()'],postdata[i][j]))
        predsentcor.append(sentence)

    #### Make unaccented token accented.
    list1 = list(tosentences(toksenorigcorr))
    list2 = predsentcor
    flist1 = towords(list1)
    flist2 = towords(list2)
    listed1 = [list(elem) for elem in flist1]
    listed2 = [list(elem) for elem in flist2]
    word = []
    for sentence in listed1:
        word.append(sentence[0])
    gloss = []
    for sentence in listed2:
        gloss.append(sentence[1])
    final = list(zip(word,gloss))
    accentpredsentcor = list(tosentences(final))


    ### Revise the New Data (Unseen Data)
    #Put the output of `predsentcor` into `overall`.
    #### Turn into list of strings first

    #With Accent
    taggedsent_withaccent = []
    for sent in accentpredsentcor:
        collector = []
        for tw in sent:
            #collector.append(tw[0]+'/'+tw[1])
            #collector.append('<'+tw[1]+'>'+ tw[0]+'<'+'/'+tw[1]+'>')
            collector.append(tw[0]+'_'+tw[1])
        collector.append('.')
        joined = ' '.join(collector)
        taggedsent_withaccent.append(joined)


    #Without Accent
    taggedsent = [] #for
    for sent in predsentcor:
        collector = []
        for tw in sent:
            #collector.append(tw[0]+'/'+tw[1])
            #collector.append('<'+tw[1]+'>'+ tw[0]+'<'+'/'+tw[1]+'>')
            collector.append(tw[0]+'_'+tw[1])
        collector.append('.')
        joined = ' '.join(collector)
        taggedsent.append(joined)

    #With Accent + Language
    if sourcelang == 'yes':
        taggedsent_withaccent_lg = [] #for
        for sent in accentpredsentcor:
            collector = []
            for tw in sent:
                hokkienlist = ['la','di','in','bo','ho','ya','o','ko','ka',
                               'an','e','ti','lo','tui','i','u','sang', 'si',
                              'lan', 'ma','kai',
                              'tsia','a', 'nan', 'lai','ai', 'it', 'kana','lak','tio','gun',
                              'La','Di','In','Bo','Ho','Ya','O','Ko','Ka',
                               'An','E','Ti','Lo','Tui','I','U','Sang', 'Si',
                              'Lan', 'Ma','Kai',
                              'Tsia','A', 'Nan', 'Lai','Ai', 'It', 'Kana','Lak','Tio','Hun']
                fillerlist = ['uhm','uhhm','umm','mm','mmm','hm','hmm','hmmm', 'uh','Uhm','Uhhm','Umm','Mm','Mmm','Hm','Hmm','Hmmm', 'Uh']
                punclist = list(stringg.punctuation)
                punclist.append('--')
                punclist.append('...')

                if tw[0] in punclist:
                    collector.append(tw[0]+'_P_'+tw[1])
                elif tw[0] in fillerlist:
                    collector.append(tw[0]+'_X_'+tw[1])
                elif tw[0] in hokkienlist:
                    collector.append(tw[0]+'_H_'+tw[1])
                elif tw[0] in tag_dic:
                    collector.append(tw[0]+'_T_'+tw[1])
                elif tw[0] in eng_dic:
                    collector.append(tw[0]+'_E_'+tw[1])
                else:
                    collector.append(tw[0]+'_H_'+tw[1])
            collector.append('.')
            joined = ' '.join(collector)
            taggedsent_withaccent_lg.append(joined)


        #Without Accent + Language
        taggedsent_lg = [] #for
        for sent in predsentcor:
            collector = []
            for tw in sent:

                hokkienlist = ['la','di','in','bo','ho','ya','o','ko','ka',
                               'an','e','ti','lo','tui','i','u','sang', 'si',
                              'lan', 'ma','kai',
                              'tsia','a', 'nan', 'lai','ai', 'it', 'kana','lak','tio','gun',
                              'La','Di','In','Bo','Ho','Ya','O','Ko','Ka',
                               'An','E','Ti','Lo','Tui','I','U','Sang', 'Si',
                              'Lan', 'Ma','Kai',
                              'Tsia','A', 'Nan', 'Lai','Ai', 'It', 'Kana','Lak','Tio','Hun']
                fillerlist = ['uhm','uhhm','umm','mm','mmm','hm','hmm','hmmm', 'uh','Uhm','Uhhm','Umm','Mm','Mmm','Hm','Hmm','Hmmm', 'Uh']
                punclist = list(stringg.punctuation)
                punclist.append('--')
                punclist.append('...')

                if tw[0] in punclist:
                    collector.append(tw[0]+'_P_'+tw[1])
                elif tw[0] in fillerlist:
                    collector.append(tw[0]+'_X_'+tw[1])
                elif tw[0] in hokkienlist:
                    collector.append(tw[0]+'_H_'+tw[1])
                elif tw[0] in tag_dic:
                    collector.append(tw[0]+'_T_'+tw[1])
                elif tw[0] in eng_dic:
                    collector.append(tw[0]+'_E_'+tw[1])
                else:
                    collector.append(tw[0]+'_H_'+tw[1])
            collector.append('.')
            joined = ' '.join(collector)
            taggedsent_lg.append(joined)


    ##Replace
    for k,v in enumerate(overall):
        v.append(taggedsent_withaccent[k])
        v.append(taggedsent[k])
        if sourcelang == 'yes':
            v.append(taggedsent_withaccent_lg[k])
            v.append(taggedsent_lg[k])

    #Export
    #Convert NSF to CSV format. The output file will be in the directory.
    overall[0][lastrow+1] = "utterance_nodia"
    overall[0][lastrow+2] = "dia_pos"
    overall[0][lastrow+3] = "nodia_pos"
    if sourcelang == 'yes':
        overall[0][lastrow+4] = "dia_sl_pos"
        overall[0][lastrow+5] = "nodia_sl_pos"


    if "Full tagged corpus - sentence as row" in filetype:
        df = pd.DataFrame(overall)
        df.to_csv('=output_full.csv', index=False, header=False, encoding='utf-8-sig')
        print("'Full tagged corpus - sentence as row' file generated.")

        #with open('05202021_output.txt','w',encoding='utf-8-sig') as out:
    #        for a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x in overall:
    #            out.write(a+'\t'+b+'\t'+c+'\t'+d +'\t'+ e +'\t'+ f +'\t'+ g +'\t'+ h +'\t'+ i +'\t'+ j+'\t'+ k+'\t'+l+'\t'+m+'\t'+n+'\t'+o+'\t'+ p +
    #            '\t'+ q +'\t'+ r +'\t'+ s +'\t'+ t +'\t'+ u +'\t'+ v +'\t' + w +'\t'+ x +'\n')
    #    print("Master file generated.")
    ##WORD column

    if "Plain tagged corpus - word as row with diacritics" in filetype:
        df = pd.DataFrame(overall)
        new_header = df.iloc[0] #grab the first row for the header
        df = df[1:] #take the data less the header row
        df.columns = new_header #set the header row as the df header

        if sourcelang == "yes":
            subsetdf = df[["tag", "interlocutor.id", "dia_sl_pos"]]

            subset = subsetdf.values.tolist()
            simp = []
            for index,row in enumerate(subset):
                simp.append((index,row[0],row[1],row[2]))

            wordsimp = []
            for index,tag,iden,utterance in simp:
                for word in utterance.split():
                    if len(list(word.split('_'))) < 2:
                        language = '?'
                        spword = '?'
                        POSword = '?'
                        tup = [tag,iden,index+1,language, spword,POSword]
                        wordsimp.append(tup)
                    else:
                        language = list(word.split('_'))[1]
                        spword = list(word.split('_'))[0]
                        POSword = list(word.split('_'))[2]
                        tup = [tag,iden,index+1,language,spword,POSword]
                        wordsimp.append(tup)
            wordsimp = wordsimp[1:]

            wordsimp[0][0] = 'tag'
            wordsimp[0][1] = 'id'
            wordsimp[0][2] = 'sentencenumber'
            wordsimp[0][3] = 'language'
            wordsimp[0][4] = 'word'
            wordsimp[0][5] = 'POS'

            ## Export
            df = pd.DataFrame(wordsimp)
            df.to_csv('=output_plaintagged_word_diac.csv', index=False, header=False, encoding='utf-8-sig')
            print("'Plain tagged corpus - word as row with diacritics (with source language)' file generated.")

        if sourcelang == "no":
            subsetdf = df[["tag", "interlocutor.id", "dia_pos"]]

            subset = subsetdf.values.tolist()
            simp = []
            for index,row in enumerate(subset):
                simp.append((index,row[0],row[1],row[2]))

            wordsimp = []
            for index,tag,iden,utterance in simp:
                for word in utterance.split():
                    if len(list(word.split('_'))) < 2:
                        spword = '?'
                        POSword = '?'
                        tup = [tag,iden,index+1, spword,POSword]
                        wordsimp.append(tup)
                    else:
                        spword = list(word.split('_'))[0]
                        POSword = list(word.split('_'))[1]
                        tup = [tag,iden,index+1,spword,POSword]
                        wordsimp.append(tup)
            wordsimp = wordsimp[1:]

            wordsimp[0][0] = 'tag'
            wordsimp[0][1] = 'id'
            wordsimp[0][2] = 'sentencenumber'
            wordsimp[0][3] = 'word'
            wordsimp[0][4] = 'POS'

            ## Export
            df = pd.DataFrame(wordsimp)
            df.to_csv('=output_plaintagged_word_diac_nosl.csv', index=False, header=False, encoding='utf-8-sig')
            print("'Plain tagged corpus - word as row with diacritics (no source language)' file generated.")


    ### Make into paragraph string
    #collector = []
    #for row in overall:
    #    collector.append(row[0])
    #    collector.append(row[18])
    #    collector.append('#')
    #final = " ".join(collector)

    ### No Diacritic Tagged Text File
    if "Plain tagged corpus - sentence as row without diacritics" in filetype:
        df = pd.DataFrame(overall)
        new_header = df.iloc[0] #grab the first row for the header
        df = df[1:] #take the data less the header row
        df.columns = new_header #set the header row as the df header

        subsetdf = df[["tag", "nodia_pos"]]
        with open('=output_txt_nodiac_tagged.txt','w',encoding='utf-8-sig') as out:
            for index, row in subsetdf.iterrows():
                out.write(row[0] + '\n'+ row[1] +'\n' +'\n')
        print("'No Diacritic Tagged Text File' generated.")

    ### Diacritic Tagged Text File
    if "Plain tagged corpus - sentence as row with diacritics" in filetype:
        df = pd.DataFrame(overall)
        new_header = df.iloc[0] #grab the first row for the header
        df = df[1:] #take the data less the header row
        df.columns = new_header #set the header row as the df header

        subsetdf = df[["tag", "dia_pos"]]
        with open('=output_txt_diac_tagged.txt','w',encoding='utf-8-sig') as out:
            for index, row in subsetdf.iterrows():
                out.write(row[0] + '\n'+ row[1] +'\n' +'\n')
        print("'Diacritic Tagged Text File' generated.")
