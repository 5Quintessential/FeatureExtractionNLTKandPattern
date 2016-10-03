import PyPDF2
import nltk
import re
import csv
import copy
import codecs
import pattern
import string
import itertools
import traceback
import numpy as np
import os.path
import sys

from nltk.tokenize import word_tokenize
from nltk.text import TextCollection
from nltk.tokenize import sent_tokenize
from nltk.tag.stanford import StanfordNERTagger
from nltk.corpus import wordnet as wn
from nltk.chunk.util import *
from nltk.chunk.regexp import *
from pattern.text import parse
from pattern.text import pprint
from pattern.graph import Graph
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO

class featurex:
    tokenized_sentence_array=[]     # Global array of POS tagged sentences
    main_subject_of_spec=[]         # Global variable to store the list of top 10 main subject candidates
    complete_processed_corpus=''    # Global to store cleaned text corpus
    subject_object_dict={}          # Global dictionary storing the (subject,object) tuple for all sentences
    CandidateTerms = []             # Global array for candidate features list
    WeightedCandidateTerm = []      # Global array of tuples (term, term_weight)
    FeatureGroups = []              # Global List for feature groups
    filepath = ''                   # Corpus file path
    filerange = ''                  # Page range

    def convert_pdf_to_txt(self):
        if not os.path.exists('corpustext.txt'):
            rsrcmgr = PDFResourceManager()
            retstr = StringIO()
            codec = 'utf-8'
            laparams = LAParams()
            device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
            fp = file(self.filepath, 'rb')
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            password = ""
            maxpages = 0
            caching = True
            indexes = [int(n) for n in self.filerange.split('-')]
            pagenos=set(range(indexes[0],indexes[1]))

            for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
                interpreter.process_page(page)

            text = retstr.getvalue()
            with open('corpustext.txt','a') as tt:
                tt.write(text)
            fp.close()
            device.close()
            retstr.close()
            return text
        else:
            with open("corpustext.txt","r") as ct:
                return ct.read()


    def __init__(self, pathinfo,rangeinfo):
        self.filepath=pathinfo
        self.filerange=rangeinfo

    def pre_process(self):
        counter = 0
        try:
            if not os.path.exists('processedtext.txt'):          
                corpus = self.convert_pdf_to_txt()
                # Remove all newline characters '\n' and other kinds of spaces            
                corpus = re.sub(r'[^\x00-\x7F]+',' ', corpus)
                corpus = re.sub( '\s+', ' ', corpus).strip()
                
                words = word_tokenize(corpus)
                spaceandspecial = [' ','{','}','(',')','[',']','"','~',',',':',';','?','\'','`','-','_','#','%']
                stopwords = nltk.corpus.stopwords.words('english')
                content = [w for w in words if w.lower() not in stopwords]
                cleanedwords = [w for w in content if w.lower() not in spaceandspecial] 
                peoplenametagger = StanfordNERTagger('stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz', 'stanford-ner/stanford-ner.jar')
                human_names = peoplenametagger.tag(cleanedwords)
                people_names = [tag[0].lower() for tag in human_names if tag[1]=='PERSON']
                cleaned_words = [w for w in cleanedwords if w.lower() not in people_names] 

                self.complete_processed_corpus = ' '.join(cleaned_words)
                with open("processedtext.txt","w") as pt:
                    pt.write(self.complete_processed_corpus)
            else:
                with open("processedtext.txt","r") as pt:
                    self.complete_processed_corpus = pt.read()

            if not os.path.exists('mainsubject.txt') or not os.path.exists('tarray.csv'):
                for sentence in sent_tokenize(self.complete_processed_corpus):
                    try:
                        words = word_tokenize(sentence)
                        self.tokenized_sentence_array.append(("S"+str(counter),sentence,copy.deepcopy(words)))
                        counter = counter + 1                    
                    except Exception: 
                        traceback.print_exc()
                        pass
                                    
                self.main_subject_of_spec = nltk.FreqDist(word_tokenize(self.complete_processed_corpus)).most_common(20)   
                # Write to files
                with open("mainsubject.txt", "w") as ms:
                    ms.write(str(self.main_subject_of_spec).strip('[]'))  
                with open("tarray.csv", "w") as ta:
                    writer = csv.writer(ta, quoting=csv.QUOTE_ALL)
                    writer.writerows(self.tokenized_sentence_array) 
            else:
                with open('tarray.csv', 'r') as tarrcsv:
                    self.tokenized_sentence_array=[tuple(line) for line in csv.reader(tarrcsv)]
        except Exception: 
            traceback.print_exc()
            pass

    def GetLastNoun(self):
        if len(self.CandidateTerms)>0:
            return self.CandidateTerms.pop()
        else: return ''

    # Remove duplicates in a list
    def RemoveDuplicates(self, listObj):
        return list(set(listObj)) 
        
    def processSimilarity(self):
        # Similarity analysis
        # Find all synonyms of all words
        wordsGroup = []
        if not os.path.exists('featureGroups.txt'):
            marker = 0            
            matches = []
            group = []
            for w in self.CandidateTerms:
                if w != "" and not(type(w) is list):
                    for word in w.split():
                        for synset in wn.synsets(word):
                            for synonym in synset.lemmas():
                                matches = [term for term in self.CandidateTerms if not(type(term) is list) and (synonym.name() in term.split() and term not in matches)]
                                matches = self.RemoveDuplicates(matches)
                                if len(matches)>0:
                                    # Constuct words group
                                    group.extend(matches)                                
                        if len(group)>0:
                            wordsGroup.append(('FG'+str(marker),copy.deepcopy(group)))                

                            for val in group:        
                                with open("featureGroups.txt", "a") as fg:
                                    fg.write('FG'+str(marker) + ' -> '+val+'\n')

                            group=[] 
                            marker = marker+1
        else:
            added = False
            with open("featureGroups.txt", "r") as fg:
                grouptext = fg.read()
            lines = grouptext.split("\n")
            for line in lines:
                arr = line.split("->")                
                for (m,v) in wordsGroup:
                    if m==arr[0]:
                        v.append(arr[1])
                        added = True
                        break
                if not added:
                    if len(arr)>1:
                        wordsGroup.append((arr[0],[arr[1]]))
                    
                added=False

        textcombine =' '
        for i,s,t in self.tokenized_sentence_array:
            textcombine = (textcombine + ''.join(s))
        corpuscol = TextCollection([textcombine])
        for g in wordsGroup:
            for w in g:
                cnt = 0
                weight = 0.0
                for t in w:
                    weight = weight + corpuscol.tf(t,textcombine)
                    cnt = cnt+1
                self.WeightedCandidateTerm.append((w,weight/cnt))
                with open("weightedGroups.txt", "a") as wg:
                    wg.write(str(self.WeightedCandidateTerm).strip('[]'))
        return wordsGroup

    def extract_candidates(self):
        if not os.path.exists('SubjectObject.txt'):
            sentences = sent_tokenize(self.complete_processed_corpus)
            orig_stdout = sys.stdout
            f = file('SubjectObject.txt', 'a')
            sys.stdout = f
            for sent in sentences:
                datas = parse(sent, relations=True, lemmata=True)
                print pprint(datas)
            sys.stdout = orig_stdout
            f.close()   
                     
        My_Tupples = []
        
        with open('SubjectObject.txt','r') as f:
            classifiedText = f.readlines()
            for line in classifiedText:
                line = line.replace("^","")
                if line !='' and line.find("                    ") == -1:
                    tuple = (line.split())
                    My_Tupples.append(tuple)
            Subjects = list()
            Objects = list()
            counter = 1
            dict = {}
            tmp = list()
            subterm = ''
            objterm = ''    
            sentcounter = 1
            isContiguous = False
            My_Tupples.remove([])
            My_Tupples.remove(['None'])
            for mylst in My_Tupples:  
                if mylst != [] and mylst != ['None']:                          
                    if mylst[0]=="WORD": # this means it is a part of one sentence until new 'WORD' term is hit
                        if len(Objects)>0:
                            if len(Subjects)>0:
                                dict["S"+str(sentcounter)] = (copy.deepcopy(Subjects),copy.deepcopy(Objects))
                                sentcounter = sentcounter+1
                                Subjects=[]
                                Objects=[]
                            else:
                                dict["S"+str(sentcounter)] = (list(),copy.deepcopy(Objects))
                                sentcounter = sentcounter+1
                                Subjects=[]
                                Objects=[]
                        elif len(Subjects)>0:
                            dict["S"+str(sentcounter)] = (copy.deepcopy(Subjects),list())
                            sentcounter = sentcounter+1
                            Subjects=[]
                            Objects=[]            
                        continue
                    elif mylst[3] == "SBJ":
                        if(isContiguous):
                            subterm += mylst[0]+' '
                        else:
                            subterm = mylst[0]+' '
                            isContiguous = True
                    elif mylst[3] == "OBJ":
                        if(isContiguous):
                            objterm += mylst[0]+' '
                        else:
                            objterm = mylst[0]+' '
                            isContiguous = True
                    else:
                        isContiguous = False
                        if len(objterm)>0:
                            Objects.append(objterm.rstrip())
                        if len(subterm)>0:
                            Subjects.append(subterm.rstrip())
                        subterm = ''
                        objterm = ''
            self.subject_object_dict = copy.deepcopy(dict)      
            grammar = ChunkRule("<JJ>?<NN>*<NNS>?<NNP>*<VBN>*<VB>*", "Feature Term")
            cp = RegexpChunkParser([grammar], chunk_label='FeatureTerm') 
            sentences = sent_tokenize(self.complete_processed_corpus)
            for sent in sentences:
                fwords = nltk.word_tokenize(sent)
                sentence = nltk.pos_tag(fwords)
                parsetree = cp.parse(sentence)
                featureterms = list(parsetree.subtrees(filter=lambda x: x.label()=='FeatureTerm'))                
                
                for term in featureterms:
                    featureterm = ''
                    for leaf in term.leaves():
                        featureterm = featureterm + str(leaf[0]) + ' '
                    featureterm = featureterm.strip()
                    self.CandidateTerms.append(featureterm)
            for i in range(len(dict)):
                for t in dict["S"+str(i+1)]:
                    if len(t)>0:
                        if (nltk.pos_tag(sw) in ('DT','PRP','PRP$','PDT') for sw in t[0]):
                            # find the previous noun and replace
                            lastnoun = self.GetLastNoun()
                            for ww in t[0]:
                                if nltk.pos_tag(ww) in ('DT','PRP','PRP$','PDT'):
                                    t[0]=t[0]+lastnoun+' '
                                else: t[0]=t[0]+ww+' '
                            self.CandidateTerms.append(t)
                        else:
                            self.CandidateTerms.append(w for w in t[0])
                            self.CandidateTerms.append(w for w in t[1])  
            with open("CandidateTerms.txt","w") as ct:
                ct.write(str(self.CandidateTerms).strip('[]'))                  
        self.FeatureGroups = self.processSimilarity()
        # set the dictionary with subject,object tuples for relationship visualization

    
    def visualize_rel(self):
        orderedPairs =[]
        for i in range(len(self.subject_object_dict)):
            orderedPair = list(itertools.product(self.subject_object_dict["S"+str(i+1)][0], self.subject_object_dict["S"+str(i+1)][1]))
            orderedPairs.append(orderedPair)
        g = Graph()
        for node in (orderedPairs):
            for n1,n2 in node:
                g.add_node(n1)
                g.add_node(n2)
                g.add_edge(n1, n2, weight=0.0, type='is-related-to')
        g.export('FeatureRelations', directed=True)
        orig_stdout = sys.stdout
        gn = file('GraphNodeWeights.txt', 'a')
        sys.stdout = gn
        for n in sorted(g.nodes, key=lambda n: n.weight):
             print '%.2f' % n.weight, n
        sys.stdout = orig_stdout
        gn.close() 

