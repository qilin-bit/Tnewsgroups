# -*- coding: utf-8 -*-
__author__ = 'qilin'

print '\nthis is Tnewsgroups program\n'

print '\n################以下模块加载语料#################################'
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

print 'twenty_train.keys:', twenty_train.keys()
# twenty_train.keys: ['description', 'DESCR', 'filenames', 'target_names', 'data', 'target']

print '\ntwenty_train.data:', type(twenty_train.data), len(twenty_train.data)
print 'twenty_train.target:', type(twenty_train.target), len(twenty_train.target)
print 'twenty_train.filenames:', type(twenty_train.filenames), len(twenty_train.filenames)
print 'twenty_train.target_names:', type(twenty_train.target_names), len(twenty_train.target_names)
'''
twenty_train.data: <type 'list'> 18846
twenty_train.target: <type 'numpy.ndarray'> 18846
twenty_train.filenames: <type 'numpy.ndarray'> 18846
twenty_train.target_names: <type 'list'> 20
'''
print
print twenty_train.target[:10]
print twenty_train.filenames[:10]

print '\ntwenty_train.target_names:'
print twenty_train.target_names

print '\nthe content of the document no.0:'
print twenty_train.data[0]

for i in range(len(twenty_train.target_names)):
    print i, twenty_train.target_names[i]

print '\n#################以下模块对语料进行预处理###########################'
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words='english')
X_train_counts = count_vect.fit_transform(twenty_train.data)
print '\nX_train_counts:', type(X_train_counts)
print X_train_counts.shape

myvoclist = count_vect.get_feature_names()
myvocdict = count_vect.vocabulary_
mystopwords = count_vect.get_stop_words()

print '\nmyvoclist:', type(myvoclist)
print len(myvoclist)
print 'myvocdict:', type(myvocdict)
print len(myvocdict)

print type(count_vect.get_stop_words())
print len(count_vect.get_stop_words())
print count_vect.get_stop_words()

print X_train_counts[0]

'''
myvoclist是所有单词组成对列表，按照字典顺序排列；
myvocdict是单词与其在字典中对应编号构成对字典，以单词为键，以编号为值；
'''