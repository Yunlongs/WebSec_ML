from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer

def load_alexa(filename):
    x = []
    with open(filename,'r') as f:
        for line in f.readlines():
            x.append(line.strip().split(',')[1])
    return x
def load_dga(file_crypto,file_post):
    x1 = []
    x2 = []
    with open(file_crypto,'r') as f:
        for line in f.readlines():
            x1.append(line.strip().split(',')[0])
    with open(file_post,'r') as f:
        for line in f.readlines():
            x2.append(line.strip().split(',')[0])
    return x1,x2

def get_feature(file_alexa,file_crypto,file_post):
    x1 = load_alexa(file_alexa)
    x2,x3 = load_dga(file_crypto,file_post)
    y1 = [0]*len(x1)
    y2 = [1]*len(x2)
    y3 = [2]*len(x3)
    x = x1+x2+x3
    y = y1+y2+y3

    cv = CountVectorizer(ngram_range=(2,2),decode_error='ignore',token_pattern=r'\w',min_df=1)
    X = cv.fit_transform(x).toarray()

    clf = GaussianNB()

    score = cross_val_score(clf,X,y,cv=3)
    print("score=",score)
get_feature("../data/dga/test-top-1000.csv","../data/dga/dga-cryptolocke-1000.txt","../data/dga/dga-post-tovar-goz-1000.txt")
