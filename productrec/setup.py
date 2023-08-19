from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import pickle
import os
from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.evaluation import auc_score
import numpy as np
from scipy import sparse
from lightfm import LightFM
from sklearn.base import clone
from utils import LightFMResizable

# file = open('ml/data.pkl', 'rb')
base_dir = os.path.dirname(os.path.dirname(__file__))
# laod dataframe from data.pkl
df = pd.read_pickle(os.path.join(base_dir,'productrec/ml/data.pkl'))

# load tfidf vectorizer from tfidf.pkl
tfd = pickle.load(open(os.path.join(base_dir,'productrec/ml/tfdf.pkl'), "rb"))

# load kmeans model from kmeans.pkl
kmeans = pickle.load(open(os.path.join(base_dir,'productrec/ml/kmeans.pkl'), "rb"))

# item recommmendation

order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = tfd.get_feature_names_out()
def get_cluster(id):
    res =[]
    for ind in order_centroids[id, :10]:
        res.append(terms[ind])
    return res
def search(term):
    x = tfd.transform(term.split())
    # print(x,term.split())
    y = kmeans.predict(x)
    return y[0]

def similar_terms(term):
    cluster = search(term)
    return get_cluster(cluster)
def get_recommendation(term):
    cluster = search(term)
    return df[df['cluster'] == cluster].index.tolist()[:10]

########################### User recommendation ##############

# generate data 
def gen_user_feature(user:object):
    feat = []
    for x in user:
        if x=='id': continue
        feat.append(x + ':' + str(user[x]))
    return (user['id'],feat)


### load data for user recommendation
dataset = pickle.load(open(os.path.join(base_dir,'productrec/ml/dataset.pkl'), "rb"))
model = pickle.load(open(os.path.join(base_dir,'productrec/ml/model.pkl'), "rb"))
user_feature_map = pickle.load(open(os.path.join(base_dir,'productrec/ml/user_feature_map.pkl'), "rb"))
id_item_map = pickle.load(open(os.path.join(base_dir,'productrec/ml/id_item_map.pkl'), "rb"))
ifs = pickle.load(open(os.path.join(base_dir,'productrec/ml/ifs.pkl'), "rb"))
nfs = pickle.load(open(os.path.join(base_dir,'productrec/ml/nfs.pkl'), "rb"))

n_items = df.shape[0]
# Save all loaded data in a file
def save_all():
    pickle.dump(dataset, open(os.path.join(base_dir,'productrec/ml/dataset.pkl'), "wb"))
    pickle.dump(model, open(os.path.join(base_dir,'productrec/ml/model.pkl'), "wb"))
    pickle.dump(nfs, open(os.path.join(base_dir,'productrec/ml/nfs.pkl'), "wb"))
    
def isNewUser(userId):
    return userId not in dataset.mapping()[0]
# helper function to add and update user model
# add interaction data for old users
def update_user(user, item_id, rating):
    global nfs,ufs
    user_id = user['id']
    (new_interactions,new_wts) = dataset.build_interactions([(user_id, item_id, rating)])
    ufs = dataset.build_user_features(nfs)
    model.fit_partial(new_interactions,user_features=ufs,item_features=ifs, sample_weight=new_wts)
# we cannot add new user without retraining the whole model
# add interaction data for new users
def add_user(user:object):
    global nfs,dataset
    dataset.fit_partial(users=[user['id']])
    nfs += ([ gen_user_feature(user) ]) 
def get_item(id):
    return df.loc[id]
# get recommendation for existing user
def recommend(userId):
    y = model.predict(userId,np.arange(n_items),item_features=ifs)
    return [id_item_map[x] for x in np.argsort(-y)][:10]
    # return np.argsort(-y)[:4]

if __name__ == "__main__":
    # print(model)
    # print(n_items)
    new_user = {
        'id':3,
        'age': 19,
        'sex':'M'
    }
    # print(new_user_recommend(new_user))
    add_user(new_user);
    # print(nfs,dataset.mapping()[0])
    
    # update_user(new_user,3168,2);
    print(recommend(3))
    # print(get_recommendation("shoe"))