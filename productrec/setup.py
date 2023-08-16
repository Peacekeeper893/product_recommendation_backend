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
    return df[df['cluster'] == cluster].index.tolist()

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

# helper function to add and update user model
# add interaction data for old users
def update_user(user, item_id, rating):
    global nfs,ufs
    user_id = user['id']
    (new_interactions,new_wts) = dataset.build_interactions([(user_id, item_id, rating)])
    nfs +=  [ gen_user_feature(user ) ] 
    ufs = dataset.build_user_features(nfs)
    model.fit_partial(new_interactions,user_features=ufs,item_features=ifs, sample_weight=new_wts)
# we cannot add new user without retraining the whole model
# add interaction data for new users
def add_user(userId):
    dataset.fit_partial(users=[userId])
def get_item(id):
    return df.loc[id]
# get recommendation for existing user
def recommend(userId):
    y = model.predict(userId,np.arange(n_items))
    return [get_item(id_item_map[x]) for x in np.argsort(-y)][:4]
    # return np.argsort(-y)[:4]
# get recommendation for new user who has not interacted with any item
def format_newuser_input(user_feature_map, user_feature_list):
  normalised_val = 1.0 
  target_indices = []
  for feature in user_feature_list:
    try:
        target_indices.append(user_feature_map[feature])
    except KeyError:
        print("new user feature encountered '{}'".format(feature))
        pass
  new_user_features = np.zeros(len(user_feature_map.keys()))
  for i in target_indices:
    new_user_features[i] = normalised_val
  new_user_features = sparse.csr_matrix(new_user_features)
  return(new_user_features)
def new_user_recommend(user):
    new_user_features = gen_user_feature(user)[-1]
    new_ufs = format_newuser_input(user_feature_map, new_user_features)
    y = model.predict(0,np.arange(n_items),user_features=new_ufs)
    return [get_item(id_item_map[x]) for x in np.argsort(-y)] [:4]
    # return  np.argsort(y)[:4]

if __name__ == "__main__":
    # print(model)
    # print(n_items)
    new_user = {
        'id':3,
        'age': 19,
        'sex':'M'
    }
    # print(new_user_recommend(new_user))
    add_user(3)
    update_user(new_user,3168,2);
    print(recommend(3))
    # print(get_recommendation("shoe"))