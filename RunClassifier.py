#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.io import imsave
from numpy.linalg import norm
from sklearn.metrics import classification_report
import os
from pathlib import Path
import shutil
seed = 0


# In[2]:


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[3]:


mapping = {0 : 'T-shirt/top',
           1 : 'Trouser',         
           2 : 'Pullover',
           3 : 'Dress',
           4 : 'Coat',           
           5 : 'Sandal',           
           6 : 'Shirt',
           7 : 'Sneaker',           
           8 : 'Bag',
           9 : 'Ankle boot'}


# In[4]:


np.random.seed(seed)
indices = np.arange(0, train_images.shape[0])
np.random.shuffle(indices)


# In[5]:


train_images_unshuffled = train_images.copy()
train_labels_unshuffled = train_labels.copy()
train_imgs = train_images_unshuffled[indices]
train_lbls = train_labels_unshuffled[indices]


# In[6]:


train_imgs = np.array(train_imgs, dtype = np.double)/255.0


# In[7]:


np.random.seed(seed)
indices = np.arange(0, test_images.shape[0])
np.random.shuffle(indices)


# In[8]:


test_images_unshuffled = test_images.copy()
test_labels_unshuffled = test_labels.copy()
test_imgs = test_images_unshuffled[indices]
test_lbls = test_labels_unshuffled[indices]


# In[9]:


test_imgs = np.array(test_imgs, dtype = np.double)/255.0


# In[10]:


slice_size = 7
train_histogram_imgs = []
for ind in range(train_imgs.shape[0]):    
    grid_concat_rep = [] # concatenation of grid vectors
    for i in np.arange(0, 28, slice_size):
        for j in np.arange(0, 28, slice_size):
            grid_concat_rep.append(train_imgs[ind, i:i+slice_size, j:j+slice_size].ravel())
    grid_concat_rep = np.array(grid_concat_rep)
    train_histogram_imgs.append(grid_concat_rep)
train_histogram_imgs = np.array(train_histogram_imgs)


# In[11]:


slice_size = 7
test_histogram_imgs = []
for ind in range(test_imgs.shape[0]):    
    grid_concat_rep = [] # concatenation of grid vectors
    for i in np.arange(0, 28, slice_size):
        for j in np.arange(0, 28, slice_size):
            grid_concat_rep.append(test_imgs[ind, i:i+slice_size, j:j+slice_size].ravel())
    grid_concat_rep = np.array(grid_concat_rep)
    test_histogram_imgs.append(grid_concat_rep)
test_histogram_imgs = np.array(test_histogram_imgs)


# In[12]:


train_histogram_imgs.shape, test_histogram_imgs.shape


# In[13]:


train_clustering_data = np.array([train_histogram_imgs[i, j, :] for i in range(60000) for j in range(16)])


# In[14]:


train_clustering_data.shape


# In[15]:


class myKmeans():
  
  def __init__(self, n_clusters = 10, random_state = 0, n_iterations = 10, n_init = 10):
    self.n_clusters = n_clusters
    self.random_state = random_state
    self.n_iterations = n_iterations
    self.cluster_centers = None           
    
  def update_membership(self):          
    for i, center in enumerate(self.cluster_centers):      
      self.distances[i, :] = np.sqrt(np.sum(np.square(self.X_train - center), axis = 1))
    self.cluster_membership = np.argmin(self.distances, axis = 0)    
    
  def update_centers(self):
    for i in range(self.n_clusters):
      self.cluster_centers[i] = np.mean(self.X_train[self.cluster_membership == i], axis = 0)      

  def fit(self, X_train):
    
    self.X_train = X_train         
    self.dimensions = self.X_train.shape
    self.n_entries = self.dimensions[0]
    self.cluster_centers = [] 
    self.init_cluster_size = (self.X_train.shape[0])//self.n_clusters
    self.cluster_membership = np.zeros(self.n_entries, dtype = int) - 1
    self.distances = np.zeros((self.n_clusters, self.X_train.shape[0]))
    
    indices = np.arange(0, self.dimensions[0])    
    np.random.seed(self.random_state)
    np.random.shuffle(indices)    
    
    start_ind = 0        
    next_ind = min(start_ind + self.init_cluster_size, self.n_entries)
    
    for cluster in range(self.n_clusters):                
      self.cluster_membership[indices[start_ind : next_ind]] = cluster
      self.cluster_centers.append(np.mean(self.X_train[indices[start_ind : next_ind]], axis = 0))
      start_ind = next_ind
      next_ind = min(start_ind + self.init_cluster_size, self.n_entries)
          
    self.cluster_centers = np.array(self.cluster_centers)    
    self.dists = []
    prev = self.cluster_membership.copy()
    ctr = 0
    for iteration in range(self.n_iterations):
      self.update_membership()
      self.update_centers()
      if np.all(prev == self.cluster_membership):
        break      
      self.dists.append(np.sum(self.distances))
      print(f"Done with {iteration + 1}th iteration")

  def soft_predict(self, X_test):
    test_distances = np.zeros((self.n_clusters, X_test.shape[0]), dtype = np.double)
    for i, center in enumerate(self.cluster_centers):    
      test_distances[i, :] = np.sqrt(np.sum(np.square(X_test - center), axis = 1))
    test_cluster_membership = np.argmin(test_distances, axis = 0)

    test_distances = -2.5 * (test_distances - np.min(test_distances, axis = 0))
    test_distances = np.exp(test_distances)
    test_distances = test_distances/np.sum(test_distances, axis = 0)

    histogram = np.sum(test_distances, axis = 1)
    histogram = histogram/np.sum(histogram)
    return histogram  


# In[16]:


def CreateVisualDictionary(n_clusters = 25, random_state = 0, n_iterations = 10):
    """
    Computes and save the visual dictionary in a folder named VisualDictionary.
    """   
    kmeans = myKmeans(n_clusters=n_clusters, random_state=random_state, n_iterations=n_iterations)
    kmeans.fit(train_clustering_data)    
    dirpath = Path('./') / 'VisualDictionary'
    if dirpath.exists() and dirpath.is_dir():
      shutil.rmtree(dirpath)
    os.mkdir('VisualDictionary')
    os.chdir('VisualDictionary')
    for i in range(kmeans.n_clusters):
      plt.imsave(f'cluster{i}.jpg', np.reshape(kmeans.cluster_centers[i], (7, 7)), cmap = 'gray')
    os.chdir('..')
    return kmeans


# In[17]:


kmeans = CreateVisualDictionary()


# In[18]:


def ComputeHistogram(feat_vector, visual_dictionary, alpha = 2.5):  
  feat_vector = np.array(feat_vector)
  if feat_vector.ndim == 1:
    feat_vector = np.reshape(feat_vector, (1, -1))
  num_clusters = visual_dictionary.shape[0]
  test_distances = np.zeros((num_clusters, feat_vector.shape[0]), dtype = np.double)
  for i, center in enumerate(visual_dictionary):    
    test_distances[i, :] = np.sqrt(np.sum(np.square(feat_vector - center), axis = 1))
  test_cluster_membership = np.argmin(test_distances, axis = 0)

  test_distances = -1 * alpha * (test_distances - np.min(test_distances, axis = 0))
  test_distances = np.exp(test_distances)
  test_distances = test_distances/np.sum(test_distances, axis = 0)

  histogram = np.sum(test_distances, axis = 1)
  histogram = histogram/np.sum(histogram)
  return histogram    


# In[19]:


def MatchHistogram(train, test, metric = 'manhattan'):
  dist = 0
  if metric == 'manhattan':
    dist = np.sum(np.abs(test - train))
  else:
    try:
      dist = np.linalg.norm(train - test, ord = int(metric))
    except:
      dist = np.linalg.norm(train - test, ord = 2)
  return dist


# In[22]:


train_histograms = []
for ind in range(60000):
  train_histograms.append(ComputeHistogram(train_histogram_imgs[ind, :, :], kmeans.cluster_centers, alpha = 2.0))
train_histograms = np.array(train_histograms)


# In[23]:


test_histograms = []
for ind in range(10000):
  test_histograms.append(ComputeHistogram(test_histogram_imgs[ind, :, :], kmeans.cluster_centers, alpha = 2.0))
test_histograms = np.array(test_histograms)


# In[24]:


test_pred_labels = []

for j in np.arange(10000):
  if((j+1)%100 == 0):
    print(f'done for {j+1} test images')
  mini = 1e9
  temp_label = -1      
  for i in range(60000):         
    dist = MatchHistogram(train_histograms[i], test_histograms[j])
    if(dist < mini):
      mini = dist
      temp_label = train_lbls[i]    
  test_pred_labels.append(temp_label)


# In[26]:


print(classification_report(test_lbls[0:10000], test_pred_labels))


# In[27]:


print(mapping)


# In[ ]:




