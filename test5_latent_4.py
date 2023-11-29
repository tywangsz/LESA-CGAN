

from build_plot_1 import build_and_train_models
from load_model_2 import load_model
from creat_latent_3 import create_latent
from load_preprocess_data import google_data_loading
from PCA_TSNE import PCA_Analysis,tSNE_Analysis
# from numpy import display
import pandas as pd
import matplotlib as plt
import numpy as np
import statsmodels.api as sm
nlags = 13
seq_length = 1
dataX, X_train, scaler = google_data_loading(seq_length)
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'



d_recipe = {}
d_recipe["class0"] = [0, True, "Normal"]
d_recipe["class2"] = [2, True, "Normal"]
d_recipe["noise"] = [None, False, "Normal"]
d_recipe["z0_4"] = [None, False, "z0_4"]
d_recipe["z1_4"] = [None, False, "z1_4"]
d_recipe["org"] = d_recipe["noise"]

steps = 1
gen0, gen1, enc0, enc1 = build_and_train_models(train_steps = steps)

def plot_corr(df):
  f = plt.figure(figsize=(80, 80))
  plt.matshow(df, fignum=f.number)
  plt.xticks(range(df.shape[1]), df.columns, fontsize=5, rotation=45)
  plt.yticks(range(df.shape[1]), df.columns, fontsize=5)
  cb = plt.colorbar()
  cb.ax.tick_params(labelsize=5)
  plt.title('Correlation Matrix', fontsize=16);

def inverse_trans(train_data):
  num_instances, num_features = train_data.shape
  train_data = np.reshape(train_data, (-1, num_features))
  train_data = scaler.inverse_transform(train_data)
  train_data = np.reshape(train_data, (num_instances, num_features))
  return train_data

viz = ["autocorr", "featcorr","timecorr", "tsna", "pca", "mean_vol","mean_price", "var_vol", "var_price", "gaf", "gafts", "gram4d", "pairrec", "recplot", "mtf", "mtf3d"]
viz = ["preds"]
preds = {}
for typed in viz:
  for en, key in enumerate(d_recipe.keys()):
    new, noise_class, z0, z1 = create_latent(dataX, class_label = d_recipe[key][0], classed=d_recipe[key][1], noise=d_recipe[key][2])

    real_feature0 = enc0.predict(new)
    feature0 = gen0.predict([noise_class, z0, real_feature0])


    real_feature1 = enc1.predict(real_feature0)
    gen_data = gen1.predict([real_feature1, z1, feature0])
    temp = gen_data.shape[1]
    gen_data1  = np.reshape(gen_data, (-1, gen_data.shape[1]))


    gen_data_t = inverse_trans(gen_data)
    gen_data_t_write = np.reshape(gen_data_t,(-1,gen_data_t.shape[1]))
    gen_data_t_write = pd.DataFrame(gen_data_t_write)
    gen_data_t_write.to_csv('1_t_Trans.csv')




