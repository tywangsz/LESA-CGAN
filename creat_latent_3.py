import numpy as np

def create_latent(dataX, batch_n = 1000, equal_batch=True, class_dim = 2, class_label = 2, z_dim = 128,z_val = 0, classed=True, noise="Normal" ):
  new1 = []
  if classed==True:
    new = dataX[dataX[:,:,80] == class_label]
    new = np.reshape(new, (new.shape[0], 1, 81))
    new1 = new[:,:,:-1]
    batch_num = new.shape[0]
    if not equal_batch:
      batch_num = batch_n
    print(batch_n)
    noise_class = np.ones((batch_num, class_dim))
    noise_class[:, class_label] = 0

  else:
    noise_class = np.eye(class_dim)[np.random.choice(class_dim, batch_n)]
    new = dataX
    batch_num = batch_n
  if not equal_batch:
    batch_num = batch_n
  if noise=="Normal":

    z0 = np.random.normal(scale=0.5, size=[batch_num, 128])
    z1 = np.random.normal(scale=0.5, size=[batch_num, z_dim])
  elif noise=="z0_4":

    z0 = np.full((batch_num, 128), z_val)
    z1 = np.random.normal(scale=0.5, size=[batch_num, z_dim])
  elif noise=="z1_4":

    z0 = np.random.normal(scale=0.5, size=[batch_num,128 ])
    z1 = np.full((batch_num, z_dim), z_val)

  return new1, noise_class, z0, z1