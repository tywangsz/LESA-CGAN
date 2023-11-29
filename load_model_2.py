from tensorflow.keras.models import load_model
from build_plot_1 import build_and_train_models


steps = 200
gen0, gen1,enc0,enc1 = build_and_train_models(train_steps = steps)

