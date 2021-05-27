import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow_graphics.math import math_helpers
df = pd.read_csv('test_control_vert_pos.csv', header=None)
test_pos = df.values.astype('float32')

all_pos_spherical = math_helpers.cartesian_to_spherical_coordinates(test_pos).numpy()
selected_pos_spherical = math_helpers.cartesian_to_spherical_coordinates(test_pos[20:23, :]).numpy()

print(all_pos_spherical[20:23, :])
print(selected_pos_spherical)
