import numpy as np

a = np.array([1, 2, 3])
b = np.array([3, 1, 3])

print(np.squeeze(np.max([a, b], axis=0)))
data = np.arange(20000)
np.save("/Users/qiang.hu/PycharmProjects/al_leak/new_experiments/data/QC_gru_remain_data.npy", data)
print(data.shape)
