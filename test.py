import numpy as np
from scipy import stats

# a = np.array([[[1, 2, 3, 4], [5, 5, 5, 6], [7, 8, 9, 10]], [[2, 2, 2, 2], [3, 3, 3, 3], [4, 11, 4, 4]]])
#
# # var_z = np.max(a, axis=-1)
# print(a.shape)
# var_z = []
# # a = a.reshape(3, 2, 4)
# for i in range(a.shape[1]):
#     print(a[:, i])
#     sorted_ind = np.argsort(np.max(a[:, i], axis=1))[-1]
#     print(sorted_ind)
#     # print(a[:, i][sorted_ind])
#     var_z.append(a[:, i][sorted_ind])
# print(np.asarray(var_z))
# print(var_z)
# var_z = np.max(var_z, axis=1)
# print(var_z)
# print(a[np.argmax(var_z)])
# print(np.max([1, 2]))
#
# a = np.array([1, 2, 3, 4])
# print(np.insert(a, 0, 9))
import re
# a = 'black'
# a = re.split('- | _', a)
# a = a.replace('-', ' ')
# a = a.replace('_', ' ')
# print(a)

# a = np.load("QC/data/CLEVER_data1.npy")
# print(a.shape)
# a = np.random.randint(0, 10000, 500)
# np.save("/Users/qiang.hu/PycharmProjects/al_leak/new_experiments/RQ2/data/cifar10_random_3.npy", a)
# print(a)
print(np.random.randint(1, 20, 3))
