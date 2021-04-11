import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# blue, orange, green, red, violet, goh, pink
colors_plt = plt.rcParams['axes.prop_cycle'].by_key()['color']

# plt.figure(figsize=(15, 5))
# ax_1 = plt.subplot(1, 3, 1)
# ax_2 = plt.subplot(1, 3, 2)
# ax_3 = plt.subplot(1, 3, 3)


# def smooth(s, window=20):
#     weights = np.repeat(1.0, window) / window
#     return np.convolve(s, weights, 'valid')


# for i, name in enumerate(['k10.txt', 'k20.txt', 'k64.txt', 'k10_small.txt', 'small_1111.txt']):
#     clean_errs = []
#     noise_errs = []
#     train_errs = []
#     f = open(name, "r")
#     for line in f.readlines():
#         if 'clean: ' in line:
#             clean_errs += [1.0 - float(line.split()[-1].split('=')[-1])]
#         if 'noise: ' in line:
#             noise_errs += [1.0 - float(line.split()[-1].split('=')[-1])]
#         if 'batch:390/391, loss' in line:
#             train_errs += [1.0 - float(line.split()[-1].split('=')[-1])]

#     ax_1.plot(smooth(train_errs), colors_plt[i])
#     ax_2.plot(smooth(noise_errs), colors_plt[i])
#     ax_3.plot(smooth(clean_errs), colors_plt[i])

# ax_1.set_xscale('log')
# ax_2.set_xscale('log')
# ax_3.set_xscale('log')
# plt.savefig('res.png')


kkk_2 = np.concatenate([
    300.0 * np.load('kkk_2_0.npy'),
    300.0 * np.load('kkk_2_1.npy'),
    300.0 * np.load('kkk_2_2.npy'),
    300.0 * np.load('kkk_2_3.npy'),
    300.0 * np.load('kkk_2_4.npy'),
    300.0 * np.load('kkk_2_5.npy'),
    300.0 * np.load('kkk_2_6.npy'),
    300.0 * np.load('kkk_2_7.npy'),
    300.0 * np.load('kkk_2_8.npy'),
    300.0 * np.load('kkk_2_9.npy')], 0)


print('df')
kkk_4 = np.concatenate([
    300.0 * np.load('kkk_4_0.npy'),
    300.0 * np.load('kkk_4_1.npy'),
    300.0 * np.load('kkk_4_2.npy'),
    300.0 * np.load('kkk_4_3.npy'),
    300.0 * np.load('kkk_4_4.npy'),
    300.0 * np.load('kkk_4_5.npy'),
    300.0 * np.load('kkk_4_6.npy'),
    300.0 * np.load('kkk_4_7.npy'),
    300.0 * np.load('kkk_4_8.npy'),
    300.0 * np.load('kkk_4_9.npy')], 0)

print('df')
kkk_10 = np.concatenate([
    300.0 * np.load('kkk_10_0.npy'),
    300.0 * np.load('kkk_10_1.npy'),
    300.0 * np.load('kkk_10_2.npy'),
    300.0 * np.load('kkk_10_3.npy'),
    300.0 * np.load('kkk_10_4.npy'),
    300.0 * np.load('kkk_10_5.npy'),
    300.0 * np.load('kkk_10_6.npy'),
    300.0 * np.load('kkk_10_7.npy'),
    300.0 * np.load('kkk_10_8.npy'),
    300.0 * np.load('kkk_10_9.npy')], 0)

print('df')
kkk_20 = np.concatenate([
    300.0 * np.load('kkk_20_0.npy'),
    300.0 * np.load('kkk_20_1.npy'),
    300.0 * np.load('kkk_20_2.npy'),
    300.0 * np.load('kkk_20_3.npy'),
    300.0 * np.load('kkk_20_4.npy'),
    300.0 * np.load('kkk_20_5.npy'),
    300.0 * np.load('kkk_20_6.npy'),
    300.0 * np.load('kkk_20_7.npy'),
    300.0 * np.load('kkk_20_8.npy'),
    300.0 * np.load('kkk_20_9.npy')], 0)

print('df')
kkk_50 = np.concatenate([
    300.0 * np.load('kkk_50_0.npy'),
    300.0 * np.load('kkk_50_1.npy'),
    300.0 * np.load('kkk_50_2.npy'),
    300.0 * np.load('kkk_50_3.npy'),
    300.0 * np.load('kkk_50_4.npy'),
    300.0 * np.load('kkk_50_5.npy'),
    300.0 * np.load('kkk_50_6.npy'),
    300.0 * np.load('kkk_50_7.npy'),
    300.0 * np.load('kkk_50_8.npy'),
    300.0 * np.load('kkk_50_9.npy')], 0)

print('df')
kkk_64 = np.concatenate([
    300.0 * np.load('kkk_64_0.npy'),
    300.0 * np.load('kkk_64_1.npy'),
    300.0 * np.load('kkk_64_2.npy'),
    300.0 * np.load('kkk_64_3.npy'),
    300.0 * np.load('kkk_64_4.npy'),
    300.0 * np.load('kkk_64_5.npy'),
    300.0 * np.load('kkk_64_6.npy'),
    300.0 * np.load('kkk_64_7.npy'),
    300.0 * np.load('kkk_64_8.npy'),
    300.0 * np.load('kkk_64_9.npy')], 0)


mean_2 = kkk_2[::2].sum(0) / 3000
mean_4 = kkk_4[::2].sum(0) / 3000
mean_10 = kkk_10[::2].sum(0) / 3000
mean_20 = kkk_20[::2].sum(0) / 3000
mean_50 = kkk_50[::2].sum(0) / 3000
mean_64 = kkk_64[::2].sum(0) / 3000

sqr_mean_2 = kkk_2[1::2].sum(0) / 3000
sqr_mean_4 = kkk_4[1::2].sum(0) / 3000
sqr_mean_10 = kkk_10[1::2].sum(0) / 3000
sqr_mean_20 = kkk_20[1::2].sum(0) / 3000
sqr_mean_50 = kkk_50[1::2].sum(0) / 3000
sqr_mean_64 = kkk_64[1::2].sum(0) / 3000

std_2 = np.sqrt(sqr_mean_2 - mean_2 ** 2)
std_4 = np.sqrt(sqr_mean_4 - mean_4 ** 2)
std_10 = np.sqrt(sqr_mean_10 - mean_10 ** 2)
std_20 = np.sqrt(sqr_mean_20 - mean_20 ** 2)
std_50 = np.sqrt(sqr_mean_50 - mean_50 ** 2)
std_64 = np.sqrt(sqr_mean_64 - mean_64 ** 2)

plt.plot(np.sort(std_2)[::-1], label='2')
plt.plot(np.sort(std_4)[::-1], label='4')
plt.plot(np.sort(std_10)[::-1], label='10')
plt.plot(np.sort(std_20)[::-1], label='20')
plt.plot(np.sort(std_50)[::-1], label='50')
plt.plot(np.sort(std_64)[::-1], label='64')

plt.xscale('log')
plt.legend()
plt.savefig('res2.png')

import pdb; pdb.set_trace()
