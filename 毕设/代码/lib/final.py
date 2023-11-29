import pickle as pkl
import numpy as np

TopK = 180
num_classes = 9

with open('temp.pkl', 'rb') as f:
    data = pkl.load(f)

# -------------- 之前的方法 ----------------- # 
# results = {}  # 保存每类前20
# for i in range(9):
#     results[i] = {}
#     samples = data[i]  # 获取每类保存的数据 [filename, label, data(32, 512)]
#     d_array = [samples[k][-1] for k in range(len(samples))]

#     # 先求和，再排序
#     d_sum = np.sum(d_array, axis=0)  # shape =（512, ）
#     d_sort_value = np.sort(d_sum)[::-1]    # 贡献值, 从大到小
#     d_sort_index = np.argsort(d_sum)[::-1] # 字段索引, 从大到小


#     results[i]['contribute'] = d_sort_value
#     results[i]['index'] = d_sort_index


# print(results)
# -------------- 之前的方法 ----------------- # 


# -------------- 修改后的方法 ----------------- # 
results = []
for i in range(num_classes):
    samples = data[i]  # 获取每类保存的数据 [filename, label, data(32, 512)]
    d_list = [samples[k][-1] for k in range(len(samples))]
    results += d_list


# step 1: (n, 32, 512)
results_array = np.array(results)
# step 2: (1, 32, 512), 按第一维度求和得到 (32, 512)
results_array = np.sum(results_array, axis=0)
# step 3: (32, 512), 按第一维度取最大值得到 (512, )
results_array = np.max(results_array, axis=0)
# step 4: (512, )按大小排序，取前20
sort_value = np.sort(results_array)[::-1]    # 贡献值进行从大到小的排序
sort_index = np.argsort(results_array)[::-1] # 贡献值的索引从大到小的排序

sort_index += 1

print("value: {}".format(sort_value[:TopK]))
print("index: {}".format(sort_index[:TopK]))

# 输出index中小于55的值
print("*"*100)
index_topK = sort_index[:TopK]
less_than_55 = index_topK[index_topK < 55]
less_than_55_sort = np.sort(less_than_55)
print("报头字段为: {}".format(less_than_55))
print("*"*100)
print("排序后字段为: {}".format(less_than_55_sort))

# -------------- 修改后的方法 ----------------- # 