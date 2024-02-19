from itertools import combinations
import numpy as np

a = [[0], [0,1,2], [1,2,3]]
# index = []
indices = []
i = 0
attn_matrix = np.array([[0,1,2,3,4,5,6],
                       [7,8,9,10,11,12,13],
                       [14,15,16,17,18,19,20],
                       [21,22,23,24,25,26,27],
                       [28,29,30,31,32,33,34],
                       [35,36,37,38,39,40,41],
                       [42,43,44,45,46,47,48]])

print(attn_matrix.shape)

for path in a:
    path[0:]= range(i, i+len(path))
    i = i+len(path)

for path in a:
    index = list(combinations(path, 2))
    indices.extend(index)
    print(index)

index_matrix = indices[5][0:2]
print(attn_matrix[index_matrix])

print(indices)