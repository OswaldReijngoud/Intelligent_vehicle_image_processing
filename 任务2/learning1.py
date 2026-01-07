
import numpy as np

A = np.array([[1, 2], [3, 4]])

# 场景：我们需要做一个“掩码(mask)”，通常初始全为 1
mask = np.ones_like(A)

# 场景：我们需要把这块区域初始值设为 -1 (表示未探索)
unexplored = np.full_like(A, -1)

print("全1数组 (ones_like):\n", mask)
print("指定值数组 (full_like):\n", unexplored)