# import numpy as np
#
# arr = np.array([1, 2, 3, 4, 5])
# print(arr)
# print(arr.shape)
# print(type(np.random))
#
# my_array = np.random.random((2,3))
# print(my_array)
# my_array_column_2 = my_array[0, (1,2)]
# print(my_array_column_2)
#
# b = np.ones((25))
# b = b.reshape((5,5))
# print(b)
# print(np.dot(arr,b))
# print(arr@b)
# a = np.arange(0, 100, 10)
# print(a)
#
# # Boolean masking
# # 导入 matplotlib.pyplot 用于数据可视化
# import matplotlib.pyplot as plt
# # 导入 NumPy 库，用于数组操作
# import numpy as np
#
# # 创建一个从 0 到 2π 的等差数列，包含 50 个点
# a = np.linspace(0, 2 * np.pi, 50)
# # 计算 a 中每个点的正弦值
# b = np.sin(a)
#
# # 绘制 a 对 b 的图像，即正弦曲线
# plt.plot(a, b)
#
# # 创建一个布尔掩码，表示 b 中所有非负值
# mask = b >= 0
# # 使用布尔掩码绘制所有非负的正弦值点，用蓝色圆圈 'bo' 表示
# plt.plot(a[mask], b[mask], 'bo')
#
# plt.show()
# # 创建另一个布尔掩码，同时满足 b 非负且 a 小于等于 π/2
# mask = (b >= 0) & (a <= np.pi / 2)
# # 使用布尔掩码绘制同时满足 b 非负且 a 小于等于 π/2 的点，用绿色圆圈 'go' 表示
# plt.plot(a[mask], b[mask], 'go')
#
# # 显示图表
# plt.show()
#
# # Where
# a = np.arange(0, 100, 10)
# b = np.where(a < 50)
# c = np.where(a >= 50)[0]
# print(type(b))
# print(b) # >>>(array([0, 1, 2, 3, 4]),)
# print(c) # >>>[5 6 7 8 9]
#
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # Create an empty matrix with the same shape as x
print(y)
# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

# Now y is the following
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print(y)
print(type(y))

tt = np.array("hell0")
print(type(tt))

A = np.array([[2,1,-2],[3,0,1],[1,1,-1]])
b = np.transpose(np.array([[-3,5,-2]]))
x = np.linalg.solve(A,b)
print(x)