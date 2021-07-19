import paddle.fluid as fluid
import paddle

paddle.enable_static()
a = fluid.data(name="a", shape=[3], dtype='float32')
#b = fluid.data(name="b", shape=[None, 1], dtype='int64')
 
# 组建网络（此处网络仅由一个操作构成，即elementwise_add）
#result = fluid.layers.elementwise_add(a,b)
result = fluid.layers.diag(a) 
# 准备运行网络
sw = fluid.core.SWPlace() # 定义运算设备，这里选择在CPU下训练
exe = fluid.Executor(sw) # 创建执行器
exe.run(fluid.default_startup_program()) # 网络参数初始化
 
# 读取输入数据
import numpy as np
#data_1 = int(input("Please enter an integer: a="))
#data_2 = int(input("Please enter an integer: b="))
x = np.ones(shape=[3], dtype=np.float32)
#y = numpy.array([[data_2]])
 
# 运行网络
outs = exe.run(
    feed={'a':x}, # 将输入数据x, y分别赋值给变量a，b
    fetch_list=[result] # 通过fetch_list参数指定需要获取的变量结果
    )
 
# 输出计算结果
print (outs)