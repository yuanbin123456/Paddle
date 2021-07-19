import paddle
import paddle.nn as nn
import numpy as np
from paddle.fluid import core
from paddle.fluid import framework

# max pool2d
framework._set_expected_place(core.SWPlace())
input = paddle.uniform(shape=[1, 2, 32, 32], dtype='float32', min=-1, max=1)
MaxPool2D = nn.MaxPool2D(kernel_size=2,
                         stride=2, padding=0)
output = MaxPool2D(input)
print("input: ",input)
print(output)


