import paddle
import paddle.nn as nn
from paddle.fluid import core
from paddle.fluid import framework
import numpy as np
framework._set_expected_place(core.SWPlace())
x_var = paddle.uniform((2, 4, 8, 8), dtype='float32', min=-1., max=1.)

#x_var = paddle.uniform((2, 4, 8, 8), dtype='float32', min=-1., max=1.)
print(x_var)
#x_var = paddle.to_tensor([[[[1, -2, -3, -4]]]], dtype='float32')
conv = nn.Conv2D(4, 6, (3, 3))
y_var = conv(x_var)
y_np = y_var.numpy()
print(y_np)

