import paddle
from paddle.fluid import core
from paddle.fluid import framework

framework._set_expected_place(core.SWPlace())
# Define the linear layer.
weight_attr = paddle.ParamAttr(
    name="weight",
    initializer=paddle.nn.initializer.Constant(value=0.5))
bias_attr = paddle.ParamAttr(
    name="bias",
    initializer=paddle.nn.initializer.Constant(value=1.0))
linear = paddle.nn.Linear(2, 4, weight_attr=weight_attr, bias_attr=bias_attr)
# linear.weight: [[0.5 0.5 0.5 0.5]
#                 [0.5 0.5 0.5 0.5]]
# linear.bias: [1. 1. 1. 1.]

x = paddle.randn((3, 2), dtype="float32")
y = linear(x)

print(y)