import paddle
from paddle.fluid import core
from paddle.fluid import framework

framework._set_expected_place(core.SWPlace())
input_data = paddle.rand(shape=[5, 100])
label_data = paddle.randint(0, 100, shape=[5,1], dtype="int64")
weight_data = paddle.rand([100])
print(label_data)
loss = paddle.nn.functional.cross_entropy(input=input_data, label=label_data, weight=weight_data)
print(loss)