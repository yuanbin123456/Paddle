import paddle
from paddle.fluid import core
from paddle.fluid import framework

framework._set_expected_place(core.SWPlace())
a = paddle.to_tensor(2.0, stop_gradient=False)
b = paddle.to_tensor(5.0, stop_gradient=True)
c = a * b
c.backward()
print("Tensor a's grad is: {}".format(a.grad))
print("Tensor b's grad is: {}".format(b.grad))