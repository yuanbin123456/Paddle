import paddle
from paddle.fluid import core
from paddle.fluid import framework
#paddle.enable_static()
#print(paddle.in_dynamic_mode())
#print (core.is_compiled_with_sw())
#place = core.XPUPlace(0)
#print(core.Place().is_compiled_with_xpu())
#p = core.Place()
#p.set_place(core.SWPlace())
#print(p, p.is_sw_place())
#print(place)
framework._set_expected_place(core.SWPlace())
#paddle.set_device(core.SWPlace())
x = paddle.to_tensor([-1, -2, -3, -4], dtype='float32')
res = paddle.fluid.layers.diag(x)
print(res)
