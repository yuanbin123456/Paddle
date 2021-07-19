import paddle.fluid as fluid
import numpy
from paddle.fluid import core
# First create the Executor.
place = core.SWPlace() # fluid.CUDAPlace(0)
exe = fluid.Executor(place)

data = fluid.layers.data(name='X', shape=[1], dtype='float32')
hidden = fluid.layers.diag(input=data, size=10)
#loss = fluid.layers.mean(hidden)
#adam = fluid.optimizer.Adam()
#adam.minimize(loss)

# Run the startup program once and only once.
exe.run(fluid.default_startup_program())

x = numpy.random.random(size=(10, 1)).astype('float32')
outs = exe.run(feed={'X': x})
