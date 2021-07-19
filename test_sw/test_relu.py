import paddle
import paddle.nn.functional as F
import numpy as np

x = paddle.to_tensor(np.array([-2, 0, 1]).astype('float32'))
out = F.relu(x) # [0., 0., 1.]
print(out)