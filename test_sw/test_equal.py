import paddle

x = paddle.to_tensor([1.0, 2.0, 3.0])
y = paddle.to_tensor([1.0, 3.0, 2.0])
result1 = paddle.equal(x, y)
print(result1)  # result1 = [True False False]