from __future__ import division
import mxnet as mx
import numpy as np
import cv2

prefix = '/home/cmf/models/insightface/model-MobileFaceNet-arcface-ms1m-refine-v1/model-y1-test2/model'
epoch = 0
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
all_layers = sym.get_internals()
sym = all_layers['fc1_output']
ctx = mx.cpu()
print(sym, ctx)
model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
data_shape = (1, 3) + (112, 112)
model.bind(data_shapes=[('data', data_shape)])
mx.viz.print_summary(sym, shape={'data': (1, 3, 112, 112)})
mx.viz.plot_network(sym, shape={'data': (1, 3, 112, 112)}).view()
# for layer in model:
#     print(layer.name)

img = cv2.imread('/home/cmf/Pictures/clipboard.png')
img = cv2.resize(img, (112, 112))
# print(arg_params,)
model.set_params(arg_params, aux_params)
data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

data = np.ones((112, 112, 3), dtype=np.float32)
data = (img - 127.5) / 128
print(data)
data = np.transpose(data, (2, 0, 1))
data = np.expand_dims(data, axis=0)
data = mx.nd.array(data)
db = mx.io.DataBatch(data=(data,))
model.forward(db, is_train=False)
embedding = model.get_outputs()[0].asnumpy()

print(embedding[0])
