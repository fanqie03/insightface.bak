from __future__ import division
import mxnet as mx
import numpy as np
import cv2


prefix='/home/cmf/models/model-y1-test2/model'
epoch=0
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
all_layers = sym.get_internals()
sym = all_layers['fc1_output']
ctx = mx.cpu()
model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
data_shape = (1, 3) + (112, 112)
model.bind(data_shapes=[('data', data_shape)])

img = cv2.imread('/home/cmf/Pictures/Selection_001.png')
img = cv2.resize(img, (112, 112))

model.set_params(arg_params, aux_params)
data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

data = np.ones((112, 112, 3), dtype=np.int8)
data = np.transpose(data, (2, 0, 1))
data = np.expand_dims(data, axis=0)
data = mx.nd.array(data)
db = mx.io.DataBatch(data=(data,))
model.forward(db, is_train=False)
embedding = model.get_outputs()[0].asnumpy()

print(embedding[0])