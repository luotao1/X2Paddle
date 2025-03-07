from __future__ import print_function
import paddle
import sys
import os
import numpy
import pickle


def rel_err(x, y):
    return numpy.max(
        numpy.abs(x - y) / (numpy.maximum(numpy.abs(x), numpy.abs(y)) + 1e-08))


with open('../dataset/DeepLabv3_ResNet50/pytorch_input.pkl', 'rb') as inp:
    input_data = pickle.load(inp)["data0"]
with open('../dataset/DeepLabv3_ResNet50/pytorch_output.pkl', 'rb') as oup:
    pytorch_output = pickle.load(oup)
f = open("result.txt", "w")
f.write("======DeepLabv3_ResNet50: \n")
try:
    # trace
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    [prog, inputs, outputs] = paddle.static.load_inference_model(
        path_prefix="pd_model_trace/inference_model/model", executor=exe)
    result = exe.run(prog, feed={inputs[0]: input_data}, fetch_list=outputs)
    df0 = pytorch_output["aux"] - result[0]
    df1 = pytorch_output["out"] - result[1]
    if numpy.max(numpy.fabs(df0)) > 1e-04:
        print("Trace Failed", file=f)
    elif numpy.max(numpy.fabs(df1)) > 1e-04:
        print("Trace Failed", file=f)
    else:
        print("Trace Successed", file=f)

    # script
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    [prog, inputs, outputs] = paddle.static.load_inference_model(
        path_prefix="pd_model_script/inference_model/model", executor=exe)
    result = exe.run(prog, feed={inputs[0]: input_data}, fetch_list=outputs)
    df0 = pytorch_output["aux"] - result[0]
    df1 = pytorch_output["out"] - result[1]
    if numpy.max(numpy.fabs(df0)) > 1e-04:
        print("Script Failed\n", file=f)
    elif numpy.max(numpy.fabs(df1)) > 1e-04:
        print("Script Failed\n", file=f)
    else:
        print("Script Successed\n", file=f)
except:
    print("!!!Failed\n", file=f)
f.close()
