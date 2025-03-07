# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from x2paddle.decoder.onnx_decoder import ONNXGraph, ONNXGraphNode, ONNXGraphDataNode
from x2paddle.core.graph import GraphNode
from x2paddle.core.util import *
from functools import reduce
import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
import logging as _logging
from collections import OrderedDict
import math
import os
import copy
import sys
import shutil

_logger = _logging.getLogger()


def _const_weight_or_none(node, necessary=False):
    if 'Constant' in node.layer_type:
        return node.value
    if isinstance(node, ONNXGraphDataNode):
        return node.weight
    if necessary:
        assert '{} should be an initializer or Constant operator.'.format(
            node.name)
    return None


def _rename_or_remove_weight(weights,
                             origin_name,
                             target_name=None,
                             is_remove=True,
                             rename_mapper=None):
    '''
    Rename parameters by Paddle's naming rule of parameters.

    Args:
        weights(dict[String:np.ndarray]): Dict stored paramters, the key in weights is name of parameter.
        origin_name(String): Name of parameter to rename or remove.
        target_name(String, optional): if target_name is not None, add new key-value pair
            {target_name:weights[origin_name]} to weights, and target_name must follow paddle's
            naming rule of parameters. Default: None.
        is_remove: if is_remove is True, remove origin key-value pair. Default: True.
        rename_mapper: Solved the same data is used for multiple OPs, key is old_name, value is new_name.
    Returns:
        None
    '''
    if rename_mapper is not None and origin_name in rename_mapper:
        origin_name = rename_mapper[origin_name]
        is_remove = False
    if origin_name not in weights:
        raise KeyError('{} not a key in {}'.format(origin_name, weights.keys()))
    if is_remove:
        # remove weight
        data = weights.pop(origin_name)
    else:
        data = weights[origin_name]
    if target_name is not None:
        # rename weight
        weights[target_name] = data
        rename_mapper[origin_name] = target_name


def _is_static_shape(shape):
    negtive_dims = 0
    error_dims = 0
    for dim in shape:
        if dim < 0:
            negtive_dims += 1
        if dim < -1:
            error_dims += 1
    if negtive_dims > 1:
        return False
    if error_dims > 0:
        return False
    return True


def _get_same_padding(in_size, kernel_size, stride, autopad):
    new_size = int(math.ceil(in_size * 1.0 / stride))
    pad_size = (new_size - 1) * stride + kernel_size - in_size
    pad0 = int(pad_size / 2)
    pad1 = pad_size - pad0
    if autopad == "SAME_UPPER":
        return [pad0, pad1]
    if autopad == "SAME_LOWER":
        return [pad1, pad0]


def print_mapping_info(func):

    def run_mapping(*args, **kwargs):
        node = args[1]
        try:
            res = func(*args, **kwargs)
        except:
            raise Exception("convert failed node:{}, op_type is {}".format(
                node.name[9:], node.layer_type))
        else:
            return res

    return run_mapping


class OpSet():

    def __init__(self, decoder, paddle_graph):
        super(OpSet, self).__init__()
        self.graph = decoder.graph
        self.paddle_graph = paddle_graph
        self.inputs_info = dict()
        self.weights = dict()
        self.nn_name2id = dict()
        self.done_weight_list = list()
        # solve for same data is used as an argument to multiple OPs.
        # PR link(wangjunjie06): https://github.com/PaddlePaddle/X2Paddle/pull/728
        self.rename_mapper = dict()
        self.elementwise_ops = {
            'Add': 'paddle.add',
            'Div': 'paddle.divide',
            'Sub': 'paddle.subtract',
            'Mul': 'paddle.multiply',
            'Pow': 'paddle.pow',
            'Less': 'paddle.less_than',
            'LessOrEqual': 'paddle.less_equal',
        }

        self.directly_map_ops = {
            'Ceil': ['paddle.ceil'],
            # reduce function
            'ReduceMean': [
                'paddle.mean',
                dict(axes='axis', keepdims='keepdim'),
                dict(axes=None, keepdims=True)
            ],
            'ReduceMin': [
                'paddle.min',
                dict(axes='axis', keepdims='keepdim'),
                dict(axes=None, keepdim=True)
            ],
            'ReduceMax': [
                'paddle.max',
                dict(axes='axis', keepdims='keepdim'),
                dict(axes=None, keepdim=True)
            ],
            'ReduceProd': [
                'paddle.prod',
                dict(axes='axis', keepdims='keepdim'),
                dict(axes=None, keepdim=True)
            ],
            # active function
            'Relu': ['paddle.nn.ReLU'],
            'LeakyRelu': [
                'paddle.nn.LeakyReLU',
                dict(alpha='negative_slope'),
                dict(negative_slope=.01)
            ],
            'Elu':
            ['paddle.nn.functional.elu',
             dict(alpha='alpha'),
             dict(alpha=1.)],
            'ThresholdedRelu': [
                'paddle.nn.functional.thresholded_relu',
                dict(alpha='threshold'),
                dict(alpha=1.)
            ],
            'Tanh': ['paddle.nn.Tanh'],
            'Sigmoid': ['paddle.nn.Sigmoid'],
            'Softsign': ['paddle.nn.Softsign'],
            'Softplus': [
                'paddle.nn.Softplus',
                dict(threshold='threshold'),
                dict(threshold=float(sys.maxsize))
            ],
            'Exp': ['paddle.exp'],
            'Log': ['paddle.log'],
            'LogSoftmax': [
                'paddle.nn.functional.log_softmax',
                dict(axis='axis'),
                dict(axis=1)
            ],
            'Softmax': ['paddle.nn.Softmax',
                        dict(axis='axis'),
                        dict(axis=1)],
            'Sqrt': ['paddle.sqrt'],
            'Floor': ['paddle.floor'],
            'Abs': ['paddle.abs'],
            'Erf': ['paddle.erf'],
            'Sin': ['paddle.sin'],
            'Cos': ['paddle.cos'],
        }

    @print_mapping_info
    def directly_map(self, node, *args, **kwargs):
        inputs = node.layer.input
        assert len(inputs) == 1, 'directly_map error with multi inputs'
        input = self.graph.get_input_node(node, idx=0, copy=True)
        onnx_attrs = node.attr_map
        if '' in onnx_attrs:
            onnx_attrs.pop('')
        if '_' in onnx_attrs:
            onnx_attrs.pop('_')
        op_info = self.directly_map_ops[node.layer_type]
        paddle_op = op_info[0]
        layer_attrs = dict()
        if len(op_info) > 1:
            attrs_name_map_dict = op_info[1]
            for onnx_attr_name, pd_attr_name in attrs_name_map_dict.items():
                if onnx_attr_name in onnx_attrs:
                    # convert for dynamic code, mv 0 to False, 1 to True
                    if pd_attr_name == "keepdim":
                        keepdims = False if onnx_attrs[
                            onnx_attr_name] == 0 else True
                        onnx_attrs[onnx_attr_name] = keepdims
                    layer_attrs[pd_attr_name] = onnx_attrs[onnx_attr_name]
                else:
                    layer_attrs[pd_attr_name] = op_info[2][onnx_attr_name]
        if paddle_op.startswith("paddle.nn") and 'functional' not in paddle_op:
            op_name = paddle_op[10:].lower()
            op_name = name_generator(op_name, self.nn_name2id)
            output_name = node.name
            layer_outputs = [op_name, output_name]

            self.paddle_graph.add_layer(kernel=paddle_op,
                                        inputs={"x": input.name},
                                        outputs=layer_outputs,
                                        **layer_attrs)
        else:
            self.paddle_graph.add_layer(kernel=paddle_op,
                                        inputs={"x": input.name},
                                        outputs=[node.name],
                                        **layer_attrs)

    @print_mapping_info
    def elementwise_map(self, node):
        op_type = self.elementwise_ops[node.layer_type]
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_input_node(node, idx=1, copy=True)
        inputs_dict = {'x': val_x.name, 'y': val_y.name}
        self.paddle_graph.add_layer(op_type,
                                    inputs=inputs_dict,
                                    outputs=[node.name])

    @print_mapping_info
    def place_holder(self, node):
        shape = node.out_shapes[0]
        for i, dim_shape in enumerate(shape):
            if dim_shape == 0 and i == 0:
                shape[i] = 1
            if dim_shape == 0 and i != 0:
                assert 'shape of input is not assigned'
        self.paddle_graph.add_layer(kernel="paddle.to_tensor",
                                    inputs={},
                                    outputs=[node.name],
                                    data=node.name)
        self.inputs_info[node.name] = [shape, node.dtype]

    @print_mapping_info
    def create_parameter(self, node, parameter=None):
        if parameter is not None:
            node = parameter
        dtype = node.dtype
        shape = node.out_shapes[0]

        if hasattr(node.weight, "shape") and len(node.weight.shape) == 0:
            if node.weight == float('inf') or node.weight == float('-inf'):
                node.weight = string(node.weight)
            self.paddle_graph.add_layer("paddle.full",
                                        inputs={},
                                        outputs=[node.name],
                                        dtype=string(dtype),
                                        shape=[1],
                                        fill_value=node.weight)
        else:
            self.weights[node.name] = node.weight
            self.paddle_graph.add_layer(
                "self.create_parameter",
                inputs={},
                outputs=[node.name],
                shape=shape,
                attr=string(node.name),
                dtype=string(dtype),
                default_initializer="paddle.nn.initializer.Constant(value=0.0)")

    def _pad_if_asymmetric(self, node, pads, val_name):  # pads: SSEE
        assert len(pads) & 1 == 0
        symmetric = True
        ndims = len(pads) // 2
        for idx_dim in range(ndims):
            if pads[idx_dim] != pads[ndims + idx_dim]:
                symmetric = False
                break
        if symmetric:
            return pads[:ndims], val_name
        val_padded = self.Pad(node, op_independent=False)
        return [0] * ndims, val_padded

    def _interpolate(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        inputs = {'x': val_x.name}
        attrs = dict()
        val_x_shape = val_x.out_shapes[0]
        if node.layer_type == 'Resize':
            if len(node.layer.input) == 2:
                # opset 10
                val_scales = self.graph.get_input_node(node, idx=1, copy=True)
                # TODO(syf): paddle.nn.functional.interpolate will support the length
                # which is the same as the rank of input.
                scale_values = _const_weight_or_none(val_scales)
                if scale_values is not None:
                    attrs['scale_factor'] = self.weights[
                        val_scales.name].tolist()[2:]
                else:
                    var_nc, var_hw = val_scales.name + '_nc', val_scales.name + '_hw'
                    self.paddle_graph.add_layer('paddle.split',
                                                inputs={"x": val_scales.name},
                                                outputs=[var_nc, var_hw],
                                                num_or_sections=[2, 2],
                                                axis=0)
                    inputs['scale_factor'] = var_hw
                mode = node.get_attr('mode', 'nearest')
                attrs.update({
                    "align_corners": False,
                    "mode": string(mode),
                    "align_mode": 1
                })
                if mode == "linear" and len(val_x_shape) == 4:
                    attrs["mode"] = string("bilinear")
                self.paddle_graph.add_layer(
                    kernel="paddle.nn.functional.interpolate",
                    inputs=inputs,
                    outputs=[node.name],
                    **attrs)
                return
            elif len(node.layer.input) == 3:
                # opset 11
                try:
                    # to avoid the error causeed by NULL value of resize inputs.
                    val_scales = self.graph.get_input_node(node,
                                                           idx=2,
                                                           copy=True)
                except:
                    val_scales = self.graph.get_input_node(node,
                                                           idx=1,
                                                           copy=True)
                # TODO(syf): paddle.nn.functional.interpolate will support the length
                # which is the same as the rank of input.
                attrs['scale_factor'] = self.weights[
                    val_scales.name].tolist()[2:]
            elif len(node.layer.input) == 4:
                # opset 11
                val_sizes = self.graph.get_input_node(node, idx=3, copy=True)
                size_values = _const_weight_or_none(val_sizes)
                if len(val_x_shape) == 3:
                    var_n, var_hw = val_sizes.name + '_n', val_sizes.name + '_hw'
                    self.paddle_graph.add_layer('paddle.split',
                                                inputs={"x": val_sizes.name},
                                                outputs=[var_n, var_hw],
                                                num_or_sections=[1, 2],
                                                axis=0)
                    self.paddle_graph.add_layer("paddle.cast",
                                                inputs={"x": var_hw},
                                                outputs=[var_hw],
                                                dtype=string('int32'))
                    inputs['size'] = var_hw
                    attrs = {
                        "align_corners": False,
                        "mode": string(node.get_attr('mode', 'nearest'))
                    }
                    mode = node.get_attr('mode', 'nearest')
                    if mode == "linear":
                        attrs["mode"] = string("bilinear")
                    if node.get_attr('coordinate_transformation_mode',
                                     'half_pixel') == 'pytorch_half_pixel':
                        attrs["align_corners"] = False
                        attrs["align_mode"] = 0
                    if node.get_attr('coordinate_transformation_mode',
                                     'half_pixel') == 'align_corners':
                        attrs["align_corners"] = True
                    self.paddle_graph.add_layer('paddle.unsqueeze',
                                                inputs={"x": val_x.name},
                                                outputs=[val_x.name],
                                                axis=0)
                    self.paddle_graph.add_layer(
                        kernel="paddle.nn.functional.interpolate",
                        inputs=inputs,
                        outputs=[node.name],
                        **attrs)
                    self.paddle_graph.add_layer('paddle.squeeze',
                                                inputs={"x": node.name},
                                                outputs=[node.name],
                                                axis=0)
                else:
                    if size_values is not None:
                        attrs["size"] = [size_values[2], size_values[3]]
                    else:
                        var_nc, var_hw = val_sizes.name + '_nc', val_sizes.name + '_hw'
                        self.paddle_graph.add_layer(
                            'paddle.split',
                            inputs={"x": val_sizes.name},
                            outputs=[var_nc, var_hw],
                            num_or_sections=[2, 2],
                            axis=0)
                        self.paddle_graph.add_layer("paddle.cast",
                                                    inputs={"x": var_hw},
                                                    outputs=[var_hw],
                                                    dtype=string('int32'))
                        inputs['size'] = var_hw
                    attrs.update({
                        "align_corners":
                        False,
                        "mode":
                        string(node.get_attr('mode', 'nearest'))
                    })
                    mode = node.get_attr('mode', 'nearest')
                    if mode == "linear":
                        attrs["mode"] = string("bilinear")
                    if node.get_attr('coordinate_transformation_mode',
                                     'half_pixel') == 'pytorch_half_pixel':
                        attrs["align_corners"] = False
                        attrs["align_mode"] = 0
                    if node.get_attr('coordinate_transformation_mode',
                                     'half_pixel') == 'align_corners':
                        attrs["align_corners"] = True
                    self.paddle_graph.add_layer(
                        kernel="paddle.nn.functional.interpolate",
                        inputs=inputs,
                        outputs=[node.name],
                        **attrs)
                return
        elif node.layer_type == 'Upsample':
            if len(node.layer.input) == 2:
                val_scales = self.graph.get_input_node(node, idx=1, copy=True)
                self.paddle_graph.add_layer("paddle.slice",
                                            inputs={"input": val_scales.name},
                                            outputs=[val_scales.name],
                                            axes=[0],
                                            starts=[2],
                                            ends=[4])
                inputs['scale_factor'] = val_scales.name
            else:
                val_scales = node.get_attr('scales')[2:]

        mode = node.get_attr('mode', 'nearest')
        attrs.update({
            "align_corners": False,
            "mode": string(mode),
            "align_mode": 1
        })
        if len(node.layer.input) == 1:
            attrs["scale_factor"] = val_scales
        if mode == "linear" and len(val_x_shape) == 4:
            attrs["mode"] = string("bilinear")
            if node.get_attr('coordinate_transformation_mode',
                             'half_pixel') == 'pytorch_half_pixel':
                attrs["align_corners"] = False
                attrs["align_mode"] = 0
            else:
                attrs["align_corners"] = True
        self.paddle_graph.add_layer(kernel="paddle.nn.functional.interpolate",
                                    inputs=inputs,
                                    outputs=[node.name],
                                    **attrs)

    @print_mapping_info
    def CumSum(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        axis = self.graph.get_input_node(node, idx=1, copy=True)
        axis_values = _const_weight_or_none(axis)
        assert axis_values is not None, 'Axis only support constant tensor!'
        layer_attrs = {'axis': axis_values}
        self.paddle_graph.add_layer('paddle.cumsum',
                                    inputs={"x": val_x.name},
                                    outputs=[node.name],
                                    **layer_attrs)

    @print_mapping_info
    def HardSigmoid(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        alpha = node.get_attr('alpha', 0.2)
        beta = node.get_attr('beta', 0.5)
        self.paddle_graph.add_layer(kernel="paddle.scale",
                                    inputs={"x": val_x.name},
                                    outputs=[node.name + "_val"],
                                    scale=alpha,
                                    bias=beta)
        self.paddle_graph.add_layer(kernel="paddle.clip",
                                    inputs={"x": node.name + "_val"},
                                    outputs=[node.name],
                                    min=0.0,
                                    max=1.0)

    @print_mapping_info
    def Shape(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        self.paddle_graph.add_layer(kernel="paddle.shape",
                                    inputs={"input": val_x.name},
                                    outputs=[node.name])
        self.paddle_graph.add_layer('paddle.cast',
                                    inputs={"x": node.name},
                                    outputs=[node.name],
                                    dtype=string('int64'))

    @print_mapping_info
    def RoiAlign(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_rois = self.graph.get_input_node(node, idx=1, copy=True)

        pooled_height = node.get_attr('output_height')
        pooled_width = node.get_attr('output_width')
        spatial_scale = node.get_attr('spatial_scale')
        sampling_ratio = node.get_attr('sampling_ratio')
        val_rois_shape = val_rois.name + '_shape'
        self.paddle_graph.add_layer(kernel="paddle.shape",
                                    inputs={"input": val_rois.name},
                                    outputs=[val_rois_shape])
        val_rois_num = val_rois.name + '_num'
        if len(val_rois.out_shapes[0]) == 4:
            self.paddle_graph.add_layer(
                'paddle.split',
                inputs={"x": val_rois_shape},
                outputs=[val_rois_num, ' _', ' _', ' _'],
                num_or_sections=[1, 1, 1, 1],
                axis=0)
        elif len(val_rois.out_shapes[0]) == 2:
            self.paddle_graph.add_layer('paddle.split',
                                        inputs={"x": val_rois_shape},
                                        outputs=[val_rois_num, ' _'],
                                        num_or_sections=[1, 1],
                                        axis=0)
        layer_attrs = {
            'pooled_height': pooled_height,
            'pooled_width': pooled_width,
            'spatial_scale': spatial_scale,
            'sampling_ratio': sampling_ratio,
        }
        self.paddle_graph.add_layer('custom_layer:ROIAlign',
                                    inputs={
                                        'input': val_x.name,
                                        'rois': val_rois.name,
                                        'rois_num': val_rois_num
                                    },
                                    outputs=[node.name],
                                    **layer_attrs)

    @print_mapping_info
    def MaxRoiPool(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_rois = self.graph.get_input_node(node, idx=1, copy=True)

        spatial_scale = node.get_attr('spatial_scale')
        pooled_height, pooled_width = node.get_attr('pooled_shape')
        layer_attrs = {
            'pooled_height': pooled_height,
            'pooled_width': pooled_width,
            'spatial_scale': spatial_scale,
        }
        self.paddle_graph.add_layer('custom_layer:ROIPooling',
                                    inputs={
                                        'input': val_x.name,
                                        'rois': val_rois.name
                                    },
                                    outputs=[node.name],
                                    **layer_attrs)

    @print_mapping_info
    def Pad(self, node, op_independent=True):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        pads = node.get_attr('pads')
        is_pads_attr = True
        if pads is None:
            val_pad = self.graph.get_input_node(node, idx=1, copy=True)
            pad_shape = val_pad.out_shapes[0]
            is_pads_attr = False
            pads = _const_weight_or_none(val_pad)
            if pads is not None:
                is_pads_attr = True
        mode = node.get_attr('mode', 'constant')
        if mode in ["edge"]:
            mode = "replicate"
        value = node.get_attr('value', 0.)
        data_shape = val_x.out_shapes[0]
        output_shape = node.out_shapes[0]
        assume_pad = False
        layer_attrs = {}
        layer_attrs['mode'] = string(mode)
        layer_attrs['value'] = value
        if not op_independent:
            output_name = node.name + '_paded'
        else:
            output_name = node.name
        nn_op_name = name_generator("pad", self.nn_name2id)
        layer_outputs = [nn_op_name, output_name]
        if is_pads_attr:
            paddings = []
            if len(pads) == 10 and sum(pads) == 0:
                pads = pads[0:6]
            if len(pads) in [2, 4, 6]:
                if data_shape:
                    assume_pad |= data_shape and 2 * (len(data_shape) -
                                                      2) == len(pads)  # NCHW
                if output_shape:
                    assume_pad |= output_shape and 2 * (len(output_shape) -
                                                        2) == len(pads)  # NCHW
                if assume_pad:
                    paddle_op = 'paddle.nn.Pad{}D'.format(len(output_shape) - 2)
                    paddings = np.array(pads).reshape(
                        (2, -1)).transpose().astype("int32")
                    paddings = np.flip(paddings, axis=0).flatten().tolist()
                    layer_attrs['padding'] = paddings
                else:
                    if data_shape:
                        assume_pad |= data_shape and 2 * len(data_shape) == len(
                            pads)  # NCHW
                    if output_shape:
                        assume_pad |= output_shape and 2 * len(
                            output_shape) == len(pads)  # NCHW
                    if assume_pad:
                        paddle_op = 'paddle.nn.functional.pad'
                        paddings = np.array(pads).reshape((
                            2,
                            -1)).transpose().astype("int32").flatten().tolist()
                        layer_attrs['pad'] = paddings
                    else:
                        raise Exception(
                            "The padding value {} is wrong!".format(pads))
            elif len(pads) == 8:
                if data_shape:
                    assume_pad |= data_shape and 2 * len(data_shape) == len(
                        pads)  # NCHW
                if output_shape:
                    assume_pad |= output_shape and 2 * len(output_shape) == len(
                        pads)  # NCHW
                if assume_pad:
                    paddle_op = 'paddle.nn.Pad2D'
                    # x1_begin,x2_begin,x3_begin,x4_begin,x1_end,x2_end,x3_end,x4_end->x1_begin,x1_end,x2_begin,x2_end,x3_begin,x3_end,x4_begin,x4_end
                    paddings = np.array(pads).reshape(
                        (2, -1)).transpose().astype("int32")
                    if mode == 'constant':
                        paddings = paddings.flatten().tolist()
                        layer_attrs['padding'] = paddings
                    else:
                        paddings = np.flip(paddings, axis=0).flatten().tolist()
                        if sum(paddings[:4]) == 0:
                            paddings = paddings[4:]
                            layer_attrs['padding'] = paddings
                        else:
                            layer_attrs["pad"] = paddings
                            paddle_op = "custom_layer:PadAllDim4WithOneInput"
                else:
                    paddle_op = 'paddle.nn.functional.pad'
                    layer_attrs["pad"] = np.array(pads).tolist()
            else:
                pad_data_temp = pads[0::2]
                pad_data_all = []
                for i in range(len(pad_data_temp)):
                    pad_data_all.append(pads[i])
                    pad_data_all.append(pads[len(pad_data_temp) + i])

                layer_attrs["pad"] = pad_data_all
                self.paddle_graph.add_layer('paddle.nn.functional.pad',
                                            inputs={'x': val_x.name},
                                            outputs=layer_outputs[1:],
                                            **layer_attrs)
                return

            self.paddle_graph.add_layer(
                paddle_op,
                inputs={'x': val_x.name},
                outputs=layer_outputs[1:]
                if paddle_op == 'paddle.nn.functional.pad' else layer_outputs,
                **layer_attrs)
            if not op_independent:
                return node.name + '_paded'
        else:
            pads_len = val_pad.out_shapes[0][0]
            if pads_len in [2, 4, 6]:
                if data_shape:
                    assume_pad |= data_shape and 2 * (len(data_shape) -
                                                      2) == pads_len  # NCHW
                if output_shape:
                    assume_pad |= output_shape and 2 * (len(output_shape) -
                                                        2) == pads_len  # NCHW
                if assume_pad:
                    if pads_len == 2:
                        data_format = "NCL"
                    elif pads_len == 4:
                        data_format = "NCHW"
                    else:
                        data_format = "NCDHW"
                    self.paddle_graph.add_layer("custom_layer:PadWithTwoInput",
                                                inputs={
                                                    'x': val_x.name,
                                                    'pad': val_pad.name
                                                },
                                                outputs=layer_outputs,
                                                value=value,
                                                mode=string(mode),
                                                data_format=string(data_format))
                else:
                    if data_shape:
                        assume_pad |= data_shape and 2 * len(
                            data_shape) == pads_len  # NCHW
                    if output_shape:
                        assume_pad |= output_shape and 2 * len(
                            output_shape) == pads_len  # NCHW
                    if assume_pad:
                        if pads_len == 4:
                            self.paddle_graph.add_layer(
                                "custom_layer:PadAllDim2",
                                inputs={
                                    'x': val_x.name,
                                    'pad': val_pad.name
                                },
                                outputs=layer_outputs,
                                value=value,
                                mode=string(mode))
                        else:
                            raise Exception("The padding value is wrong!")
            elif pads_len in [8, -1]:
                if data_shape:
                    assume_pad |= data_shape and 2 * len(
                        data_shape) == pads_len  # NCHW
                if output_shape:
                    assume_pad |= output_shape and 2 * len(
                        output_shape) == pads_len  # NCHW

                assume_pad |= pads_len == -1

                if assume_pad:
                    self.paddle_graph.add_layer("custom_layer:PadAllDim4",
                                                inputs={
                                                    'x': val_x.name,
                                                    'pad': val_pad.name
                                                },
                                                outputs=layer_outputs,
                                                value=value,
                                                mode=string(mode))
            else:
                raise Exception("The padding value is wrong!")
            if not op_independent:
                return node.name + '_paded'

    @print_mapping_info
    def Unsqueeze(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        axes = node.get_attr('axes')
        if axes is None:
            axes_node = self.graph.get_input_node(node, idx=1, copy=True)
            axes = _const_weight_or_none(axes_node, necessary=True)
        # deal with scalar(0D) tensor
        if len(val_x.out_shapes[0]) == 0 and len(axes) == 1 and axes[0] == 0:
            self.paddle_graph.add_layer('paddle.reshape',
                                        inputs={"x": val_x.name},
                                        outputs=[node.name],
                                        shape=[1])
        else:
            self.paddle_graph.add_layer('paddle.unsqueeze',
                                        inputs={"x": val_x.name},
                                        axis=axes,
                                        outputs=[node.name])

    @print_mapping_info
    def Shrink(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        bias = node.get_attr('bias')
        lambd = node.get_attr('lambd')
        assert bias == 0.0, 'not support bias!=0'
        self.paddle_graph.add_layer('paddle.nn.functional.hardshrink',
                                    inputs={"x": val_x.name},
                                    outputs=[node.name],
                                    threshold=lambd)

    @print_mapping_info
    def Constant(self, node):
        val_output = self.graph.get_node(node.layer.output[0], copy=True)

        value = node.get_attr('value')
        dtype = np.dtype(value.dtype)
        output_dtype = val_output.dtype
        if output_dtype:
            assert dtype == output_dtype, 'tensor dtype unmatches storage dtype'

        shape = node.get_attr('shape', None)

        if shape is None:
            shape = val_output.out_shapes[0]
        if shape is None:
            shape = list(value.shape)
            _logger.warning(
                'in (Constant -> %s): '
                'attribute "shape" of %s not inferred, '
                'using value as 1-D tensor may lead to fails', val_output.name,
                val_output.name)
        if len(value) == 1:
            value = value.tolist()
            value = value[0]
            if value == float('inf') or value == float('-inf'):
                value = string(value)
            self.paddle_graph.add_layer("paddle.full",
                                        inputs={},
                                        outputs=[node.name],
                                        dtype=string(dtype),
                                        shape=[1],
                                        fill_value=value)
        else:
            value = np.reshape(value, shape)
            self.weights[node.name] = value
            self.paddle_graph.add_layer(
                "self.create_parameter",
                inputs={},
                outputs=[node.name],
                shape=shape,
                attr=string(node.name),
                dtype=string(dtype),
                default_initializer="paddle.nn.initializer.Constant(value=0.0)")

    @print_mapping_info
    def Resize(self, node):
        self._interpolate(node)

    @print_mapping_info
    def Upsample(self, node):
        self._interpolate(node)

    @print_mapping_info
    def InstanceNormalization(self, node):
        op_name = name_generator("instanse_norm", self.nn_name2id)
        output_name = node.name
        layer_outputs = [op_name, output_name]
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_scale = self.graph.get_input_node(node, idx=1, copy=True)
        val_b = self.graph.get_input_node(node, idx=2, copy=True)
        epsilon = node.get_attr('epsilon', 1e-5)
        self.weights[op_name + '.scale'] = self.weights[val_scale.name]
        self.weights[op_name + '.bias'] = self.weights[val_b.name]
        layer_attrs = {
            'num_features': node.out_shapes[0][1],
            'epsilon': epsilon,
        }
        dim = len(val_x.out_shapes[0])
        if dim == 3:
            paddle_op = "paddle.nn.InstanceNorm1D"
        elif dim == 4:
            paddle_op = "paddle.nn.InstanceNorm2D"
        elif dim == 5:
            paddle_op = "paddle.nn.InstanceNorm3D"
        else:
            raise Exception(
                "The paddle only support 2D, 3D, 4D or 5D input in InstanceNormalization."
            )
        self.paddle_graph.add_layer(paddle_op,
                                    inputs={"x": val_x.name},
                                    outputs=layer_outputs,
                                    **layer_attrs)

    @print_mapping_info
    def Expand(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_shape = self.graph.get_input_node(node, idx=1, copy=True)
        val_x_dtype = val_x.dtype
        name_ones = node.name + '_ones'
        shape_values = _const_weight_or_none(val_shape)
        if shape_values is None:
            attr_ones = {
                'shape': val_shape.name,
                'dtype': string(val_x_dtype),
                'fill_value': 1
            }
        else:
            attr_ones = {
                'shape': shape_values.tolist(),
                'dtype': string(val_x_dtype),
                'fill_value': 1
            }
        self.paddle_graph.add_layer('paddle.full',
                                    inputs={},
                                    outputs=[name_ones],
                                    **attr_ones)
        inputs_dict = {'x': name_ones, 'y': val_x.name}
        self.paddle_graph.add_layer('paddle.multiply',
                                    inputs=inputs_dict,
                                    outputs=[node.name])

    @print_mapping_info
    def GatherND(self, node):
        x = self.graph.get_input_node(node, idx=0, copy=True)
        index = self.graph.get_input_node(node, idx=1, copy=True)
        inputs = {'x': x.name, 'index': index.name}
        self.paddle_graph.add_layer("paddle.gather_nd",
                                    inputs=inputs,
                                    outputs=[node.name])

    @print_mapping_info
    def Gather(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        indices = self.graph.get_input_node(node, idx=1, copy=True)
        indices_values = _const_weight_or_none(indices, necessary=True)
        if isinstance(indices_values, np.ndarray):
            indices_values = indices_values.tolist()
        indices_shape = indices.out_shapes[0]
        val_x_shape = val_x.out_shapes[0]
        axis = node.get_attr('axis', 0)
        if len(indices_shape) == 1 or \
            (indices_values is not None and isinstance(indices_values, int)) or \
                (indices_values is not None and len(indices_values) == 1):
            self.paddle_graph.add_layer('paddle.gather',
                                        inputs={
                                            'x': val_x.name,
                                            'index': indices.name
                                        },
                                        outputs=[node.name],
                                        axis=axis)
            # deal with indice is scalar(0D) Tensor
            if isinstance(indices_values, int) and len(val_x_shape) != 1:
                self.paddle_graph.add_layer('paddle.squeeze',
                                            inputs={'x': node.name},
                                            outputs=[node.name],
                                            axis=[axis])
        else:
            # if val_x is DataNode, convert gather to embedding
            if axis == 0 and isinstance(val_x, ONNXGraphDataNode):
                indices_cast = indices.name + '_cast'
                self.paddle_graph.add_layer('paddle.cast',
                                            inputs={"x": indices.name},
                                            outputs=[indices_cast],
                                            dtype=string('int64'))
                op_name = name_generator("embedding", self.nn_name2id)
                output_name = node.name
                layer_outputs = [op_name, output_name]
                self.weights[op_name + '.weight'] = _const_weight_or_none(val_x)
                self.paddle_graph.add_layer('paddle.nn.Embedding',
                                            inputs={"x": indices_cast},
                                            outputs=layer_outputs,
                                            num_embeddings=val_x_shape[0],
                                            embedding_dim=val_x_shape[1])
            else:
                self.paddle_graph.add_layer('paddle.reshape',
                                            inputs={"x": indices.name},
                                            outputs=[indices.name + "_reshape"],
                                            shape=[-1])
                gather_1d = node.name + '_1D'
                self.paddle_graph.add_layer('paddle.gather',
                                            inputs={
                                                'x': val_x.name,
                                                'index':
                                                indices.name + "_reshape"
                                            },
                                            outputs=[gather_1d],
                                            axis=axis)
                # if shape is known
                if len(indices_shape) != 0 and len(val_x_shape) != 0:
                    self.paddle_graph.add_layer('paddle.reshape',
                                                inputs={'x': gather_1d},
                                                outputs=[node.name],
                                                shape=val_x_shape[:axis] +
                                                indices_shape +
                                                val_x_shape[axis + 1:])
                else:
                    all_shape_name = list()
                    self.paddle_graph.add_layer(kernel="paddle.shape",
                                                inputs={"input": val_x.name},
                                                outputs=[val_x.name + "_shape"])
                    self.paddle_graph.add_layer(
                        kernel="paddle.shape",
                        inputs={"input": indices.name},
                        outputs=[indices.name + "_shape"])
                    self.paddle_graph.add_layer(
                        "paddle.slice",
                        inputs={"input": val_x.name + "_shape"},
                        outputs=[val_x.name + "_shape_slice_start"],
                        axes=[0],
                        starts=[0],
                        ends=[axis])
                    all_shape_name.append(val_x.name + "_shape_slice_start")
                    all_shape_name.append(indices.name + "_shape")
                    self.paddle_graph.add_layer(
                        "paddle.slice",
                        inputs={"input": val_x.name + "_shape"},
                        outputs=[val_x.name + "_shape_slice_end"],
                        axes=[0],
                        starts=[axis + 1],
                        ends=[2147483647])
                    all_shape_name.append(val_x.name + "_shape_slice_end")
                    self.paddle_graph.add_layer(
                        'paddle.concat',
                        inputs={"x": all_shape_name},
                        outputs=[node.name + "_all_shape"],
                        axis=0)
                    self.paddle_graph.add_layer('paddle.reshape',
                                                inputs={'x': gather_1d},
                                                outputs=[node.name],
                                                shape=node.name + "_all_shape")

    @print_mapping_info
    def ScatterND(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        indices = self.graph.get_input_node(node, idx=1, copy=True)
        updates = self.graph.get_input_node(node, idx=2, copy=True)
        if len(indices.out_shapes[0]) == 1:
            self.paddle_graph.add_layer('paddle.scatter',
                                        inputs={
                                            'x': val_x.name,
                                            'index': indices.name,
                                            'updates': updates.name
                                        },
                                        outputs=[node.name])
        else:
            input_inner_indices = node.name + '_input_inner_indices'
            shape = val_x.out_shapes[0]
            self.paddle_graph.add_layer('paddle.reshape',
                                        inputs={"x": indices.name},
                                        outputs=[indices.name],
                                        shape=indices.out_shapes[0])

            zeros_like_val_x = val_x.name + '_zeros'
            self.paddle_graph.add_layer('paddle.zeros_like',
                                        inputs={"x": val_x.name},
                                        outputs=[zeros_like_val_x])
            self.paddle_graph.add_layer('paddle.scatter_nd_add',
                                        inputs={
                                            'x': zeros_like_val_x,
                                            'index': indices.name,
                                            'updates': updates.name
                                        },
                                        outputs=[input_inner_indices])
            indices_mask = node.name + '_indices_mask'
            constant_minus_one = node.name + '_constant_minus_one'
            # full_like support create tensor shape like input tensor
            self.paddle_graph.add_layer('paddle.full_like',
                                        inputs={"x": updates.name},
                                        outputs=[constant_minus_one],
                                        dtype=string(updates.dtype),
                                        fill_value=-1)
            self.paddle_graph.add_layer('paddle.scatter_nd_add',
                                        inputs={
                                            'x': zeros_like_val_x,
                                            'index': indices.name,
                                            'updates': constant_minus_one
                                        },
                                        outputs=[indices_mask])
            constant_one = node.name + '_constant_1'
            # full_like support create tensor shape like input tensor
            self.paddle_graph.add_layer('paddle.full_like',
                                        inputs={"x": val_x.name},
                                        outputs=[constant_one],
                                        dtype=string(val_x.dtype),
                                        fill_value=1)
            input_out_indices_mask = node.name + '_input_out_indices_mask'
            self.paddle_graph.add_layer("paddle.add",
                                        inputs={
                                            "x": indices_mask,
                                            "y": constant_one
                                        },
                                        outputs=[input_out_indices_mask])

            input_out_indices = node.name + '_input_out_indices'
            self.paddle_graph.add_layer("paddle.multiply",
                                        inputs={
                                            "x": val_x.name,
                                            "y": input_out_indices_mask
                                        },
                                        outputs=[input_out_indices])

            self.paddle_graph.add_layer("paddle.add",
                                        inputs={
                                            "x": input_inner_indices,
                                            "y": input_out_indices
                                        },
                                        outputs=[node.name])

    @print_mapping_info
    def Range(self, node):
        val_start = self.graph.get_input_node(node, idx=0, copy=True)
        val_limit = self.graph.get_input_node(node, idx=1, copy=True)
        val_delta = self.graph.get_input_node(node, idx=2, copy=True)
        dtype = val_start.dtype
        inputs = {
            'start': val_start.name,
            'end': val_limit.name,
            'step': val_delta.name
        }
        self.paddle_graph.add_layer('paddle.arange',
                                    inputs=inputs,
                                    outputs=[node.name],
                                    dtype=string(dtype))

    @print_mapping_info
    def Slice(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        starts, ends, axes, steps = None, None, None, None
        layer_attrs = {}
        if val_x.dtype == 'uint8':
            self.paddle_graph.add_layer('paddle.cast',
                                        inputs={"x": val_x.name},
                                        outputs=[val_x.name],
                                        dtype=string('int32'))
        if len(node.inputs) > 1:
            starts = self.graph.get_input_node(node, idx=1, copy=True)
            ends = self.graph.get_input_node(node, idx=2, copy=True)
            starts_value = _const_weight_or_none(starts)
            if starts_value is not None:
                starts_value = starts_value.tolist()
            ends_value = _const_weight_or_none(ends)
            if ends_value is not None:
                ends_value = ends_value.tolist()
            if len(node.inputs) > 2:
                s_len = len(val_x.out_shapes[0])
                axes = list(range(s_len))
            if len(node.inputs) > 3:
                axes_node = self.graph.get_input_node(node, idx=3, copy=True)
                axes = _const_weight_or_none(axes_node, necessary=True).tolist()
            if len(node.inputs) > 4:
                steps = self.graph.get_input_node(node, idx=4, copy=True)
                steps = _const_weight_or_none(steps).tolist()

            layer_attrs = {
                "axes": axes,
                "starts": starts.name,
                "ends": ends.name
            }
            if starts_value is not None and ends_value is not None and axes is not None:
                starts_value = starts_value.copy()
                ends_value = ends_value.copy()
                for idx in range(len(ends_value)):
                    if len(
                            val_x.out_shapes[0]
                    ) != 0 and starts_value[idx] >= val_x.out_shapes[0][
                            axes[idx]] and val_x.out_shapes[0][axes[idx]] > 0:
                        starts_value[idx] = val_x.out_shapes[0][axes[idx]] - 1
                        ends_value[idx] = val_x.out_shapes[0][axes[idx]]
                    elif ends_value[idx] > 2**31 - 1:
                        ends_value[idx] = 2**31 - 1
                    elif ends_value[idx] < -2**31:
                        ends_value[idx] = -2**31
                # If stride is -1 and starts and ends meet the conditions, just reverse it directly
                if steps == [-1] and len(starts_value) == 1 and len(
                        ends_value
                ) == 1 and starts_value[0] == -1 and ends_value[0] == -2**31:
                    self.paddle_graph.add_layer("paddle.flip",
                                                inputs={"x": val_x.name},
                                                outputs=[node.name],
                                                axis=axes)
                    return
                layer_attrs = {
                    "axes": axes,
                    "starts": starts_value,
                    "ends": ends_value
                }
            else:
                if starts.dtype != 'int32':
                    starts_cast = starts.name + '_cast'
                    self.paddle_graph.add_layer('paddle.cast',
                                                inputs={"x": starts.name},
                                                outputs=[starts_cast],
                                                dtype=string('int32'))
                    layer_attrs['starts'] = starts_cast
                if ends.dtype != 'int32':
                    ends_cast = ends.name + '_cast'
                else:
                    ends_cast = ends.name
                self.paddle_graph.add_layer('paddle.cast',
                                            inputs={"x": ends.name},
                                            outputs=[ends_cast],
                                            dtype=string('int32'))
                layer_attrs['ends'] = ends_cast
        else:
            starts = node.get_attr('starts')
            ends = node.get_attr('ends')
            axes = node.get_attr('axes')
            output_shape = val_x.out_shapes[0]

            if axes is None:
                axes = [i for i in range(len(starts))]
            for idx in range(len(ends)):
                if ends[idx] > 2**31 - 1:
                    ends[idx] = 2**31 - 1
                elif ends[idx] < -2**31:
                    ends[idx] = 0
            layer_attrs = {"axes": axes, "starts": starts, "ends": ends}

        if steps is not None:
            layer_attrs['strides'] = steps
            self.paddle_graph.add_layer('paddle.strided_slice',
                                        inputs={"x": val_x.name},
                                        outputs=[node.name],
                                        **layer_attrs)
        else:
            self.paddle_graph.add_layer('paddle.slice',
                                        inputs={"input": val_x.name},
                                        outputs=[node.name],
                                        **layer_attrs)
        if val_x.dtype == 'uint8':
            self.paddle_graph.add_layer('paddle.cast',
                                        inputs={"x": node.name},
                                        outputs=[node.name],
                                        dtype=string('uint8'))

    @print_mapping_info
    def ConstantOfShape(self, node):
        val_shape = self.graph.get_input_node(node, idx=0, copy=True)

        value = node.get_attr('value')
        dtype = value.dtype
        value = value.tolist()
        assert len(value) == 1, ('given value not Scalar, shape of value > 1, '
                                 'this is not supported')
        if len(value) == 1:
            value = value[0]
            if value == float('inf') or value == float('-inf'):
                value = string(value)
            layer_attrs = {'dtype': string(dtype), 'fill_value': value}
            self.paddle_graph.add_layer("paddle.full",
                                        inputs={'shape': val_shape.name},
                                        outputs=[node.name],
                                        **layer_attrs)

    @print_mapping_info
    def Clip(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_node(node.layer.output[0], copy=True)
        max_value, min_value = None, None
        if len(node.inputs) == 1:
            max_value = node.get_attr('max')
            min_value = node.get_attr('min')
            layer_attrs = {
                'max': max_value,
                'min': min_value,
            }

            self.paddle_graph.add_layer('paddle.clip',
                                        inputs={"x": val_x.name},
                                        outputs=[node.name],
                                        **layer_attrs)
        else:
            if len(node.inputs) == 2:
                val_ipt = self.graph.get_input_node(node, idx=1, copy=True)

                index = node.get_input_index(val_ipt.name)

                val_value = _const_weight_or_none(val_ipt)
                if val_value.shape == (1, ):
                    val_value = val_value[0]

                if index == 1:
                    layer_attrs = {'min': val_value}

                if index == 2:
                    layer_attrs = {'max': val_value}

                self.paddle_graph.add_layer('paddle.clip',
                                            inputs={"x": val_x.name},
                                            outputs=[node.name],
                                            **layer_attrs)
            else:
                if len(node.inputs) == 3:
                    min_ipt = self.graph.get_input_node(node, idx=1, copy=True)
                    max_ipt = self.graph.get_input_node(node, idx=2, copy=True)
                    self.paddle_graph.add_layer('paddle.clip',
                                                inputs={
                                                    "x": val_x.name,
                                                    "min": min_ipt.name,
                                                    "max": max_ipt.name
                                                },
                                                outputs=[node.name])
                else:
                    raise Exception("max_value or min_value can't be None")

    @print_mapping_info
    def ReduceSum(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        if len(node.inputs) == 1:
            keepdims = node.get_attr('keepdims')
            if keepdims is None:
                keepdims = True
            axes_value = node.get_attr('axes')
            layer_attrs = {'axis': axes_value, 'keepdim': keepdims}
            self.paddle_graph.add_layer('paddle.sum',
                                        inputs={"x": val_x.name},
                                        outputs=[node.name],
                                        **layer_attrs)
        else:
            axes = self.graph.get_input_node(node, idx=1, copy=True)
            axes_value = _const_weight_or_none(axes)
            if axes_value.shape == (1, ):
                axes_value = axes_value[0]
            keepdims = node.get_attr('keepdims')
            if keepdims is None:
                layer_attrs = {'axis': axes_value}
            else:
                layer_attrs = {'axis': axes_value, 'keepdim': keepdims}

            self.paddle_graph.add_layer('paddle.sum',
                                        inputs={"x": val_x.name},
                                        outputs=[node.name],
                                        **layer_attrs)

    @print_mapping_info
    def Max(self, node):
        if len(node.inputs) == 2:
            val_x = self.graph.get_input_node(node, idx=0, copy=True)
            val_y = self.graph.get_input_node(node, idx=1, copy=True)
            self.paddle_graph.add_layer("paddle.maximum",
                                        inputs={
                                            "x": val_x.name,
                                            "y": val_y.name
                                        },
                                        outputs=[node.name])
        else:
            val_x = self.graph.get_input_node(node, idx=0, copy=True)
            temp_name = "max_"
            for i in range(1, len(node.inputs)):
                val_y = self.graph.get_input_node(node, idx=i, copy=True)
                temp_name = temp_name + str(i)
                if i == len(node.inputs) - 1:
                    self.paddle_graph.add_layer("paddle.maximum",
                                                inputs={
                                                    "x": val_x.name,
                                                    "y": val_y.name
                                                },
                                                outputs=[node.name])
                else:
                    self.paddle_graph.add_layer("paddle.maximum",
                                                inputs={
                                                    "x": val_x.name,
                                                    "y": val_y.name
                                                },
                                                outputs=[temp_name])
                val_x.name = temp_name

    @print_mapping_info
    def Min(self, node):
        if len(node.inputs) == 2:
            val_x = self.graph.get_input_node(node, idx=0, copy=True)
            val_y = self.graph.get_input_node(node, idx=1, copy=True)
            self.paddle_graph.add_layer("paddle.minimum",
                                        inputs={
                                            "x": val_x.name,
                                            "y": val_y.name
                                        },
                                        outputs=[node.name])
        else:
            val_x = self.graph.get_input_node(node, idx=0, copy=True)
            temp_name = "min_"
            for i in range(1, len(node.inputs)):
                val_y = self.graph.get_input_node(node, idx=i, copy=True)
                temp_name = temp_name + str(i)
                if i == len(node.inputs) - 1:
                    self.paddle_graph.add_layer("paddle.minimum",
                                                inputs={
                                                    "x": val_x.name,
                                                    "y": val_y.name
                                                },
                                                outputs=[node.name])
                else:
                    self.paddle_graph.add_layer("paddle.minimum",
                                                inputs={
                                                    "x": val_x.name,
                                                    "y": val_y.name
                                                },
                                                outputs=[temp_name])
                val_x.name = temp_name

    @print_mapping_info
    def GreaterOrEqual(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_input_node(node, idx=1, copy=True)
        self.paddle_graph.add_layer("paddle.greater_equal",
                                    inputs={
                                        "x": val_x.name,
                                        "y": val_y.name
                                    },
                                    outputs=[node.name])

    @print_mapping_info
    def And(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_input_node(node, idx=1, copy=True)
        self.paddle_graph.add_layer("paddle.logical_and",
                                    inputs={
                                        "x": val_x.name,
                                        "y": val_y.name
                                    },
                                    outputs=[node.name])

    @print_mapping_info
    def Split(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        split = node.get_attr('split')
        axis = node.get_attr('axis', 0)
        if split is None:
            split_num = len(node.layer.output)
            try:
                # split is an input of this node
                split_node = self.graph.get_input_node(node, idx=1, copy=True)
                split_value = _const_weight_or_none(split_node)
                layer_attrs = {
                    'num_or_sections': split_value.tolist(),
                    'axis': axis,
                }
            except:
                layer_attrs = {
                    'num_or_sections': split_num,
                    'axis': axis,
                }
            outputs_list = list()
            for i in range(len(node.layer.output)):
                if hasattr(node, 'index'):
                    outputs_list.append("{}_p{}".format(node.layer_name, i))
                else:
                    outputs_list.append("{}".format(node.layer.output[i]))
            if split_num > 1:
                self.paddle_graph.add_layer('paddle.split',
                                            inputs={"x": val_x.name},
                                            outputs=outputs_list,
                                            **layer_attrs)
            else:
                self.paddle_graph.add_layer("paddle.cast",
                                            inputs={"x": val_x.name},
                                            outputs=outputs_list,
                                            dtype=string(val_x.dtype))

        else:
            layer_attrs = {
                'num_or_sections': split,
                'axis': axis,
            }
            outputs_list = list()
            if isinstance(split, list) or isinstance(split, tuple):
                if len(split) == 1:
                    outputs_list.append(node.name)
                else:
                    for i in range(len(split)):
                        outputs_list.append("{}_p{}".format(node.layer_name, i))
            else:
                outputs_list.append(node.name)
            if len(split) > 1:
                self.paddle_graph.add_layer('paddle.split',
                                            inputs={"x": val_x.name},
                                            outputs=outputs_list,
                                            **layer_attrs)
            else:
                self.paddle_graph.add_layer("paddle.cast",
                                            inputs={"x": val_x.name},
                                            outputs=outputs_list,
                                            dtype=string(val_x.dtype))

    @print_mapping_info
    def Reshape(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_shape = self.graph.get_input_node(node, idx=1, copy=True)
        val_reshaped = self.graph.get_node(node.layer.output[0], copy=True)
        shape_value = _const_weight_or_none(val_shape)
        shape_dims = len(val_shape.out_shapes[0])

        if shape_value is not None:
            self.paddle_graph.add_layer('paddle.reshape',
                                        inputs={'x': val_x.name},
                                        outputs=[node.name],
                                        shape=shape_value.tolist())
        elif len(node.out_shapes[0]) > 0 and _is_static_shape(
                node.out_shapes[0]):
            self.paddle_graph.add_layer('paddle.reshape',
                                        inputs={'x': val_x.name},
                                        outputs=[node.name],
                                        shape=node.out_shapes[0])
        else:
            # shape may be [], come form Gather by scalar indices
            if len(val_shape.out_shapes[0]) > 0:
                self.paddle_graph.add_layer('paddle.reshape',
                                            inputs={'x': val_shape.name},
                                            outputs=[val_shape.name],
                                            shape=val_shape.out_shapes[0])
            if val_shape.dtype != "int32":
                self.paddle_graph.add_layer('paddle.cast',
                                            inputs={'x': val_shape.name},
                                            outputs=[val_shape.name],
                                            dtype=string("int32"))
            self.paddle_graph.add_layer('paddle.reshape',
                                        inputs={
                                            'x': val_x.name,
                                            'shape': val_shape.name
                                        },
                                        outputs=[node.name])

    @print_mapping_info
    def Cast(self, node):
        val_input = self.graph.get_input_node(node, idx=0, copy=True)
        val_output = self.graph.get_node(node.layer.output[0], copy=True)

        dtype = node.get_attr('to')
        if not isinstance(dtype, np.dtype):
            dtype = TENSOR_TYPE_TO_NP_TYPE[dtype]

        output_dtype = val_output.dtype
        if output_dtype:
            assert dtype == output_dtype, 'dtype of to unmatches output'
        self.paddle_graph.add_layer('paddle.cast',
                                    inputs={'x': val_input.name},
                                    outputs=[node.name],
                                    dtype=string(dtype))

    @print_mapping_info
    def Not(self, node):
        val_input = self.graph.get_input_node(node, idx=0, copy=True)
        self.paddle_graph.add_layer('paddle.logical_not',
                                    inputs={'x': val_input.name},
                                    outputs=[node.name])

    @print_mapping_info
    def AveragePool(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)

        auto_pad = node.get_attr('auto_pad', 'NOTSET')
        kernel_shape = node.get_attr("kernel_shape")
        poolnd = len(kernel_shape)
        strides = node.get_attr("strides")
        pad_mode = node.get_attr("pads")
        ceil_mode = bool(node.get_attr('ceil_mode', 0))
        pads = node.get_attr('pads', [0] * (poolnd * 2))

        paddings, val_x = self._pad_if_asymmetric(node, pads, val_x)

        if auto_pad == "SAME_UPPER" or auto_pad == "SAME_LOWER":
            input_shape = val_x.out_shapes[0]
            pad_h = _get_same_padding(input_shape[2], kernel_shape[0],
                                      strides[0], auto_pad)
            pad_w = _get_same_padding(input_shape[3], kernel_shape[1],
                                      strides[1], auto_pad)
            paddings = pad_h + pad_w

        op_name = name_generator("pool", self.nn_name2id)
        output_name = node.name
        layer_outputs = [op_name, output_name]
        paddle_op = 'paddle.nn.AvgPool{}D'.format(poolnd)
        assert 1 <= poolnd <= 3, 'only Pool1D, Pool2D and Pool3D are supported'
        layer_attrs = {
            "kernel_size": kernel_shape,
            "stride": strides,
            "padding": paddings,
            "ceil_mode": ceil_mode,
            "exclusive": 'True',
        }
        self.paddle_graph.add_layer(
            paddle_op,
            inputs={'x': val_x if isinstance(val_x, str) else val_x.name},
            outputs=layer_outputs,
            **layer_attrs)

    @print_mapping_info
    def Concat(self, node):
        inputs_list = []
        dtypes = set()
        for i in range(len(node.layer.input)):
            ipt = self.graph.get_input_node(node, idx=i, copy=True)
            inputs_list.append(ipt.name)
            dtypes.add(ipt.dtype)
        if len(dtypes) > 1:
            assert 'Unspported situation happened, please create issue on https://github.com/PaddlePaddle/X2Paddle/issues.'
        axis = node.get_attr('axis')
        self.paddle_graph.add_layer('paddle.concat',
                                    inputs={"x": inputs_list},
                                    outputs=[node.name],
                                    axis=axis)

    @print_mapping_info
    def Flatten(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        output_shape = val_x.out_shapes[0]
        axis = node.get_attr('axis', 1)
        if axis == 0:
            self.paddle_graph.add_layer('paddle.reshape',
                                        inputs={"x": val_x.name},
                                        outputs=[node.name],
                                        shape=[1, -1])
        else:
            if len(output_shape) != 0:
                shape_list = [1, 1]
                for s in output_shape[:axis]:
                    shape_list[0] *= s
                for s in output_shape[axis:]:
                    shape_list[1] *= s
                self.paddle_graph.add_layer('paddle.reshape',
                                            inputs={"x": val_x.name},
                                            outputs=[node.name],
                                            shape=shape_list)
            else:
                # flatten + reshape
                self.paddle_graph.add_layer("paddle.flatten",
                                            inputs={"x": val_x.name},
                                            outputs=[val_x.name + "_flatten"],
                                            start_axis=0,
                                            stop_axis=-1)
                self.paddle_graph.add_layer(
                    'paddle.reshape',
                    inputs={'x': val_x.name + "_flatten"},
                    outputs=[node.name],
                    shape=[0, -1])

    @print_mapping_info
    def Gemm(self, node):
        val_a = self.graph.get_input_node(node, idx=0, copy=True)
        val_b = self.graph.get_input_node(node, idx=1, copy=True)
        val_c = self.graph.get_input_node(node, idx=2, copy=True)

        alpha = node.get_attr('alpha', 1.)  # optional
        beta = node.get_attr('beta', 1.)  # optional
        trans_a = bool(node.get_attr('transA', 0))  # optional
        trans_b = bool(node.get_attr('transB', 0))  # optional
        val_mm = node.name + '_mm'
        matmul_inputs = {"x": val_a.name, "y": val_b.name}
        attr_matmul = {
            "transpose_x": trans_a,
            "transpose_y": trans_b,
        }
        if abs(alpha - 1.0) < 1e-5:
            if abs(beta - 0.0) < 1e-5:
                self.paddle_graph.add_layer('paddle.matmul',
                                            inputs=matmul_inputs,
                                            outputs=[node.name],
                                            **attr_matmul)
            else:
                self.paddle_graph.add_layer('paddle.matmul',
                                            inputs=matmul_inputs,
                                            outputs=[val_mm],
                                            **attr_matmul)
                if abs(beta - 1.0) < 1e-5:
                    add_inputs = {"x": val_mm, "y": val_c.name}
                    self.paddle_graph.add_layer("paddle.add",
                                                inputs=add_inputs,
                                                outputs=[node.name])
                else:
                    var_beta = node.name + '_beta'
                    self.paddle_graph.add_layer("paddle.scale",
                                                inputs={"x": val_c.name},
                                                outputs=[var_beta],
                                                scale=beta)
                    add_inputs = {"x": val_mm, "y": var_beta}
                    self.paddle_graph.add_layer("paddle.add",
                                                inputs=add_inputs,
                                                outputs=[node.name])
        else:
            if abs(beta - 0.0) < 1e-5:
                self.paddle_graph.add_layer('paddle.matmul',
                                            inputs=matmul_inputs,
                                            outputs=[val_mm],
                                            **attr_matmul)
                self.paddle_graph.add_layer("paddle.scale",
                                            inputs={"x": val_mm},
                                            outputs=[node.name],
                                            scale=alpha)
            else:
                self.paddle_graph.add_layer('paddle.matmul',
                                            inputs=[matmul_inputs],
                                            outputs=[val_mm],
                                            **attr_matmul)
                self.paddle_graph.add_layer("paddle.scale",
                                            inputs={"x": val_mm},
                                            outputs=[val_mm],
                                            scale=alpha)
                if abs(beta - 1.0) < 1e-5:
                    add_inputs = {"x": val_mm, "y": val_c.name}
                    self.paddle_graph.add_layer("paddle.add",
                                                inputs=add_inputs,
                                                outputs=[node.name])
                else:
                    var_beta = node.name + '_beta'
                    self.paddle_graph.add_layer("paddle.scale",
                                                inputs={"x": val_c.name},
                                                outputs=[var_beta],
                                                scale=beta)
                    add_inputs = {"x": val_mm, "y": var_beta}
                    self.paddle_graph.add_layer("paddle.add",
                                                inputs=add_inputs,
                                                outputs=[node.name])

    @print_mapping_info
    def Sum(self, node):
        val_inps = node.layer.input
        inputs_dict = {
            "x": self.graph.get_input_node(node, idx=0, copy=True).name,
            "y": self.graph.get_input_node(node, idx=1, copy=True).name,
        }
        self.paddle_graph.add_layer("paddle.add",
                                    inputs=inputs_dict,
                                    outputs=[node.name])

        for idx, ipt in enumerate(val_inps[2:]):
            y = self.graph.get_input_node(node, idx=idx, copy=True)
            inputs_dict = {
                "x": node.name,
                "y": y.name,
            }
            self.paddle_graph.add_layer("paddle.add",
                                        inputs=inputs_dict,
                                        outputs=[node.name])

    @print_mapping_info
    def MatMul(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_input_node(node, idx=1, copy=True)
        x_shape = val_x.out_shapes[0]
        y_shape = val_y.out_shapes[0]
        inputs_dict = {"x": val_x.name, "y": val_y.name}
        if len(y_shape) != 0 and y_shape[0] == 1 and len(
                x_shape) != 0 and x_shape[-1] != 1 and x_shape[0] != 1:
            y_squeeze = val_y.name + '_squeeze'
            self.paddle_graph.add_layer("paddle.squeeze",
                                        inputs={"x": val_y.name},
                                        outputs=[y_squeeze],
                                        axis=[0])
            inputs_dict['y'] = y_squeeze
            self.paddle_graph.add_layer("paddle.matmul",
                                        inputs=inputs_dict,
                                        outputs=[node.name])
        else:
            self.paddle_graph.add_layer("paddle.matmul",
                                        inputs=inputs_dict,
                                        outputs=[node.name])

    @print_mapping_info
    def BatchNormalization(self, node):
        op_name = name_generator("batchnorm", self.nn_name2id)
        output_name = node.name
        layer_outputs = [op_name, output_name]
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_scale = self.graph.get_input_node(node, idx=1, copy=True)
        val_b = self.graph.get_input_node(node, idx=2, copy=True)
        val_mean = self.graph.get_input_node(node, idx=3, copy=True)
        val_var = self.graph.get_input_node(node, idx=4, copy=True)

        momentum = node.get_attr('momentum', .9)
        epsilon = node.get_attr('epsilon', 1e-5)
        c = val_x.out_shapes[0][1]

        # solved the same data is used as an argument to multiple OPs.
        _rename_or_remove_weight(self.weights,
                                 val_scale.name,
                                 op_name + '.weight',
                                 rename_mapper=self.rename_mapper)
        _rename_or_remove_weight(self.weights,
                                 val_b.name,
                                 op_name + '.bias',
                                 rename_mapper=self.rename_mapper)
        _rename_or_remove_weight(self.weights,
                                 val_var.name,
                                 op_name + '._variance',
                                 rename_mapper=self.rename_mapper)
        _rename_or_remove_weight(self.weights,
                                 val_mean.name,
                                 op_name + '._mean',
                                 rename_mapper=self.rename_mapper)

        # Attribute: spatial is used in BatchNormalization-1,6,7
        spatial = bool(node.get_attr('spatial'))
        layer_attrs = {
            "num_channels": c,
            "momentum": momentum,
            "epsilon": epsilon,
            "is_test": True,
            "use_global_stats": False,
        }
        self.paddle_graph.add_layer("paddle.nn.BatchNorm",
                                    inputs={"x": val_x.name},
                                    outputs=layer_outputs,
                                    **layer_attrs)

    @print_mapping_info
    def Transpose(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        s_len = len(val_x.out_shapes[0])
        perm_default = list(range(s_len))
        perm_default.reverse()
        perm = node.get_attr('perm', perm_default)
        self.paddle_graph.add_layer("paddle.transpose",
                                    inputs={"x": val_x.name},
                                    outputs=[node.name],
                                    perm=perm)

    @print_mapping_info
    def PRelu(self, node):
        op_name = name_generator("prelu", self.nn_name2id)
        output_name = node.name
        layer_outputs = [op_name, output_name]
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_slope = self.graph.get_input_node(node, idx=1, copy=True)

        mode = 'channel'
        shape_slope = val_slope.out_shapes[0]
        if shape_slope == [1] * len(shape_slope):
            mode = 'all'

        if mode == "element":
            self.paddle_graph.add_layer("paddle.zeros",
                                        inputs={},
                                        outputs=[output_name + "__zeros"],
                                        shape=shape_slope,
                                        dtype=string(node.dtype))
            self.paddle_graph.add_layer("paddle.maximum",
                                        inputs={
                                            "x": val_x.name,
                                            "y": output_name + "__zeros"
                                        },
                                        outputs=[output_name + "__max"])
            self.paddle_graph.add_layer("paddle.minimum",
                                        inputs={
                                            "x": val_x.name,
                                            "y": output_name + "__zeros"
                                        },
                                        outputs=[output_name + "__min"])
            self.paddle_graph.add_layer("paddle.multiply",
                                        inputs={
                                            "x": val_slope.name,
                                            "y": output_name + "__min"
                                        },
                                        outputs=[output_name + "__mul"])
            self.paddle_graph.add_layer("paddle.add",
                                        inputs={
                                            "x": output_name + "__max",
                                            "y": output_name + "__mul"
                                        },
                                        outputs=[output_name])
        else:
            if mode == 'channel':
                slope_data = _const_weight_or_none(val_slope)
                if slope_data is None:
                    self.paddle_graph.add_layer("paddle.reshape",
                                                inputs={"x": val_slope.name},
                                                outputs=[val_slope.name],
                                                shape=[shape_slope[0]])
                    self.paddle_graph.add_layer("paddle.nn.functional.prelu",
                                                inputs={
                                                    "x": val_x.name,
                                                    "weight": val_slope.name
                                                },
                                                outputs=[node.name])
                    return
                _rename_or_remove_weight(self.weights, val_slope.name)
                if len(shape_slope) > 1:
                    self.weights[op_name + '._weight'] = np.reshape(
                        slope_data, shape_slope[0])
                num_parameters = val_x.out_shapes[0][1]
            else:
                num_parameters = 1
                slope_data = self.weights[val_slope.name]
                _rename_or_remove_weight(self.weights, val_slope.name)
                self.weights[op_name + '._weight'] = np.reshape(slope_data, [1])
            self.paddle_graph.add_layer("paddle.nn.PReLU",
                                        inputs={"x": val_x.name},
                                        outputs=layer_outputs,
                                        num_parameters=num_parameters)

    @print_mapping_info
    def Squeeze(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        axes = node.get_attr('axes')
        if axes is None:
            axes_node = self.graph.get_input_node(node, idx=1, copy=True)
            axes = _const_weight_or_none(axes_node, necessary=True)
        # deal with scalar(0D) tensor
        if len(val_x.out_shapes[0]) <= 1 and len(axes) == 1 and axes[0] == 0:
            self.paddle_graph.add_layer("paddle.cast",
                                        inputs={"x": val_x.name},
                                        outputs=[node.name],
                                        dtype=string(val_x.dtype))
        else:
            self.paddle_graph.add_layer("paddle.squeeze",
                                        inputs={"x": val_x.name},
                                        outputs=[node.name],
                                        axis=axes)

    @print_mapping_info
    def Equal(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_input_node(node, idx=1, copy=True)
        self.paddle_graph.add_layer("paddle.equal",
                                    inputs={
                                        'x': val_x.name,
                                        'y': val_y.name
                                    },
                                    outputs=[node.name])

    @print_mapping_info
    def Greater(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_input_node(node, idx=1, copy=True)
        self.paddle_graph.add_layer("paddle.greater_than",
                                    inputs={
                                        'x': val_x.name,
                                        'y': val_y.name
                                    },
                                    outputs=[node.name])

    @print_mapping_info
    def Where(self, node):
        condition = self.graph.get_input_node(node, idx=0, copy=True)
        val_x = self.graph.get_input_node(node, idx=1, copy=True)
        val_y = self.graph.get_input_node(node, idx=2, copy=True)

        self.paddle_graph.add_layer("paddle.where",
                                    inputs={
                                        'condition': condition.name,
                                        'x': val_x.name,
                                        'y': val_y.name
                                    },
                                    outputs=[node.name])

    @print_mapping_info
    def NonZero(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        self.paddle_graph.add_layer("paddle.nonzero",
                                    inputs={"x": val_x.name},
                                    outputs=[val_x.name],
                                    as_tuple=False)
        self.paddle_graph.add_layer('paddle.transpose',
                                    inputs={"x": val_x.name},
                                    outputs=[node.name],
                                    perm=[1, 0])

    @print_mapping_info
    def Identity(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        self.paddle_graph.add_layer("paddle.assign",
                                    inputs={"x": val_x.name},
                                    outputs=[node.name])

    @print_mapping_info
    def Tile(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_repeats = self.graph.get_input_node(node, idx=1, copy=True)
        repeats = _const_weight_or_none(val_repeats)

        if repeats is None:
            repeats = val_repeats.name
            if val_repeats.dtype != 'int32':
                self.paddle_graph.add_layer("paddle.cast",
                                            inputs={"x": repeats},
                                            outputs=["{}_tmp".format(repeats)],
                                            dtype=string("int32"))
                repeats = "{}_tmp".format(repeats)

        elif isinstance(repeats, int):
            repeats = [repeats]

        elif type(repeats) is np.ndarray:
            repeats = repeats.tolist()

        attr = {
            'expand_times': repeats,
            "name": string(node.name),
        }
        self.paddle_graph.add_layer("paddle.tile",
                                    inputs={"x": val_x.name},
                                    outputs=[node.name],
                                    repeat_times=repeats)

    @print_mapping_info
    def MaxPool(self, node):
        op_name = name_generator("pool", self.nn_name2id)
        output_name = node.name
        layer_outputs = [op_name, output_name]
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        auto_pad = node.get_attr('auto_pad', 'NOTSET')
        assert node.get_attr(
            "dilations") is None, 'only dilations = 0 is supported'  # optional

        kernel_shape = node.get_attr("kernel_shape")
        poolnd = len(kernel_shape)
        strides = node.get_attr("strides")
        pad_mode = node.get_attr("pads")
        ceil_mode = bool(node.get_attr('ceil_mode', 0))  # optional
        pads = node.get_attr('pads', [0] * (poolnd * 2))  # optional
        paddle_op = 'paddle.nn.MaxPool{}D'.format(poolnd)
        assert 1 <= poolnd <= 3, 'only Pool1D, Pool2D and Pool3D are supported'

        paddings, val_x = self._pad_if_asymmetric(node, pads, val_x)

        if auto_pad == "SAME_UPPER" or auto_pad == "SAME_LOWER":
            input_shape = val_x.out_shapes[0]
            pad_h = _get_same_padding(input_shape[2], kernel_shape[0],
                                      strides[0], auto_pad)
            pad_w = _get_same_padding(input_shape[3], kernel_shape[1],
                                      strides[1], auto_pad)
            paddings = pad_h + pad_w

        layer_attrs = {
            "kernel_size": kernel_shape,
            "stride": strides,
            "padding": paddings,
            "ceil_mode": ceil_mode,
        }
        self.paddle_graph.add_layer(
            paddle_op,
            inputs={'x': val_x if isinstance(val_x, str) else val_x.name},
            outputs=layer_outputs,
            **layer_attrs)

    @print_mapping_info
    def GlobalMaxPool(self, node):
        op_name = name_generator("pool", self.nn_name2id)
        output_name = node.name
        layer_outputs = [op_name, output_name]
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        input_shape = val_x.out_shapes[0]
        if len(input_shape) == 4:
            poolnd = 2
        elif len(input_shape) == 5:
            poolnd = 3
        elif len(input_shape) == 3:
            poolnd = 1
        paddle_op = 'paddle.nn.AdaptiveMaxPool{}D'.format(poolnd)
        assert 1 <= poolnd <= 3, 'only Pool1D, Pool2D and Pool3D are supported'
        output_shape = node.out_shapes[0]
        self.paddle_graph.add_layer(paddle_op,
                                    inputs={'x': val_x.name},
                                    outputs=layer_outputs,
                                    output_size=output_shape[2:])

    @print_mapping_info
    def Neg(self, node):
        import paddle
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        v0, v1, v2 = paddle.__version__.split('.')
        if int(v0) >= 2 and int(v1) >= 2:
            self.paddle_graph.add_layer("paddle.neg",
                                        inputs={'x': val_x.name},
                                        outputs=[node.name])
        else:
            val_y = node.name + "_y"
            dtype = np.dtype(val_x.dtype)
            self.paddle_graph.add_layer("paddle.full",
                                        inputs={},
                                        outputs=[val_y],
                                        dtype=string(dtype),
                                        shape=[1],
                                        fill_value=-1)
            self.paddle_graph.add_layer("paddle.multiply",
                                        inputs={
                                            'x': val_x.name,
                                            'y': val_y
                                        },
                                        outputs=[node.name])

    @print_mapping_info
    def SpaceToDepth(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        blocksize = node.get_attr('blocksize')
        val_x_shape = val_x.out_shapes[0]
        b, c, h, w = val_x_shape
        self.paddle_graph.add_layer(
            'paddle.reshape',
            inputs={"x": val_x.name},
            outputs=[node.name],
            shape=[b, c, h // blocksize, blocksize, w // blocksize, blocksize])
        self.paddle_graph.add_layer('paddle.transpose',
                                    inputs={"x": node.name},
                                    outputs=[node.name],
                                    perm=[0, 3, 5, 1, 2, 4])
        self.paddle_graph.add_layer(
            'paddle.reshape',
            inputs={"x": node.name},
            outputs=[node.name],
            shape=[b, c * (blocksize**2), h // blocksize, w // blocksize])

    @print_mapping_info
    def GatherElements(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        indices = self.graph.get_input_node(node, idx=1, copy=True)
        axis = node.get_attr('axis')
        val_x_shape = val_x.out_shapes[0]
        indices_shape = indices.out_shapes[0]
        axis = axis if axis >= 0 else axis + len(val_x_shape)
        if axis == 0:
            axis_perm = [i for i in range(len(val_x_shape))]
            data_swaped = val_x.name
            index_swaped = indices.name
        else:
            axis_perm = [i for i in range(len(val_x_shape))]
            axis_perm[axis] = 0
            axis_perm[0] = axis
            data_swaped = val_x.name + "_transpose"
            self.paddle_graph.add_layer("paddle.transpose",
                                        inputs={'x': val_x.name},
                                        perm=axis_perm,
                                        outputs=[data_swaped])
            index_swaped = indices.name + "_transpose"
            self.paddle_graph.add_layer("paddle.transpose",
                                        inputs={'x': indices.name},
                                        perm=axis_perm,
                                        outputs=[index_swaped])
            temp = indices_shape[0]
            indices_shape[0] = indices_shape[axis]
            indices_shape[axis] = temp

        idx_tensors_per_axis_pre = [
            indices_shape[i] for i in range(len(indices_shape))
        ]
        name_list = list()
        for i in range(len(idx_tensors_per_axis_pre)):
            tensor_name = val_x.name + "_meshgrid_" + str(i)
            self.paddle_graph.add_layer(kernel="paddle.linspace",
                                        inputs={},
                                        outputs=[tensor_name],
                                        start=0,
                                        stop=idx_tensors_per_axis_pre[i] - 1,
                                        num=idx_tensors_per_axis_pre[i])
            name_list.append(tensor_name)

        self.paddle_graph.add_layer("paddle.meshgrid",
                                    inputs=name_list,
                                    outputs=name_list)

        self.paddle_graph.add_layer("paddle.cast",
                                    inputs={"x": index_swaped},
                                    outputs=[index_swaped],
                                    dtype=string("float32"))
        import copy
        copy_name_list = copy.copy(name_list)
        copy_name_list[0] = index_swaped
        new_name_list = list()
        for i in range(len(copy_name_list)):
            unsqueeze_name = copy_name_list[i] + "_unsqueeze"
            self.paddle_graph.add_layer("paddle.unsqueeze",
                                        inputs={"x": copy_name_list[i]},
                                        axis=-1,
                                        outputs=[unsqueeze_name])
            new_name_list.append(unsqueeze_name)
        concat_name = val_x.name + "_concated_layer"
        self.paddle_graph.add_layer("paddle.concat",
                                    inputs={'x': new_name_list},
                                    axis=-1,
                                    outputs=[concat_name])
        self.paddle_graph.add_layer("paddle.cast",
                                    inputs={"x": concat_name},
                                    outputs=[concat_name],
                                    dtype=string("int32"))
        gather_nd_name = "gather_nd_layer"
        self.paddle_graph.add_layer("paddle.gather_nd",
                                    inputs={
                                        'x': data_swaped,
                                        "index": concat_name
                                    },
                                    outputs=[gather_nd_name])

        self.paddle_graph.add_layer("paddle.transpose",
                                    inputs={'x': gather_nd_name},
                                    perm=axis_perm,
                                    outputs=[node.name])

    @print_mapping_info
    def GlobalAveragePool(self, node):
        op_name = name_generator("pool", self.nn_name2id)
        output_name = node.name
        layer_outputs = [op_name, output_name]
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        input_shape = val_x.out_shapes[0]
        if len(input_shape) == 4:
            poolnd = 2
        elif len(input_shape) == 5:
            poolnd = 3
        elif len(input_shape) == 3:
            poolnd = 1
        paddle_op = 'paddle.nn.AdaptiveAvgPool{}D'.format(poolnd)
        assert 1 <= poolnd <= 3, 'only Pool1D, Pool2D and Pool3D are supported'
        output_shape = node.out_shapes[0]
        self.paddle_graph.add_layer(paddle_op,
                                    inputs={'x': val_x.name},
                                    outputs=layer_outputs,
                                    output_size=output_shape[2:])

    @print_mapping_info
    def Conv(self, node):
        output_name = node.name
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_w = self.graph.get_input_node(node, idx=1, copy=True)

        if val_w.name in self.weights.keys():
            op_name = name_generator("conv", self.nn_name2id)
        else:
            op_name = output_name

        layer_outputs = [op_name, output_name]

        has_bias = len(node.layer.input) == 3
        if has_bias:
            val_b = self.graph.get_input_node(node, idx=2, copy=True)
        auto_pad = node.get_attr('auto_pad', 'NOTSET')

        kernel_shape = node.get_attr('kernel_shape')
        convnd = len(kernel_shape)
        num_out_channels = val_w.out_shapes[0][0]
        num_in_channels = val_w.out_shapes[0][1]
        paddle_op = 'paddle.nn.Conv{}D'.format(convnd)

        num_groups = node.get_attr('group', 1)
        strides = node.get_attr('strides', [1] * convnd)
        dilations = node.get_attr('dilations', [1] * convnd)
        pads = node.get_attr('pads', [0] * (convnd * 2))

        input_shape = val_x.out_shapes[0]
        paddings = np.array(pads).reshape((2, -1)).transpose().astype("int32")
        paddings = paddings.flatten().tolist()

        if auto_pad in ["SAME_UPPER", "SAME_LOWER"]:
            # Warning: SAME_UPPER and SAME_LOWER does not yet support dynamic shapes
            if input_shape[2] == -1 or input_shape[3] == -1:
                _logger.warning(
                    'SAME_UPPER and SAME_LOWER does not yet support dynamic shapes, the conversion result may have a diff!!!'
                )
            pad_h = _get_same_padding(input_shape[2], kernel_shape[0],
                                      strides[0], auto_pad)
            pad_w = _get_same_padding(input_shape[3], kernel_shape[1],
                                      strides[1], auto_pad)
            paddings = pad_h + pad_w

        layer_inputs = {'x': val_x if isinstance(val_x, str) else val_x.name}
        if val_w.name not in self.weights.keys():
            layer_attrs = {
                "stride": strides,
                "padding": paddings,
                "dilation": dilations,
                "groups": num_groups,
            }
            layer_inputs['weight'] = val_w.name
            if has_bias:
                layer_inputs['bias'] = val_b.name

            paddle_op = 'paddle.nn.functional.conv{}d'.format(convnd)
            self.paddle_graph.add_layer(paddle_op,
                                        inputs=layer_inputs,
                                        outputs=[node.name],
                                        **layer_attrs)
            return

        layer_attrs = {
            "in_channels": num_in_channels * num_groups,
            "out_channels": num_out_channels,
            "kernel_size": kernel_shape,
            "stride": strides,
            "padding": paddings,
            "dilation": dilations,
            "groups": num_groups,
        }
        remove_weight = True if val_w.name in self.done_weight_list else False
        if remove_weight:
            self.done_weight_list.append(val_w.name)
        _rename_or_remove_weight(self.weights,
                                 val_w.name,
                                 op_name + '.weight',
                                 remove_weight,
                                 rename_mapper=self.rename_mapper)
        if has_bias:
            remove_bias = True if val_b.name in self.done_weight_list else False
            if remove_bias:
                self.done_weight_list.append(val_b.name)
            _rename_or_remove_weight(self.weights,
                                     val_b.name,
                                     op_name + '.bias',
                                     remove_bias,
                                     rename_mapper=self.rename_mapper)
        else:
            layer_attrs["bias_attr"] = False
        # deal with dynamic shape
        if len(input_shape) == 0:
            self.paddle_graph.add_layer(
                "paddle.reshape",
                inputs=layer_inputs,
                outputs=[layer_inputs["x"]],
                shape=[0, num_in_channels * num_groups, 0, -1])
        if len(input_shape) != 0 and reduce(
                lambda x, y: x * y, input_shape) in [1, -1
                                                     ] and 1 not in input_shape:
            input_shape[1] = num_in_channels * num_groups
            input_shape[0] = 0
            input_shape[2] = 0
            self.paddle_graph.add_layer("paddle.reshape",
                                        inputs=layer_inputs,
                                        outputs=[layer_inputs["x"]],
                                        shape=input_shape)
        self.paddle_graph.add_layer(paddle_op,
                                    inputs=layer_inputs,
                                    outputs=layer_outputs,
                                    **layer_attrs)

    @print_mapping_info
    def ConvTranspose(self, node):
        output_name = node.name
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_w = self.graph.get_input_node(node, idx=1, copy=True)

        if val_w.name in self.weights.keys():
            op_name = name_generator("conv_trans", self.nn_name2id)
        else:
            op_name = output_name

        layer_outputs = [op_name, output_name]

        val_b = None
        if len(node.layer.input) > 2:
            val_b = self.graph.get_input_node(node, idx=2, copy=True)
        auto_pad = node.get_attr('auto_pad', 'NOTSET')
        out_padding = node.get_attr('output_padding', [0, 0])
        kernel_shape = node.get_attr('kernel_shape')
        assert kernel_shape, 'kernel_shape not inferred'
        convnd = len(kernel_shape)
        assert 2 <= convnd <= 3, 'only Conv2DTranspose and Conv3DTranspose supported'
        num_in_channels = val_w.out_shapes[0][0]
        num_out_channels = val_w.out_shapes[0][1]
        paddle_op = 'paddle.nn.Conv{}DTranspose'.format(convnd)

        num_groups = node.get_attr('group', 1)
        strides = node.get_attr('strides', [1] * convnd)
        dilations = node.get_attr('dilations', [1] * convnd)
        output_size = node.get_attr('output_shape', [])
        pads = node.get_attr('pads', [0] * (convnd * 2))

        paddings = np.array(pads).reshape((2, -1)).transpose().astype("int32")
        paddings = paddings.flatten().tolist()

        if len(output_size) != 0:
            paddings = [0] * 4
            total_paddings = list()
            total_paddings.append((val_x.out_shapes[0][2] - 1) * strides[0] +
                                  dilations[0] * (kernel_shape[0] - 1) + 1 +
                                  out_padding[0] - output_size[0])
            total_paddings.append((val_x.out_shapes[0][3] - 1) * strides[1] +
                                  dilations[1] * (kernel_shape[1] - 1) + 1 +
                                  out_padding[1] - output_size[1])
            if auto_pad == "SAME_UPPER":
                for i in range(len(total_paddings)):
                    paddings[2 * i] = total_paddings[0] - \
                        total_paddings[0] // 2
                    paddings[2 * i + 1] = total_paddings[0] // 2
            else:
                for i in range(len(total_paddings)):
                    paddings[2 * i] = total_paddings[0] // 2
                    paddings[2 * i +
                             1] = total_paddings[0] - total_paddings[0] // 2
        else:
            output_size = [0, 0]

            output_size[0] = (val_x.out_shapes[0][2] - 1) * strides[
                0] - 2 * paddings[0] + dilations[0] * (kernel_shape[0] -
                                                       1) + 1 + out_padding[0]
            output_size[1] = (val_x.out_shapes[0][3] - 1) * strides[
                1] - 2 * paddings[1] + dilations[1] * (kernel_shape[1] -
                                                       1) + 1 + out_padding[1]

        # Conv2DTranspose缺少output_size，只能在forward里头传进output_size
        inputs_dict = {'x': val_x if isinstance(val_x, str) else val_x.name}
        if val_w.name not in self.weights.keys():
            layer_attrs = {
                "stride": strides,
                "dilation": dilations,
                "padding": paddings,
                "groups": num_groups,
                "output_padding": out_padding
            }
            paddle_op = 'paddle.nn.functional.conv{}d_transpose'.format(convnd)

            inputs_dict['weight'] = val_w.name
            if len(node.layer.input) > 2:
                inputs_dict['bias'] = val_b.name

            self.paddle_graph.add_layer(paddle_op,
                                        inputs=inputs_dict,
                                        outputs=[node.name],
                                        **layer_attrs)
            return

        layer_attrs = {
            "in_channels": num_in_channels,
            "out_channels": num_out_channels * num_groups,
            "kernel_size": kernel_shape,
            "stride": strides,
            "dilation": dilations,
            "padding": paddings,
            "groups": num_groups,
            "output_padding": out_padding
        }

        _rename_or_remove_weight(self.weights,
                                 val_w.name,
                                 op_name + '.weight',
                                 rename_mapper=self.rename_mapper)
        if val_b is not None:
            _rename_or_remove_weight(self.weights,
                                     val_b.name,
                                     op_name + '.bias',
                                     rename_mapper=self.rename_mapper)
        else:
            layer_attrs["bias_attr"] = False
        self.paddle_graph.add_layer(kernel=paddle_op,
                                    inputs=inputs_dict,
                                    outputs=layer_outputs,
                                    **layer_attrs)

    @print_mapping_info
    def ArgMax(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        axis = node.get_attr('axis')
        keepdims = False if node.get_attr('keepdims') == 0 else True
        layer_attrs = {'axis': axis, 'keepdim': keepdims}
        self.paddle_graph.add_layer('paddle.argmax',
                                    inputs={"x": val_x.name},
                                    outputs=[node.name],
                                    **layer_attrs)

    @print_mapping_info
    def Size(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        self.paddle_graph.add_layer("paddle.shape",
                                    inputs={"input": val_x.name},
                                    outputs=[node.name])
        self.paddle_graph.add_layer('paddle.cast',
                                    inputs={"x": node.name},
                                    outputs=[node.name],
                                    dtype=string('int64'))
        self.paddle_graph.add_layer("paddle.prod",
                                    inputs={"x": node.name},
                                    outputs=[node.name])

    @print_mapping_info
    def Sign(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        if node.dtype not in ["float16", "float32", "float64"]:
            self.paddle_graph.add_layer("paddle.cast",
                                        inputs={"x": val_x.name},
                                        outputs=[val_x.name],
                                        dtype=string("float32"))
        self.paddle_graph.add_layer("paddle.sign",
                                    inputs={"x": val_x.name},
                                    outputs=[node.name])
        if node.dtype not in ["float16", "float32", "float64"]:
            self.paddle_graph.add_layer("paddle.cast",
                                        inputs={"x": node.name},
                                        outputs=[node.name],
                                        dtype=string(node.dtype))

    @print_mapping_info
    def OneHot(self, node):
        nn_op_name = name_generator("onehot", self.nn_name2id)
        output_name = node.name
        layer_outputs = [nn_op_name, output_name]
        indices = self.graph.get_input_node(node, idx=0, copy=True)
        depth = self.graph.get_input_node(node, idx=1, copy=True)
        values = self.graph.get_input_node(node, idx=2, copy=True)
        axis = node.get_attr('axis', -1)
        self.paddle_graph.add_layer("custom_layer:OneHot",
                                    inputs={
                                        "indices": indices.name,
                                        "depth": depth.name,
                                        "values": values.name
                                    },
                                    outputs=layer_outputs,
                                    axis=axis)

    @print_mapping_info
    def Reciprocal(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        self.paddle_graph.add_layer("paddle.reciprocal",
                                    inputs={"x": val_x.name},
                                    outputs=[node.name])

    @print_mapping_info
    def LSTM(self, node):
        x = self.graph.get_input_node(node, idx=0, copy=True)
        input_weight = self.graph.get_input_node(node, idx=1, copy=True)
        hidden_weight = self.graph.get_input_node(node, idx=2, copy=True)

        input_nums = len(node.layer.input)
        exist_input_nums = 3
        have_bias = False
        if input_nums > 3 and node.layer.input[3] != '':
            bias = self.graph.get_input_node(node,
                                             idx=exist_input_nums,
                                             copy=True)
            have_bias = True
            exist_input_nums += 1
        if input_nums > 4 and node.layer.input[4] != '':
            sequence_lens = self.graph.get_input_node(node,
                                                      idx=exist_input_nums,
                                                      copy=True)
            exist_input_nums += 1
        if input_nums > 5 and node.layer.input[5] != '':
            init_h = self.graph.get_input_node(node,
                                               idx=exist_input_nums,
                                               copy=True)
            init_h_shape = init_h.out_shapes[0]
            if len(init_h_shape) != 0 and reduce(lambda x, y: x * y,
                                                 init_h_shape) not in [1, -1]:
                self.paddle_graph.add_layer('paddle.reshape',
                                            inputs={"x": init_h.name},
                                            outputs=[init_h.name],
                                            shape=init_h.out_shapes[0])
            exist_input_nums += 1
        if input_nums > 6 and node.layer.input[6] != '':
            init_c = self.graph.get_input_node(node,
                                               idx=exist_input_nums,
                                               copy=True)
            init_c_shape = init_c.out_shapes[0]
            if len(init_c_shape) != 0 and reduce(lambda x, y: x * y,
                                                 init_c_shape) not in [1, -1]:
                self.paddle_graph.add_layer('paddle.reshape',
                                            inputs={"x": init_c.name},
                                            outputs=[init_c.name],
                                            shape=init_c.out_shapes[0])

        input_weight_np = _const_weight_or_none(input_weight)
        _rename_or_remove_weight(self.weights, input_weight.name)
        hidden_size = node.get_attr('hidden_size', input_weight_np.shape[1] / 4)
        input_size = input_weight_np.shape[2]
        hidden_weight_np = _const_weight_or_none(hidden_weight)
        _rename_or_remove_weight(self.weights, hidden_weight.name)
        bias_np = _const_weight_or_none(bias)
        _rename_or_remove_weight(self.weights, bias.name)
        input_bias_np = bias_np[:, :4 * hidden_size]
        hidden_bias_np = bias_np[:, 4 * hidden_size:]

        # parameters order in paddle:lstm:
        # 1. gate order in paddle is: input, forget, cell, output.
        # 2. gate orfer in onnx is: input, output, forget, cell.

        def reform_weights(w, n, intervals):
            slices = [w[:, x * n:y * n] for x, y in intervals]
            return np.concatenate(slices, axis=1)

        def transform_weight_with_bias(weights, n, intervals):
            return [reform_weights(w, n, intervals) for w in weights]

        reform_permutation = [(0, 1), (2, 4), (1, 2)]

        weights = transform_weight_with_bias(
            [input_weight_np, hidden_weight_np, input_bias_np, hidden_bias_np],
            hidden_size, reform_permutation)

        op_name = name_generator("lstm", self.nn_name2id)
        y_out = node.output(0)
        yh_out = node.output(1)
        yc_out = node.output(2)
        direction = node.get_attr('direction', 'forward')

        def generate_paddle_param_names(op_name, suffix=''):
            param_names = []
            param_names.extend(['{}.weight_ih_l0{}', '{}.weight_hh_l0{}'])
            if have_bias != False:
                param_names.append('{}.bias_ih_l0{}')
            if have_bias != False:
                param_names.append('{}.bias_hh_l0{}')
            param_names = [x.format(op_name, suffix) for x in param_names]
            return param_names

        def assign_params(op_name, weights, weight_idx=0, suffix=''):
            param_names = generate_paddle_param_names(op_name, suffix)
            for param_name, weight in zip(param_names, weights):
                self.weights[param_name] = weight[weight_idx]

        if direction == 'backward':
            raise Exception(
                "LSTM support 'forward' or 'bidirectional', except '{}'.".
                format(direction))
        else:
            assign_params(op_name, weights)
            if direction == 'bidirectional':
                assign_params(op_name, weights, 1, '_reverse')

        self.paddle_graph.add_layer('paddle.nn.LSTM',
                                    inputs={
                                        'input': x.name,
                                        'initial_states':
                                        (init_h.name, init_c.name)
                                    },
                                    outputs=[op_name, y_out, yh_out, yc_out],
                                    input_size=input_size,
                                    hidden_size=hidden_size,
                                    num_layers=1,
                                    direction=string(direction),
                                    time_major=True)

        self.paddle_graph.add_layer('paddle.reshape',
                                    inputs={"x": y_out},
                                    outputs=[y_out],
                                    shape=[0, 0, -1, hidden_size])
        self.paddle_graph.add_layer('paddle.transpose',
                                    inputs={"x": y_out},
                                    outputs=[y_out],
                                    perm=[0, 2, 1, 3])

    @print_mapping_info
    def TopK(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_k = self.graph.get_input_node(node, idx=1, copy=True)
        # If the topk result is the entire graph output, modify the graph result
        graph_output_new = list()
        if node.layer_name in self.graph.output_nodes:
            graph_output_new = [
                "{}_p{}".format(node.layer_name, 0)
                if x == node.layer_name else x for x in self.graph.output_nodes
            ]
            self.paddle_graph.outputs = graph_output_new
        layer_attrs = dict()
        layer_attrs["axis"] = node.get_attr('axis', -1)
        layer_attrs["largest"] = True if node.get_attr('largest',
                                                       1) == 1 else False
        layer_attrs["sorted"] = True if node.get_attr('sorted',
                                                      1) == 1 else False
        k = _const_weight_or_none(val_k)
        if isinstance(k, (list, tuple, np.ndarray)):
            k = k[0]
        # If k can get the value directly, it is used as an attribute; otherwise it is used as an input tensor
        if k is not None:
            layer_attrs["k"] = k
            self.paddle_graph.add_layer("paddle.topk",
                                        inputs={"x": val_x.name},
                                        outputs=[
                                            "{}_p{}".format(node.layer_name, 0),
                                            "{}_p{}".format(node.layer_name, 1)
                                        ],
                                        **layer_attrs)
        else:
            if val_k.dtype != "int32":
                self.paddle_graph.add_layer("paddle.cast",
                                            inputs={"x": val_k.name},
                                            outputs=[val_k.name],
                                            dtype=string('int32'))
            self.paddle_graph.add_layer("paddle.topk",
                                        inputs={
                                            "x": val_x.name,
                                            "k": val_k.name
                                        },
                                        outputs=[
                                            "{}_p{}".format(node.layer_name, 0),
                                            "{}_p{}".format(node.layer_name, 1)
                                        ],
                                        **layer_attrs)

    @print_mapping_info
    def LRN(self, node):
        op_name = name_generator("lrn", self.nn_name2id)
        output_name = node.name
        layer_outputs = [op_name, output_name]
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        alpha = node.get_attr('alpha', 0.0001)
        beta = node.get_attr('beta', 0.75)
        bias = node.get_attr('bias', 1.0)
        size = node.get_attr('size')
        layer_attrs = {'size': size, 'alpha': alpha, 'beta': beta, 'k': bias}
        self.paddle_graph.add_layer("paddle.nn.LocalResponseNorm",
                                    inputs={"x": val_x.name},
                                    outputs=layer_outputs,
                                    **layer_attrs)

    @print_mapping_info
    def DepthToSpace(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        blocksize = node.get_attr('blocksize')
        mode = node.get_attr('mode', "DCR")
        val_x_shape = val_x.out_shapes[0]
        b, c, h, w = val_x_shape
        if mode == "DCR":
            self.paddle_graph.add_layer(
                'paddle.reshape',
                inputs={"x": val_x.name},
                outputs=[node.name],
                shape=[b, blocksize, blocksize, c // (blocksize**2), h, w])
            self.paddle_graph.add_layer('paddle.transpose',
                                        inputs={"x": node.name},
                                        outputs=[node.name],
                                        perm=[0, 3, 4, 1, 5, 2])
            self.paddle_graph.add_layer(
                'paddle.reshape',
                inputs={"x": node.name},
                outputs=[node.name],
                shape=[b, c // (blocksize**2), h * blocksize, w * blocksize])
        else:
            self.paddle_graph.add_layer(
                'paddle.reshape',
                inputs={"x": val_x.name},
                outputs=[node.name],
                shape=[b, c // (blocksize**2), blocksize, blocksize, h, w])
            self.paddle_graph.add_layer('paddle.transpose',
                                        inputs={"x": node.name},
                                        outputs=[node.name],
                                        perm=[0, 1, 4, 2, 5, 3])
            self.paddle_graph.add_layer(
                'paddle.reshape',
                inputs={"x": node.name},
                outputs=[node.name],
                shape=[b, c // (blocksize**2), h * blocksize, w * blocksize])

    @print_mapping_info
    def NonMaxSuppression(self, node):
        nn_op_name = name_generator("nms", self.nn_name2id)
        output_name = node.name
        layer_outputs = [nn_op_name, output_name]
        boxes = self.graph.get_input_node(node, idx=0, copy=True)
        scores = self.graph.get_input_node(node, idx=1, copy=True)
        inputs_len = len(node.layer.input)
        layer_attrs = dict()
        layer_attrs["keep_top_k"] = -1
        layer_attrs["nms_threshold"] = 0.0
        layer_attrs["score_threshold"] = 0.0
        if inputs_len > 2:
            max_output_boxes_per_class = self.graph.get_input_node(node,
                                                                   idx=2,
                                                                   copy=True)
            max_output_boxes_per_class = _const_weight_or_none(
                max_output_boxes_per_class)
            if len(scores.out_shapes[0]) != 0:
                num_classes = scores.out_shapes[0][1]
            else:
                num_classes = 1
            if max_output_boxes_per_class is not None:
                max_output_boxes_per_class = max_output_boxes_per_class.tolist()
                if isinstance(max_output_boxes_per_class, int):
                    layer_attrs[
                        "keep_top_k"] = max_output_boxes_per_class * num_classes
                else:
                    layer_attrs["keep_top_k"] = max_output_boxes_per_class[
                        0] * num_classes
        if inputs_len > 3:
            iou_threshold = self.graph.get_input_node(node, idx=3, copy=True)
            layer_attrs["nms_threshold"] = _const_weight_or_none(
                iou_threshold).tolist()[0]
        if inputs_len > 4:
            score_threshold = self.graph.get_input_node(node, idx=4, copy=True)
            layer_attrs["score_threshold"] = _const_weight_or_none(
                score_threshold).tolist()[0]
        self.paddle_graph.add_layer("custom_layer:NMS",
                                    inputs={
                                        "bboxes": boxes.name,
                                        "scores": scores.name
                                    },
                                    outputs=layer_outputs,
                                    **layer_attrs)

    @print_mapping_info
    def ReduceL1(self, node):
        output_name = node.name
        layer_outputs = [output_name]
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        axes = node.get_attr('axes')
        keepdims = False if node.get_attr('keepdims') == 0 else True
        layer_attrs = {'p': 1, 'axis': axes, 'keepdim': keepdims}
        self.paddle_graph.add_layer("paddle.norm",
                                    inputs={"x": val_x.name},
                                    outputs=layer_outputs,
                                    **layer_attrs)

    @print_mapping_info
    def ReduceL2(self, node):
        output_name = node.name
        layer_outputs = [output_name]
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        axes = node.get_attr('axes')
        keepdims = False if node.get_attr('keepdims') == 0 else True
        layer_attrs = {'p': 2, 'axis': axes, 'keepdim': keepdims}
        self.paddle_graph.add_layer("paddle.norm",
                                    inputs={"x": val_x.name},
                                    outputs=layer_outputs,
                                    **layer_attrs)
