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

import os
import sys
import importlib
import shutil
import numpy as np
import logging
import paddle
import torch

from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config


def compare(result, expect, delta=1e-10, rtol=1e-10):
    """
    param meaning:
    result: torch result
    expect: paddle result
    delta: absolute error
    rtol: relative error
    """
    if type(result) == np.ndarray:
        if type(expect) == list:
            expect = expect[0]
        expect = np.array(expect)
        res = np.allclose(result, expect, atol=delta, rtol=rtol, equal_nan=True)
        # print wrong result
        if res is False:
            if result.dtype == np.bool_:
                diff = abs(result.astype("int32") - expect.astype("int32"))
            else:
                diff = abs(result - expect)
            logging.error("Output has diff! max diff: {}".format(np.amax(diff)))
        if result.dtype != expect.dtype:
            logging.error(
                "Different output data types! res type is: {}, and expect type is: {}"
                .format(result.dtype, expect.dtype))
        assert res
        assert result.shape == expect.shape, "result.shape: {} != expect.shape: {}".format(
            result.shape, expect.shape)
        assert result.dtype == expect.dtype, "result.dtype: {} != expect.dtype: {}".format(
            result.dtype, expect.dtype)
    elif isinstance(result, (list, tuple)):
        for i in range(len(result)):
            if isinstance(result[i], (np.generic, np.ndarray)):
                compare(result[i], expect[i], delta, rtol)
            else:
                compare(result[i].numpy(), expect[i], delta, rtol)
    # deal with scalar tensor
    elif len(expect) == 1:
        compare(result, expect[0], delta, rtol)
    else:
        raise Exception("Compare diff wrong!!!!!!")


def randtool(dtype, low, high, shape):
    """
    np random tools
    """
    if dtype == "int":
        return np.random.randint(low, high, shape)

    elif dtype == "float":
        return low + (high - low) * np.random.random(shape)

    elif dtype == "bool":
        return np.random.randint(low, high, shape).astype("bool")


class TorchConverter(object):
    """
     Torch model transfer to paddle
    """

    def __init__(self,
                 model,
                 file_name,
                 op_type=[],
                 inputs_shape=[],
                 delta=1e-5,
                 rtol=1e-5,
                 run_dynamic=False):
        self.op_type = op_type
        assert isinstance(self.op_type,
                          str), "The dtype of op_type must be string!"
        self.seed = 33
        np.random.seed(self.seed)
        paddle.seed(self.seed)
        self.places = ['cpu']
        self.model = model
        self.name = file_name
        self.pwd = os.getcwd()
        self.delta = delta
        self.rtol = rtol
        self.kwargs_dict = {"input_data": ()}
        self.input_feed = {}
        self.inputs_dtype = []
        self.inputs_shape = inputs_shape
        self.run_dynamic = run_dynamic

    def set_input_data(self, group_name, *args):
        """
        set input data
        """
        self.kwargs_dict[group_name] = args
        if isinstance(self.kwargs_dict[group_name][0], tuple):
            self.kwargs_dict[group_name] = self.kwargs_dict[group_name][0]

        i = 0
        add_inputs_shape = False
        if len(self.inputs_shape) == 0:
            add_inputs_shape = True
        for in_data in self.kwargs_dict[group_name]:
            if isinstance(in_data, list):
                for data in in_data:
                    self.inputs_dtype.append(str(data.dtype))
                    self.input_feed[str(i)] = data
                    if add_inputs_shape:
                        self.inputs_shape.append(data.shape)
                    i += 1
            else:
                if isinstance(in_data, tuple):
                    in_data = in_data[0]
                self.inputs_dtype.append(str(in_data.dtype))
                self.input_feed[str(i)] = in_data
                if add_inputs_shape:
                    self.inputs_shape.append(in_data.shape)
                i += 1

    def _mkdir(self):
        """
        make dir to save all
        """
        save_path = os.path.join(self.pwd, self.name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    def _torch_to_paddle(self, ):
        """
        convert torch to paddle
        """
        from x2paddle.convert import pytorch2paddle

        paddle_path = os.path.join(self.pwd, self.name, self.name + '_paddle')
        input_examples = list()
        for i in range(len(self.input_feed)):
            input_examples.append(torch.Tensor(self.input_feed[str(i)]))
        pytorch2paddle(self.model,
                       paddle_path,
                       jit_type="trace",
                       input_examples=input_examples,
                       convert_to_lite=False,
                       disable_feedback=True)

    def _mk_paddle_res(self, ):
        """
        make paddle res
        """
        # input data
        paddle_tensor_feed = list()
        result = list()
        for i in range(len(self.input_feed)):
            paddle_tensor_feed.append(paddle.to_tensor(self.input_feed[str(i)]))

        ## PaddleInference not support float64
        if "float64" in self.inputs_dtype:
            self.run_dynamic = True

        # TODO(megemini): create_predictor stuck
        self.run_dynamic = True

        if self.run_dynamic:
            paddle_path = os.path.join(self.pwd, self.name,
                                       self.name + '_paddle/')
            restore = paddle.load(os.path.join(paddle_path, "model.pdparams"))
            sys.path.insert(0, paddle_path)
            import x2paddle_code
            # Solve the problem of function overloading caused by traversing the model
            importlib.reload(x2paddle_code)
            model = getattr(x2paddle_code, "Net")()
            model.set_dict(restore)
            model.eval()
            result = model(*paddle_tensor_feed)
        else:
            paddle_model_path = os.path.join(
                self.pwd, self.name,
                self.name + '_paddle/inference_model/model.pdmodel')
            paddle_param_path = os.path.join(
                self.pwd, self.name,
                self.name + '_paddle/inference_model/model.pdiparams')
            config = Config()
            config.set_prog_file(paddle_model_path)
            if os.path.exists(paddle_param_path):
                config.set_params_file(paddle_param_path)
            # initial GPU memory(M), device ID
            config.enable_use_gpu(200, 0)
            # optimize graph and fuse op
            config.switch_ir_optim(False)
            config.enable_memory_optim()
            # disable feed, fetch OP, needed by zero_copy_run
            config.switch_use_feed_fetch_ops(False)
            config.disable_glog_info()
            pass_builder = config.pass_builder()
            predictor = create_predictor(config)
            input_names = predictor.get_input_names()
            output_names = predictor.get_output_names()
            for i in range(len(input_names)):
                input_tensor = predictor.get_input_handle(input_names[i])
                input_tensor.copy_from_cpu(self.input_feed[str(i)])
            predictor.run()
            for output_name in output_names:
                output_tensor = predictor.get_output_handle(output_name)
                result.append(output_tensor.copy_to_cpu())
        shutil.rmtree(os.path.join(self.pwd, self.name))
        # get paddle outputs
        if isinstance(result, (tuple, list)):
            if isinstance(result[0], np.ndarray):
                result = tuple(out for out in result)
            else:
                result = tuple(out.numpy() for out in result)
        else:
            if isinstance(result, np.ndarray):
                result = (result, )
            else:
                result = (result.numpy(), )
        return result

    def _mk_torch_res(self, ):
        """
        make torch res
        """
        input_examples = list()
        for i in range(len(self.input_feed)):
            input_examples.append(torch.Tensor(self.input_feed[str(i)]))

        torch_outs = self.model(*input_examples)
        # get torch outputs
        if isinstance(torch_outs, (tuple, list)):
            torch_outs = tuple(out.numpy() for out in torch_outs)
        else:
            torch_outs = (torch_outs.numpy(), )
        return torch_outs

    def run(self):
        """
        1. make torch model
        2. convert torch to paddle
        3. use torch to make res
        4. compare diff
        """
        self._mkdir()
        for place in self.places:
            paddle.set_device(place)
            torch_res = None
            paddle_res = None
            # run torch api and make torch res
            self._torch_to_paddle()
            torch_res = self._mk_torch_res()
            paddle_res = self._mk_paddle_res()
            compare(torch_res, paddle_res, delta=self.delta, rtol=self.rtol)
