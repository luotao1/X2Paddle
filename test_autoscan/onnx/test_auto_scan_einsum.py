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

from auto_scan_test import OPConvertAutoScanTest
import hypothesis.strategies as st
import unittest


class TestEinsumConvert(OPConvertAutoScanTest):
    """
    ONNX op: Einsum
    OPset version: 12~15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(st.integers(min_value=10, max_value=50),
                     min_size=1,
                     max_size=1))

        input_dtype = draw(st.sampled_from(["float32", "float64"]))

        equation = "i,j->ij"

        config = {
            "op_names": ["Einsum"],
            "test_data_shapes": [input_shape, input_shape],
            "test_data_types": [[input_dtype], [input_dtype]],
            "inputs_shape": [],
            "min_opset_version": 12,
            "max_opset_version": 15,
            "inputs_name": ["x", "y"],
            "outputs_name": ["z"],
            "delta": 1e-4,
            "rtol": 1e-4,
            "enable_onnx_checker": True,
        }
        attrs = {
            "equation": equation,
        }

        return (config, attrs)

    def test(self):
        self.run_and_statis(max_examples=100)


if __name__ == "__main__":
    unittest.main()
