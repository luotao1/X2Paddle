# 进行转换
x2paddle -f onnx -m ../dataset/IDG_multi_modal_crop_cnn_model/multi_modal_crop_cnn_model.onnx -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
