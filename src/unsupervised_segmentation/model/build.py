from ts_openSARship.model.model_parts import Backbone, Neck
from ts_openSARship.model.head_3 import Head
import torch.nn as nn
import torch
import logging
from typing import Union, Dict
from ts_openSARship.model.util.initilizers import initialize_biases, custom_init
from ts_openSARship.post_process.model import export_model_to_onnx
import gc
from .MODEL_CONFIG import BACKBONE_CPS_BLOCKS, N_SCALES
from ts_openSARship.dataset.boxes import anchor_prediction2x1y1x2y2
from ts_openSARship.post_process.targets import NMS_post


class ObjectDetector(nn.Module):
    def __init__(
        self,
        input_channels: int = 2,
        number_classes: int = 2,
        weight: float = 1,
        stride: int = 32,
        dropout: float = 0.05,
        gaussians_sog: int = 2,
        head_metadata_values: int = 6,
    ):
        """

        save model without post processing ONNX

        import torch

        # Initialize the model and set it to evaluation mode
        model_without_pp = ObjectDetector()
        model_without_pp.eval()
        dummy_input = torch.randn(1, 2, 256, 256)  # Example input tensor

        # Export the model to ONNX
        torch.onnx.export(model_without_pp, dummy_input, "model_without_post_processing.onnx",
                        opset_version=12, input_names=['input'], output_names=['output'],
                        dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})


        save model with post processing

        # Initialize the model and set it to evaluation mode with post-processing enabled
        model_with_pp = ObjectDetector()
        model_with_pp.eval()

        # Since post-processing flag affects the forward pass, use a lambda to add the flag
        dummy_input = torch.randn(1, 2, 256, 256)  # Example input tensor
        torch.onnx.export(lambda x: model_with_pp(x, post_processing=True), dummy_input, "model_with_post_processing.onnx",
                        opset_version=12, input_names=['input'], output_names=['output'],
                        dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})


        """

        super(ObjectDetector, self).__init__()

        self.n_scales = N_SCALES
        self.strides = (stride // 4, stride // 2, stride)
        self.input_channels = input_channels
        self.verbose = 1
        self.number_classes = number_classes
        self.dropout = dropout

        self._background_threshold = 0.1

        self.head_metadata_values = head_metadata_values

        self._ship_index_in_target = 5
        self.gaussians_sog = gaussians_sog

        self.training_mode = True
        self.export_mode = False
        self._post_process = False

        ##############################
        ##### Building the model #####
        ##############################

        self.backbone = Backbone(input_channels, weight=weight, stride=self.strides[-1], CPS_NBlocks=BACKBONE_CPS_BLOCKS, dropout=dropout)
        self.neck = Neck(weight=weight, CPS_NBlocks=3)
        self.head = Head(
            weight=weight, number_classes=number_classes, dropout=dropout, gaussians_sog=self.gaussians_sog, metadata_values=self.head_metadata_values
        )

        ##############################
        ##### initializing the model #####
        ##############################
        self.backbone.apply(custom_init)  # Initialize convolutional layers
        self.neck.apply(initialize_biases)
        self.head.apply(custom_init)
        # Calculate the bin indices corresponding to the mean width and length

        bias_vector = torch.full((2,), 0.1, dtype=torch.float32)
        for ix in range(len(self.head.conv_classificaiton)):
            self.head.conv_classificaiton[ix].bias.data = bias_vector

        bias_vector = torch.full((2,), 0.5, dtype=torch.float32)
        for ix in range(len(self.head.xcyc_layer)):
            self.head.xcyc_layer[ix].bias.data = bias_vector

        bias_vector = torch.full((2,), 0.00002, dtype=torch.float32)
        for ix in range(len(self.head.wh_layer)):
            self.head.wh_layer[ix].bias.data = bias_vector

        self.head.length_width_conv[0].bias.data = torch.tensor([0.00002, 0.00002, 0.00002, 0.00002])
        self.head.length_width_conv[1].bias.data = torch.tensor([0.00002, 0.00002, 0.00002, 0.00002])
        self.head.length_width_conv[2].bias.data = torch.tensor([0.00002, 0.00002, 0.00002, 0.00002])

        for ix in range(len(self.head.conv_cog)):
            self.head.conv_cog[ix].bias.data = torch.tensor([0.00002, 0.00002])

        for ix in range(len(self.head.conv_sog)):
            if self.gaussians_sog == 2:
                self.head.conv_sog[ix].bias.data = torch.tensor([0.00002, 0.00002, 0.00002, 0.00002])
            else:
                self.head.conv_sog[ix].bias.data = torch.tensor([0.00002, 0.00002])
        """

        # TODO add the bias to the length and sog and width according to data.


        """
        ##############################
        #### post processing specks ####
        ##############################
        self.post_softmax = torch.nn.Softmax(dim=1)
        self._iou_threshold = 0.90
        self._skip_box_thr = 0.1
        self._conf_type = "max"
        self._background_threshold = 0.1

        # self.wbf_postprocessing = WBF_post_torch(iou_thr=self._iou_threshold, skip_box_thr=self._skip_box_thr , conf_type=self._conf_type)
        self.nms = NMS_post(iou_thr=self._iou_threshold, background_threshold=self._background_threshold)

    def forward(self, x, metadata=None):

        x = self.pad_to_stride(x)
        features = self.backbone(x)
        neck = self.neck(features)
        outputs = self.head(neck, metadata)

        # x = self.pad_to_stride(x)
        #  features = self.backbone(x)
        #  #neck = self.neck(features)
        #  outputs = self.head(neck, metadata)

        # only entering if it is in eval model and post_processing is true
        if not self.training_mode and self._post_process:
            outputs = self.post_processing(outputs, x.shape)

        return outputs

    def train(self, training_mode=True):
        super().train(training_mode)
        self.training_mode = training_mode
        return self

    def eval(self):
        return self.train(False)

    def post_processing(self, outputs, img_size):
        if self.export_mode is False:
            outputs = anchor_prediction2x1y1x2y2(outputs, strides=self.strides, image_size=img_size[-2:])
            if outputs[0].shape[-3] > 20:
                prob_ix = self.head.class_index[1]
            else:
                prob_ix = self._ship_index_in_target

            predictions = []
            for img_ix in range(outputs[0].shape[0]):  # 0 to 2
                img_temp = torch.cat(
                    [
                        outputs[scale_ix][img_ix, :, outputs[scale_ix][img_ix, prob_ix, :, :] > self._background_threshold].T
                        for scale_ix in range(len(outputs))
                        if outputs[scale_ix][img_ix, prob_ix, :, :].any()
                    ],
                    dim=0,
                )
                if img_temp.nelement() > 0:
                    pred = self.nms(img_temp, prob_ix)
                    pred[
                        :, self.head.target_index["size_index"][0] : self.head.target_index["size_index"][0] + self.head.width_outputs
                    ] = self.post_softmax(
                        pred[:, self.head.target_index["size_index"][0] : self.head.target_index["size_index"][0] + self.head.width_outputs]
                    )
                    pred[
                        :,
                        self.head.target_index["size_index"][0]
                        + self.head.width_outputs : self.head.target_index["size_index"][0]
                        + self.head.width_outputs
                        + self.head.length_outputs,
                    ] = self.post_softmax(
                        pred[
                            :,
                            self.head.target_index["size_index"][0]
                            + self.head.width_outputs : self.head.target_index["size_index"][0]
                            + self.head.width_outputs
                            + self.head.length_outputs,
                        ]
                    )
                    predictions.append(pred)

            outputs = predictions
        return outputs

    def pad_to_stride(self, tensor):
        height, width = tensor.shape[-2:]
        pad_height = (self.strides[-1] - height % self.strides[-1]) * (height % self.strides[-1] != 0)
        pad_width = (self.strides[-1] - width % self.strides[-1]) * (width % self.strides[-1] != 0)
        # Padding format: (left, right, top, bottom)
        padded_tensor = nn.functional.pad(tensor, (0, pad_width, 0, pad_height))

        return padded_tensor

    def export(
        self,
        file_path: str,
        metadata: Union[None, dict, Dict] = None,
        simplify_model: bool = False,
        opset_version: int = 11,
    ):
        self.export_mode = True

        x = torch.randn(1, self.input_channels, 256, 256).cpu()

        if file_path.endswith(".onnx"):
            logging.warning("ONNX gives nummerically different resutls! Better to use .pt")
            try:
                export_model_to_onnx(
                    self,
                    file_path,
                    x,
                    metadata=metadata,
                    simplify_model=simplify_model,
                    opset_version=opset_version,
                    verbose=self.verbose - 1,
                )
                self.export_type = "onnx"
            except Exception as e:
                logging.info(f"Error exporting model: {e}")

        elif file_path.endswith((".pth", ".torch", ".pytorch", ".pt")):
            try:
                traced_model = torch.jit.trace(self, x, strict=False)
                traced_model.save(file_path)
                self.export_type = "torch_jitter"
                del traced_model
                gc.collect()
            except Exception as e:
                logging.info(f"Error exporting model: {e}")

        return None

    def targets(self, targets, img_size):
        targets = anchor_prediction2x1y1x2y2(targets, strides=self.strides, image_size=img_size[-2:])

        targets = [
            torch.cat(
                [
                    targets[scale][image_inx, :, :, :][:, targets[scale][image_inx, 2, :, :] > 0]
                    for scale in range(len(targets))
                    if targets[scale][image_inx, 2, :, :].any()
                ],
                1,
            ).T
            for image_inx in range(targets[0].shape[0])
        ]
        return targets

    @property
    def training_mode(self):
        return self._training_mode

    @training_mode.setter
    def training_mode(self, value: bool):
        assert isinstance(value, bool), "The training_mode must be a boolean"
        self._training_mode = value
        return None

    @property
    def post_process(self):
        return self._post_process

    @post_process.setter
    def post_process(self, value: bool):
        self._post_process = value
        if value:
            self.eval()
        return None

    @property
    def export_mode(self):
        return self._export_mode

    @export_mode.setter
    def export_mode(self, value: bool):
        self._export_mode = value
        if value:
            self.eval()
            self.cpu()
        else:
            self.train()
        return None

    @property
    def background_threshold(self):
        return self._background_threshold

    @background_threshold.setter
    def background_threshold(self, value: float):
        assert 0 <= value <= 1, "The background_threshold must be between 0 and 1"
        self._background_threshold = value
        return None

    @property
    def iou_threshold(self):
        return self._iou_threshold

    @iou_threshold.setter
    def iou_threshold(self, value: float):
        assert 0 <= value <= 1, "The iou_threshold must be between 0 and 1"
        self._iou_threshold = value
        return None

    @property
    def skip_box_thr(self):
        return self._skip_box_thr

    @skip_box_thr.setter
    def skip_box_thr(self, value: float):
        assert 0 <= value <= 1, "The skip_box_thr must be between 0 and 1"
        self._skip_box_thr = value
        return None

    @property
    def conf_type(self):
        return self._conf_type

    @conf_type.setter
    def conf_type(self, value: str):
        assert value in ["max", "mean"], "The conf_type must be one of the following: 'max', 'mean'"
        self._conf_type = value
        return None
