import torch.nn as nn
from typing import Union, Dict
from ts_openSARship.model.loss.losses import (
    ClassificationLoss,
    BBoxRegressionLoss,
    WeightedSpeedLoss2D,
    CogLoss,
    SizeLoss,
    CIoULoss,
    IoULoss,
)
from ts_openSARship.model.loss.loss_util import pred_and_target2bbox
import torch


class LossComposer(nn.Module):
    """ "
    #     WeightedSpeedLoss2DWeighted,
    used to compose the total loss using a combination of different losses..
    # target has size B x 11 x F x F, where F is e.g. 32
    # prediction has size B x (N) x F x F, where N depends on the model..
    """

    def __init__(
        self,
        use_focal_loss: bool = True,
        weight_bbox: float = 1,
        weight_cls: float = 1,
        weight_size: float = 1,
        weight_sog: float = 1,
        weight_cog: float = 1,
        focal_gamma: float = 4,
        focal_alpha: Union[None, list] = [0.2, 1],
        cls_reduction: Union[None, str] = "none",
        target_index: Union[Dict, dict, None] = None,
        gaussians_sog: int = 2,
    ):
        super(LossComposer, self).__init__()

        self.target_index = target_index
        self.cls_reduction = cls_reduction
        self.gaussians_sog = gaussians_sog  # number of gaussians for the sog
        # loss functions
        bbox_reduction = "sum"
        size_reduction = "none"
        sog_reduction = "none"
        cog_reduction = "sum"

        self.sog_reduction = sog_reduction
        self.cog_reduction = cog_reduction
        self.size_reduction = size_reduction

        self.bboxloss_fn_l1 = BBoxRegressionLoss(reduction=bbox_reduction)
        self.bboxloss_fn_ci = CIoULoss(reduction=bbox_reduction)
        self.bboxloss_fn_iou = IoULoss(GIoU=False, DIoU=False, CIoU=True)
        self.bboxloss_fn = IoULoss(xywh=True, GIoU=True, DIoU=False, CIoU=True)

        self.clsloss_fn = ClassificationLoss(
            use_focal_loss=use_focal_loss,
            gamma=focal_gamma,
            alpha=focal_alpha,
            reduction=self.cls_reduction,
        )

        self.sizeloss_fn = SizeLoss(
            reduction=size_reduction,
        )

        self.sogloss_fn = WeightedSpeedLoss2D(reduction=sog_reduction)
        self.cogloss_fn = CogLoss(reduction=cog_reduction)
        self.bce_loss = nn.BCELoss()
        self.mseloss = nn.MSELoss()

        # loss internel regularizations
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        #  self.regularisation_lower_bound = 0.001
        #  self.regularisation_upper_bound = 25
        # loss wights
        self.weight_bbox = weight_bbox  # nn.Parameter(torch.tensor(weight_bbox, dtype=torch.float32))
        self.weight_cls = weight_cls  # nn.Parameter(torch.tensor(weight_cls, dtype=torch.float32))
        self.weight_size = weight_size  # nn.Parameter(torch.tensor(weight_size, dtype=torch.float32))
        self.weight_sog = weight_sog  # nn.Parameter(torch.tensor(weight_sog, dtype=torch.float32))
        self.weight_cog = weight_cog  # nn.Parameter(torch.tensor(weight_cog, dtype=torch.float32))

        # for logging purposes only
        self.loss_dict = {
            "list_of_losses": [
                str(self.bboxloss_fn_iou),
                str(self.bboxloss_fn_ci),
                str(self.clsloss_fn),
                str(self.sizeloss_fn),
                str(self.sogloss_fn),
                str(self.cogloss_fn),
            ],
            "weight_bbox": self.weight_bbox,
            "weight_cls": self.weight_cls,
            "weight_size": self.weight_size,
            "weight_sog": self.weight_sog,
            "weight_cog": self.weight_cog,
            "used_focal": use_focal_loss,
            "focal_gamma": self.focal_gamma,
            "focal_alpha": self.focal_alpha,
        }

        self.outputs = {
            "loss": 0,
            "bbox_loss_value": 0,
            "loss_width": 0,
            "loss_length": 0,
            "cls_loss_value": 0,
            "size_loss_value": 0,
            "sog_loss_value": 0,
            "cog_loss_value": 0,
        }

        self.eps = 1e-6

        self.target_sog_index = 8
        self.target_cog_index = 9

    def forward(self, prediction, target, stride: int = 4, image_size: int = 256):
        """
        stride is the number of pixels for each cell

        target [B x 11 x H x W]
            11: [l_x, b_y, exp(w), exp(h), features 1, background class, ships, other features]


        weight_bbox = self.regularisation_lower_bound + (self.regularisation_upper_bound - self.regularisation_lower_bound) * torch.sigmoid(
            self.weight_bbox
        )
        weight_cls = self.regularisation_lower_bound + (self.regularisation_upper_bound - self.regularisation_lower_bound) * torch.sigmoid(
            self.weight_cls
        )
        weight_size = self.regularisation_lower_bound + (self.regularisation_upper_bound - self.regularisation_lower_bound) * torch.sigmoid(
            self.weight_size
        )
        weight_sog = self.regularisation_lower_bound + (self.regularisation_upper_bound - self.regularisation_lower_bound) * torch.sigmoid(
            self.weight_sog
        )
        weight_cog = self.regularisation_lower_bound + (self.regularisation_upper_bound - self.regularisation_lower_bound) * torch.sigmoid(
            self.weight_cog
        )
        """
        try:
            try:
                cls_loss_value = torch.tensor(0.0)
                bbox_loss_value = torch.tensor(0.0)
                size_loss_value = torch.tensor(0.0)
                cog_loss_value = torch.tensor(0.0)
                sog_loss_value = torch.tensor(0.0)
                loss_width = torch.tensor(0.0)
                loss_length = torch.tensor(0.0)
            except Exception as e:
                print(f"Error in initialisation: {e}")
                pass
            ##############################
            ## Classification Loss     ##
            ##############################
            try:
                class_target = target[:, 4:6, :, :]
                class_pred = prediction[:, self.target_index["class_index"][0] : self.target_index["class_index"][-1] + 1, :, :]
                cls_loss_value = self.clsloss_fn(class_pred, class_target)
            except Exception as e:
                print(f"Error in cls_loss_value calculation: {e}")
                pass

            # try:
            #    class_pred = F.softmax(class_pred, dim=1)
            # except:
            #    pass

            try:
                filtered_pred_boxes, filtered_target_boxes = pred_and_target2bbox(
                    prediction, target, cell_size=stride, eps=1e-6, image_size=(image_size, image_size)
                )
            except Exception as e:
                print(f"Error in bbox calculation: {e}")
                print(" image_size: ", image_size)
                print("prediction: ", prediction.shape)
                print("target: ", target.shape)
                pass

            # Calculate individual losses if there are valid predictions and targets
            try:
                if filtered_pred_boxes.numel() > 0:
                    try:
                        if torch.isnan(filtered_pred_boxes).any() or torch.isnan(filtered_target_boxes).any():
                            print("NaN values detected in filtered_pred_boxes or filtered_target_boxes")
                        bbox_loss_value = self.bboxloss_fn(filtered_pred_boxes[:, :4], filtered_target_boxes[:, :4])
                    except Exception as e:
                        print(f"Error in bbox_loss_value calculation: {e}")

                    try:
                        sog_loss_value = self.sogloss_fn(
                            filtered_pred_boxes[:, self.target_index["sog_index"][0] : self.target_index["sog_index"][-1] + 1],
                            filtered_target_boxes[:, self.target_sog_index : self.target_sog_index + 1],
                        )
                    except Exception as e:
                        print(f"Error in sog_loss_value calculation: {e}")
                        pass

                    try:
                        cog_loss_value = self.cogloss_fn(filtered_pred_boxes[:, self.target_index["cog_index"][0] :], filtered_target_boxes[:, 9:])
                    except Exception as e:
                        print(f"Error in cog_loss_value calculation: {e}")
                        pass

                    try:
                        size_loss_value, loss_width, loss_length = self.sizeloss_fn(
                            filtered_pred_boxes[:, self.target_index["size_index"][0] : self.target_index["size_index"][-1] + 1],
                            filtered_target_boxes[:, 6:8],
                        )
                    except Exception as e:
                        print(f"Error in size_loss_value calculation: {e}")
                        pass
            except Exception as e:
                print(f"Error in individual loss calculation with filtered_pred_boxes.numel() > 0:: {e}")
                print("filtered_pred_boxes: ", filtered_pred_boxes.shape)
                print(" image_size: ", image_size)
                print("prediction: ", prediction.shape)
                print("target: ", target.shape)
                pass

            if self.sog_reduction not in ["mean", "sum"]:
                sog_loss_value = torch.where(sog_loss_value.gt(0), self.weight_size * sog_loss_value, size_loss_value)
                sog_loss_value = sog_loss_value.mean()
            else:
                size_loss_value = self.weight_size * size_loss_value

            if self.size_reduction not in ["mean", "sum"]:
                size_loss_value = torch.where(size_loss_value.gt(0), self.weight_size * size_loss_value, size_loss_value)
                size_loss_value = size_loss_value.mean()
            else:
                size_loss_value = self.weight_size * size_loss_value

            # Combine losses

            try:
                combined_loss = (
                    self.weight_bbox * bbox_loss_value
                    + self.weight_cls * cls_loss_value
                    + size_loss_value
                    + sog_loss_value
                    + self.weight_cog * cog_loss_value
                )
            except Exception as e:
                print(f"Error in combined_loss calculation: {e}")
                pass

            # Construct loss dictionary to return
            try:
                loss_dict = {
                    "loss": combined_loss,
                    "bbox_loss_value": bbox_loss_value,
                    "loss_width": loss_width,
                    "loss_length": loss_length,
                    "cls_loss_value": cls_loss_value,
                    "size_loss_value": size_loss_value,
                    "sog_loss_value": sog_loss_value,
                    "cog_loss_value": cog_loss_value,
                }

                self.loss_dict["weight_bbox"] = self.weight_bbox
                self.loss_dict["weight_cls"] = self.weight_cls
                self.loss_dict["weight_size"] = self.weight_size
                self.loss_dict["weight_sog"] = self.weight_sog
                self.loss_dict["weight_cog"] = self.weight_cog
            except Exception as e:
                print(f"Error in loss_dict construction: {e}")

            # Check for NaN values in the loss dictionary
            try:
                for key, value in loss_dict.items():
                    if torch.isnan(value).any():  # Convert to a single boolean using `.any()`
                        print(f"NaN detected in {key} with value {value}")
                    # if torch.isnan(value):
                    #    print(f"NaN detected in {key} with value {value}")
            except Exception as e:
                print(f"Error in NaN check: {e}")

            return loss_dict
        except Exception as e:
            print(f"Error in forward pass: {e}")
            print("prediction: ", prediction.shape)
            print("target: ", target.shape)
            print("stride: ", stride)
            print("image_size: ", image_size)

            pass
