import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFPN_Concat(nn.Module):
    """
    BiFPN Concat Layer cho họ mô hình YOLO.
    Tự động nội suy (interpolate) các feature map về cùng kích thước không gian
    với feature map đầu tiên trong danh sách trước khi nối (concatenate).
    """

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        # x là một list chứa các tensor đầu vào (ví dụ: [P4_current, P5_topdown, P3_bottomup])
        target_shape = x[0].shape[2:]  # Lấy (Height, Width) của tensor đầu tiên làm chuẩn

        outs = [x[0]]
        for i in range(1, len(x)):
            tensor = x[i]
            # Nếu kích thước không khớp, tiến hành resize (Nearest Neighbor cho tốc độ nhanh)
            if tensor.shape[2:] != target_shape:
                tensor = F.interpolate(tensor, size=target_shape, mode='nearest')
            outs.append(tensor)

        # Ghép nối các tensor dọc theo chiều channel (dimension=1)
        return torch.cat(outs, self.d)