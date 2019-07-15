import torch


def diff_mse(x, y):
    x_vec = x.view(1, -1).squeeze()
    y_vec = y.view(1, -1).squeeze()
    return torch.mean(torch.pow((x_vec - y_vec), 2)).item()


def conv2d_scalar(x_in, conv_weight, conv_bias, device):
    N_batch, C_x, S_x_h, S_x_w = x_in.shape
    B = conv_bias.size(0)
    K = conv_weight.size(-1)
    S_conv_h = S_x_h - K + 1
    S_conv_w = S_x_w - K + 1
    z_conv = torch.zeros(N_batch, B, S_conv_h, S_conv_w).to(device)
    for n in range(N_batch):
        for c_out in range(B):
            for m in range(S_conv_h):
                for l in range(S_conv_w):
                    for c_in in range(C_x):
                        for i in range(K):
                            for j in range(K):
                                z_conv[n][c_out][m][l] += x_in[n][c_in][m + i][l + j] * conv_weight[c_out][c_in][i][j]
                    z_conv[n][c_out][m][l] += conv_bias[c_out]
    return z_conv


def conv2d_vector(x_in, conv_weight, conv_bias, device):
    N_batch, C_in, S_in_h, S_in_w = x_in.shape
    B = conv_bias.size(0)
    K = conv_weight.size(-1)
    S_conv_h = S_in_h - K + 1
    S_conv_w = S_in_w - K + 1
    conv_weight_rows = conv_weight2rows(conv_weight)    
    z_conv = torch.zeros(N_batch, B, S_conv_h, S_conv_w).to(device)
    for n in range(N_batch):
        col = im2col(x_in[n], K, device).t()
        z_conv[n] = (conv_weight_rows.matmul(col) + conv_bias.view(-1, 1)).view(B, S_conv_h, S_conv_w)
    return z_conv


def im2col(X, kernel_size, device, stride=1):  # could be more optimize in matrix manipulation way
    C_in, S_in_h, S_in_w = X.shape
    S_out_h = int((S_in_h - kernel_size) / stride) + 1
    S_out_w = int((S_in_w - kernel_size) / stride) + 1
    cols = torch.zeros(C_in * S_out_h * S_out_w, kernel_size**2).to(device)
    for c in range(C_in):
        for i in range(S_out_h):
            for j in range(S_out_w):
                cols[c * S_out_h * S_out_w + i * S_out_h + j] = X[c][i * stride:i * stride + kernel_size,
                                                                     j * stride:j * stride + kernel_size].contiguous().view(1, -1)
    return cols


def conv_weight2rows(conv_weight):
    N_out, N_in = conv_weight.shape[:2]
    w = conv_weight.view(N_out * N_in, -1)
    return w


def pool2d_scalar(a, device):
    N_batch, B, S_conv_h, S_conv_w = a.shape
    S_pool_h = int(S_conv_h / 2)
    S_pool_w = int(S_conv_w / 2)
    z_pool = torch.zeros(N_batch, B, S_pool_h, S_pool_w).to(device)
    for n in range(N_batch):
        for c in range(B):
            for m in range(S_pool_h):
                for l in range(S_pool_w):
                    z_pool[n][c][m][l] = max(a[n][c][2*m][2*l], a[n][c][2*m][2*l + 1], a[n][c][2*m + 1][2*l], a[n][c][2*m + 1][2*l + 1])
    return z_pool


def pool2d_vector(a, device):
    N_batch, B, S_conv_h, S_conv_w = a.shape
    S_pool_h = int(S_conv_h / 2)
    S_pool_w = int(S_conv_w / 2)
    z_pool = torch.zeros(N_batch, B, S_pool_h, S_pool_w).to(device)
    for n in range(N_batch):
        col = im2col(a[n], 2, device, stride=2).t()
        max_pick = col.max(dim=0)[0].view(B, S_pool_h, S_pool_w)
        z_pool[n] = max_pick
    return z_pool


def relu_scalar(a, device):
    N_batch, P_1 = a.shape
    z_relu = torch.zeros(N_batch, P_1).to(device)
    for n in range(N_batch):
        for i in range(P_1):
            z_relu[n][i] = max(a[n][i], 0)
    return z_relu


def relu_vector(a, device):
    z_relu = a.to(device)
    z_relu[a < 0] = 0
    return z_relu


def reshape_scalar(a, device):
    N_batch, B, S_pool_h, S_pool_w = a.shape
    N_reshape = B * S_pool_h * S_pool_w
    z_reshaped = torch.zeros(N_batch, N_reshape).to(device)
    for n in range(N_batch):
        for c in range(B):
            for m in range(S_pool_h):
                for l in range(S_pool_w):
                    j = c * S_pool_h * S_pool_w + m * S_pool_h + l
                    z_reshaped[n][j] = a[n][c][m][l]
    return z_reshaped


def reshape_vector(a, device):
    N_batch = a.size(0)
    z_reshaped = a.to(device).view(N_batch, -1)
    return z_reshaped


def fc_layer_scalar(a, weight, bias, device):
    N_batch, N_reshape = a.shape
    P_1 = bias.size(0)
    z_fc = torch.zeros(N_batch, P_1).to(device)
    for n in range(N_batch):
        for j in range(P_1):
            for i in range(N_reshape):
                z_fc[n][j] += weight[j][i] * a[n][i]
            z_fc[n][j] += bias[j]
    return z_fc


def fc_layer_vector(a, weight, bias, device):
    z_fc = (a.matmul(weight.t()) + bias).to(device)
    return z_fc
