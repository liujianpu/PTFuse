import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from collections import deque
import math
import pdb
from torchvision import transforms
import clip
class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()
        self.Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean')
    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_ssim_vis=self.Loss_ssim(image_y,generate_img)
        loss_ssim_ir=self.Loss_ssim(image_ir,generate_img)
        loss_ssim=loss_ssim_vis+loss_ssim_ir
        loss_total=loss_in+10*loss_grad+loss_ssim
        return loss_total

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.register_buffer("weightx", kernelx)
        self.register_buffer("weighty", kernely)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)




class EdgeScorer(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel边缘检测器
        self.sobel = nn.Conv2d(1, 2, 3, padding=1, bias=False)
        sobel_kernel = torch.tensor([
            [[[1, 0, -1], [2, 0, -2], [1, 0, -1]]],  # Horizontal
            [[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]   # Vertical
        ]).float()
        self.sobel.weight.data = sobel_kernel
        
    def forward(self, x):
        # 输入: (B,C,H,W)
        gray = x.mean(dim=1, keepdim=True) if x.shape[1]==3 else x
        edges = self.sobel(gray)                     # (B,2,H,W)
        edge_magnitude = edges.norm(dim=1)           # 边缘强度图 (B,H,W)
        return edge_magnitude.mean(dim=[1,2])        # 块的平均边缘强度 (B,)

class SelfContrastiveLoss(nn.Module):
    def __init__(self, temp=0.1, pos_samples=2, neg_samples=4):
        super().__init__()
        self.temp = temp
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples
        self.scorer = EdgeScorer()
        
    def forward(self, fus):
        B, C, H, W = fus.shape
        
        # 分割为16x16块 (B,36,C,16,16)
        patches = fus.unfold(2, 16, 16).unfold(3, 16, 16)  # (B,C,6,6,16,16)
        patches = patches.permute(0,2,3,1,4,5).reshape(B*6 * 6, C, 16, 16)
        
        # 计算各块边缘得分 (B*36,)
        scores = self.scorer(patches).view(B, 6 * 6)  # (B,36)
        
        # 动态锚点选择策略
        top_values, top_indices = scores.topk(self.pos_samples, dim=1) 
        anchor_idx = top_indices[:, 0]            # 最高分块作为锚点 (B,)
        pos_idx = top_indices[:, 1]               # 次高分块作为正样本 (B,)
        _, neg_idx = (-scores).topk(self.neg_samples, dim=1)  # 最低分块作为负样本 (B,4)
        # 特征提取 (使用块的平均颜色)
        patch_features = patches.mean(dim=[2,3]).view(B, 6 * 6, C)  # (B,36,C)
        
        # 收集对比样本
        anchor = patch_features[torch.arange(B), anchor_idx]  # (B,C)
        positive = patch_features[torch.arange(B), pos_idx]   # (B,C)
        negative = patch_features[torch.arange(B)[:, None], neg_idx]  # (B,4,C)
        
        # 计算相似度
        pos_sim = F.cosine_similarity(anchor.unsqueeze(1), positive.unsqueeze(1), dim=-1)  # (B,1)
        neg_sim = F.cosine_similarity(anchor.unsqueeze(1), negative, dim=-1)               # (B,4)
        
        # 构造对比损失
        logits = torch.cat([pos_sim, neg_sim], dim=1) / self.temp
        labels = torch.zeros(B, dtype=torch.long).to(fus.device)
        return F.cross_entropy(logits, labels)



# class edge(nn.Module):
#     def __init__(self, a=0.25):
#         super(edge, self).__init__()
#         self.a = a
        
#         # 定义各算子（单通道输入）
#         self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
#         self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
#         self.laplacian = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

#         # 初始化固定权重
#         with torch.no_grad():
#             # Sobel_x
#             self.sobel_x.weight.copy_(
#                 torch.tensor([[[[1, 0, -1],
#                                [2, 0, -2],
#                                [1, 0, -1]]]], dtype=torch.float32))
#             # Sobel_y
#             self.sobel_y.weight.copy_(
#                 torch.tensor([[[[1, 2, 1],
#                                [0, 0, 0],
#                                [-1, -2, -1]]]], dtype=torch.float32))
#             # Laplacian
#             self.laplacian.weight.copy_(
#                 torch.tensor([[[[0, 1, 0],
#                                [1, -4, 1],
#                                [0, 1, 0]]]], dtype=torch.float32))
            
#             # 冻结所有参数
#             for param in self.parameters():
#                 param.requires_grad = False

#     def forward(self, x):
#         """处理多通道输入的改进版本"""
#         # 转换为灰度 (假设输入为RGB或类似格式)
#         if x.shape[1] > 1:
#             gray = x.mean(dim=1, keepdim=True)  # (N,1,H,W)
#         else:
#             gray = x
        
#         # 计算各算子响应
#         sobel_x = self.sobel_x(gray)
#         sobel_y = self.sobel_y(gray)
#         sobel_mag = torch.sqrt(sobel_x**2 + sobel_y**2 + 1e-8)
#         laplacian = self.laplacian(gray).abs()
        
#         # 融合边缘特征
#         return self.a * (sobel_mag + laplacian)

# class SF(nn.Module):
#     """空间频率计算模块"""
#     def __init__(self):
#         super(SF, self).__init__()
        
#     def forward(self, x):
#         """输入形状: (N,C,H,W) 返回形状: (N,)"""
#         # 转换为灰度
#         if x.shape[1] > 1:
#             x = x.mean(dim=1, keepdim=True)
        
#         # 计算空间频率
#         dx = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])  # 行差分
#         dy = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])  # 列差分
        
#         RF = torch.sqrt(torch.mean(dx**2, dim=[1,2,3]))
#         CF = torch.sqrt(torch.mean(dy**2, dim=[1,2,3]))
#         return torch.sqrt(RF**2 + CF**2)

# class SelfContrastiveLoss(nn.Module):
#     def __init__(self, queue_size=2048, num_pos=4, num_neg=16, temp=0.1, 
#                  patch_size=16, momentum=0.999, edge_alpha=0.25):
#         super().__init__()
#         # 基础参数
#         self.queue_size = queue_size
#         self.num_pos = num_pos
#         self.num_neg = num_neg
#         self.temp = temp
#         self.patch_size = patch_size
#         self.momentum = momentum
        
#         # 边缘检测模块
#         self.edge = edge(a=edge_alpha)
        
#         # 空间频率模块
#         self.sf = SF()
        
#         # 初始化队列 (使用register_buffer保证设备同步)
#         self.register_buffer("pos_queue", torch.randn(queue_size, patch_size**2))
#         self.register_buffer("neg_queue", torch.randn(queue_size, patch_size**2))
#         self.pos_queue = F.normalize(self.pos_queue, dim=1)  # 初始归一化
#         self.neg_queue = F.normalize(self.neg_queue, dim=1)
        
#         # 队列指针
#         self.pos_queue_ptr = 0
#         self.neg_queue_ptr = 0

#     def compute_patch_scores(self, patches):
#         """输入形状: (N,C,H,W) 返回形状: (N,)"""
#         # 标准化到[0,255]
#         norm_patches = (patches - patches.min()) / (patches.max() - patches.min()) * 255.0
        
#         # 计算各指标
#         SFScore = self.sf(norm_patches)                          # 空间频率
#         SDScore = torch.std(norm_patches, dim=[1,2,3])           # 标准差
#         EdgeScore = self.edge(norm_patches).flatten(1).mean(-1)  # 边缘强度
        
#         return EdgeScore  #睡前没有改这个评价方式和255

#     def forward(self, fus):
#         """输入形状: (B,C,H,W)"""
#         B, C, H, W = fus.shape
#         device = fus.device
        
#         # 分块处理
#         patches = fus.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
#         patches = patches.contiguous().view(B, -1, C, self.patch_size, self.patch_size)
#         num_patches = patches.size(1)
#         patches = patches.view(-1, C, self.patch_size, self.patch_size)  # (B*num_patches, C, 16, 16)
        
#         # 计算各patch得分
#         scores = self.compute_patch_scores(patches)  # (B*num_patches,)
        
#         # 选择正负样本
#         _, topk_idx = torch.topk(scores, self.num_pos)
#         _, botk_idx = torch.topk(-scores, self.num_neg)
        
#         # 计算特征（保留梯度）
#         edge_features = self.edge(patches).flatten(1)  # (N, 256)
#         features = F.normalize(edge_features, dim=1)    # L2归一化
        
#         # ========== 队列更新 ==========
#         with torch.no_grad():
#             # 正样本队列更新
#             pos_features = features[topk_idx]
#             num_new = pos_features.size(0)
#             pos_ptr = self.pos_queue_ptr
            
#             # 替换队列中的旧数据
#             self.pos_queue[pos_ptr:pos_ptr+num_new] = pos_features
#             self.pos_queue_ptr = (pos_ptr + num_new) % self.queue_size
            
#             # 负样本队列更新（同理）
#             neg_features = features[botk_idx]
#             num_new_neg = neg_features.size(0)
#             neg_ptr = self.neg_queue_ptr
            
#             self.neg_queue[neg_ptr:neg_ptr+num_new_neg] = neg_features
#             self.neg_queue_ptr = (neg_ptr + num_new_neg) % self.queue_size
        
#         # ========== 对比损失计算 ==========
#         # 从队列中采样
#         rand_pos = torch.randperm(self.queue_size, device=device)[:self.num_pos]
#         rand_neg = torch.randperm(self.queue_size, device=device)[:self.num_neg]
#         pos_keys = self.pos_queue[rand_pos]  # (num_pos, 256)
#         neg_keys = self.neg_queue[rand_neg]  # (num_neg, 256)
        
#         # 计算相似度
#         sim_pos = torch.mm(features, pos_keys.T)  # (N, num_pos)
#         sim_neg = torch.mm(features, neg_keys.T)  # (N, num_neg)
        
#         # 构造logits和标签
#         logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temp
#         labels = torch.zeros(features.size(0), dtype=torch.long, device=device)  # 正样本在0位置
        
#         return F.cross_entropy(logits, labels)


# class SelfContrastiveLoss(nn.Module):
#     def __init__(self, queue_size=512, num_pos=4, num_neg=8, temp=0.07, patch_size=16):
#         super().__init__()
#         self.queue_size = queue_size
#         self.num_pos = num_pos
#         self.num_neg = num_neg
#         self.temp = temp
#         self.patch_size = patch_size
        
#         # 初始化正/负样本队列（存储梯度幅值特征）
#         self.pos_queue = deque(maxlen=queue_size)
#         self.neg_queue = deque(maxlen=queue_size)
        
#         # Sobel算子（固定权重）
#         self.sobel_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
#         self.sobel_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
#         self.sobel_x.weight.data.copy_(torch.tensor([[[[1,0,-1],[2,0,-2],[1,0,-1]]]], dtype=torch.float32))
#         self.sobel_y.weight.data.copy_(torch.tensor([[[[1,2,1],[0,0,0],[-1,-2,-1]]]], dtype=torch.float32))
#         self.sobel_x.requires_grad_(False)
#         self.sobel_y.requires_grad_(False)

#         self.edge = edge()
#         # self.sf = SF().to('cuda')
    
#     def compute_gradient_features(self, x):
#         # 输入x: (N, C, H, W), 输出特征: (N, H*W) [展平后的梯度幅值]
#         grad_x = self.sobel_x(x)
#         grad_y = self.sobel_y(x)
#         grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)  # (N, C, H, W)
#         grad_mag_flat = grad_mag.flatten(start_dim=2)        # (N, C, H*W)
#         return grad_mag_flat.mean(dim=1)                     # (N, H*W) 或直接取均值 (N,)
    
#     def forward(self, fus):
#         B, C, H, W = fus.shape
#         # 分块处理
#         patches = fus.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
#         patches = patches.contiguous().view(B, -1, C, self.patch_size, self.patch_size)
#         patches = patches.view(-1, C, self.patch_size, self.patch_size)  # (B*num_patches, C, 16, 16)
        
#         # 提取梯度特征并确保二维 (N, 256)
#         # grad_x = self.sobel_x(patches)
#         # grad_y = self.sobel_y(patches)
#         # grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
#         grad_mag = self.edge(patches*255.0)
#         features = grad_mag.flatten(start_dim=1)  # (N, 16 * 16) = (N, 256)
#         features = F.normalize(features, dim=-1)   # L2归一化
#         norm_fus = (fus - torch.min(fus)) / (torch.max(fus) - torch.min(fus))
#         # 根据梯度均值选择正负样本std_score = torch.std(x, dim=[1,2,3])
#         # SFScore = self.sf(norm_fus)
#         # SDScore = torch.std(norm_fus*255.0)
#         # EDGEScore = features.mean(dim=-1)  # (N,)
#         # scores = SFScore + SDScore + EDGEScore  # (N,)
#         EDGEScore = features.mean(dim=-1)  # (N,)
#         scores =  EDGEScore 

#         _, topk_idx = torch.topk(scores, self.num_pos)
#         _, botk_idx = torch.topk(-scores, self.num_neg)
#         with torch.no_grad():  # 关键修改：禁止梯度记录
#             features = features.detach()  # 分离特征与计算图
#             pos_features = features[topk_idx]
#             neg_features = features[botk_idx]
        
#         # 更新队列并确保二维
#         self.pos_queue.extend(pos_features.chunk(pos_features.size(0)))  # 分割为[num_pos个 (1, 256)]
#         self.neg_queue.extend(neg_features.chunk(neg_features.size(0)))
        
#         # 队列不足时跳过计算
#         if len(self.pos_queue) < self.num_pos or len(self.neg_queue) < self.num_neg:
#             return torch.tensor(0.0).to(fus.device)
        
#         # 从队列中取出并堆叠为二维张量
#         pos_keys = torch.cat(list(self.pos_queue), dim=0).to(fus.device)  # (queue_size, 256)
#         neg_keys = torch.cat(list(self.neg_queue), dim=0).to(fus.device)  # (queue_size, 256)
        
#         # 随机采样队列中的特征
#         rand_pos = torch.randperm(pos_keys.size(0))[:self.num_pos]
#         rand_neg = torch.randperm(neg_keys.size(0))[:self.num_neg]
#         pos_keys = pos_keys[rand_pos]  # (num_pos, 256)
#         neg_keys = neg_keys[rand_neg]  # (num_neg, 256)
        
#         # 计算相似度（二维矩阵乘法）
#         sim_pos = torch.mm(features, pos_keys.T)  # (B*num_patches, num_pos)
#         sim_neg = torch.mm(features, neg_keys.T)  # (B*num_patches, num_neg)
        
#         # 计算对比损失
#         logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temp
#         labels = torch.zeros(features.size(0), dtype=torch.long).to(fus.device)
#         loss = F.cross_entropy(logits, labels)
#         return loss
    


# class edge(nn.Module):
#     def __init__(self,a = 0.25):
#         super(edge, self).__init__()
#         self.a = a
#         self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
#         self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
#         self.laplacian = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

#         with torch.no_grad():
#             # 赋值卷积核权重
#             self.sobel_x.weight.copy_(torch.tensor([[[[1, 0, -1],
#                                                       [2, 0, -2],
#                                                       [1, 0, -1]]]], dtype=torch.float32))
#             self.sobel_y.weight.copy_(torch.tensor([[[[1, 2, 1],
#                                                       [0, 0, 0],
#                                                       [-1, -2, -1]]]], dtype=torch.float32))
#             self.laplacian.weight.copy_(torch.tensor([[[[0, 1, 0],
#                                                         [1, -4, 1],
#                                                         [0, 1, 0]]]], dtype=torch.float32))
#             # 禁用梯度更新
#             for param in self.parameters():
#                 param.requires_grad = False

#     def forward(self, fus):
#         sobel_fus = torch.sqrt(torch.pow(self.sobel_x(fus), 2) + torch.pow(self.sobel_y(fus), 2)+1e-8)
#         laplacian_fus = self.laplacian(fus)
#         edge_fus = self.a * (sobel_fus + laplacian_fus)
#         return edge_fus 
    


# class FusionContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.1,input_dim=9216):
#         super().__init__()
#         self.temperature = temperature
#         self.feature_extractor = nn.AdaptiveAvgPool2d((1, 1))  # 保持你的特征提取方式
#         self.projection = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128)
#         )
#         self.mae_loss = nn.L1Loss()
#     def forward(self, fusion_images):
#         """
#         Args:
#             fusion_images (Tensor): 形状为 (B², C, H, W) 的融合图像
#         Returns:
#             contrast_loss (Tensor): 对比损失值
#         """

#         B_square = fusion_images.size(0)
#         B = int(math.sqrt(B_square))
        
#         # 特征提取 + 归一化
#         # features = self.projection(fusion_images.view(B_square, -1))
#         features = fusion_images
#         # pdb.set_trace()
#         # features = F.normalize(features, p=2, dim=1)  # 关键：L2归一化
        
#         # 锚点：位置 (0,0) → 索引 0
#         # anchor = features[0]  # (feature_dim,)
        
#         # # 正样本：位置 (B-1, B-1) → 索引 B²-1
#         # positive = features[B_square - 1]  # (feature_dim,)
        
#         # # 负样本：所有非锚点和非正样本的位置
#         # negatives = features[1:-1]  # 排除索引0和B²-1
#         anchor = features[0]
#         positive = features[B_square - 1]
#         negatives = features[1:-1]
#         # pdb.set_trace()
#         # 计算余弦相似度
#         if negatives.size(0) == 0:
#             # 处理空值情况，例如跳过当前损失计算或返回默认值
#             return torch.tensor(0.0)  # 示例：返回零损失
#         sim_pos = self.mae_loss(anchor.unsqueeze(0), positive.unsqueeze(0))  # (1,)
#         sim_neg1 = self.mae_loss(anchor.unsqueeze(0), negatives[0].unsqueeze(0)) 
#         sim_neg2 = self.mae_loss(anchor.unsqueeze(0), negatives[1].unsqueeze(0))
#         sim_neg = (sim_neg1 + sim_neg2)/ 2
#         # pdb.set_trace()
#         # pdb.set_trace()
#         # 计算 InfoNCE 损失
#         numerator = torch.exp(sim_pos / self.temperature)
#         denominator = numerator + torch.exp(sim_neg / self.temperature).sum()
#         contrast_loss = -torch.log(numerator / denominator)
        
#         return contrast_loss

class FusionContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1,input_dim=9216):
        super().__init__()
        self.temperature = temperature
        # self.feature_extractor = nn.AdaptiveAvgPool2d((1, 1))  # 保持你的特征提取方式
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.resize = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        self.preprocess = transforms.Compose([self.resize, self.normalize])
        self.mae_loss = nn.L1Loss()

    def forward(self, fusion_images):
        """
        Args:
            fusion_images (Tensor): 形状为 (B², C, H, W) 的融合图像
        Returns:
            contrast_loss (Tensor): 对比损失值
        """
        B_square = fusion_images.size(0)
        B = int(math.sqrt(B_square))
        
        # 特征提取 + 归一化
        # features = self.projection(fusion_images.view(B_square, -1))
        # pdb.set_trace()
        # features = F.normalize(features, p=2, dim=1)  # 关键：L2归一化
        with torch.no_grad():
            fusion_images = fusion_images.repeat(1, 3, 1, 1)
            fusion_images = self.preprocess(fusion_images)
            features = self.clip_model.encode_image(fusion_images)
        
        # 锚点：位置 (0,0) → 索引 0
        anchor = features[0]  # (feature_dim,)
        
        # 正样本：位置 (B-1, B-1) → 索引 B²-1
        positive = features[B_square - 1]  # (feature_dim,)
        
        # 负样本：所有非锚点和非正样本的位置
        negatives = features[1:-1]  # 排除索引0和B²-1，这么看他不兼容B大于2的情况，但我B是2，问题不在这里

        # 计算余弦相似度
        if negatives.size(0) == 0:
            # 处理空值情况，例如跳过当前损失计算或返回默认值
            return torch.tensor(0.0)  # 示例：返回零损失
        sim_pos = self.mae_loss(anchor.unsqueeze(0), positive.unsqueeze(0))  # (1,)
        sim_neg1 = self.mae_loss(anchor.unsqueeze(0), negatives[0].unsqueeze(0)) 
        sim_neg2 = self.mae_loss(anchor.unsqueeze(0), negatives[1].unsqueeze(0))
        sim_neg = (sim_neg1 + sim_neg2)/ 2
        # 计算 InfoNCE 损失
        numerator = torch.exp(sim_pos / self.temperature)
        denominator = numerator + torch.exp(sim_neg / self.temperature).sum()
        contrast_loss = -torch.log(numerator / denominator)

        return contrast_loss
    

class FusionContrastiveLossV2(nn.Module):
    def __init__(self, temperature=0.1, patch_size=16):
        super().__init__()
        self.temperature = temperature
        self.patch_size = patch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # CLIP 图像预处理
        self.clip_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                (0.26862954, 0.26130258, 0.27577711))
        ])
        
        # 加载CLIP模型
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()

    def split_into_patches(self, x):
        """将图像分割为指定大小的块"""
        B, C, H, W = x.shape
        # 展开为块
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, -1, C, self.patch_size, self.patch_size)
        return x  # (B, num_patches, C, 16, 16)

    def select_samples(self, patches):
        """选择锚点、正样本和负样本"""
        B, num_patches, C, H, W = patches.shape
        grid_size = int(math.sqrt(num_patches))  # 6x6网格
        
        anchors, positives, negatives = [], [], []
        
        for i in range(B):
            # 计算块方差选择锚点
            patch_vars = patches[i].view(num_patches, -1).var(dim=1)
            anchor_idx = patch_vars.argmax()
            
            # 计算PSNR选择正样本
            anchor_patch = patches[i, anchor_idx]
            mse = ((patches[i] - anchor_patch.unsqueeze(0)) ** 2).mean(dim=(1,2,3))
            psnr = 20 * torch.log10(torch.tensor(1.0, device=mse.device)) - 10 * torch.log10(mse + 1e-8)  # MAX=1.0
            psnr[anchor_idx] = -float('inf')  # 排除锚点
            positive_idx = psnr.argmax()
            
            # 选择副对角线作为负样本
            neg_indices = [row*grid_size + (grid_size-1-row) for row in range(grid_size)]
            neg_patches = patches[i, neg_indices]
            
            anchors.append(anchor_patch)
            positives.append(patches[i, positive_idx])
            negatives.append(neg_patches)
            
        return torch.stack(anchors), torch.stack(positives), torch.cat(negatives)
        
        
    #     anchors, positives, negatives = [], [], []
            
    #     anchors, positives, negatives = [], [], []
            
    #         anchors.append(anchor_patch)
    #         positives.append(positive_patches)  # 每个元素是两个patch
    #         negatives.append(neg_patches)
        
    #     return (
    #         torch.stack(anchors),          # [B, C, H, W]
    #         torch.stack(positives),        # [B, 2, C, H, W]
    #         torch.cat(negatives)           # [B * grid_size, C, H, W]
    #     )

    #         anchors.append(anchor_patch)
    #         positives.append(positive_patches)  # 每个元素是两个patch
    #         negatives.append(neg_patches)
        
    #     return (
    #         torch.stack(anchors),          # [B, C, H, W]
    #         torch.stack(positives),        # [B, 2, C, H, W]
    #         torch.cat(negatives)           # [B * grid_size, C, H, W]
    #     )
    def extract_features(self, patches):
        """使用CLIP提取块特征"""
        # 转换到3通道并预处理
        if patches.size(1) == 1:
            patches = patches.repeat(1, 3, 1, 1)
        patches = self.clip_preprocess(patches)
        
        with torch.no_grad():
            features = self.clip_model.encode_image(patches)
        return F.normalize(features.float(), p=2, dim=1)

    def forward(self, fusion_images):
        """对比损失计算流程"""
        # 1. 分割为块
        patches = self.split_into_patches(fusion_images)
        
        # 2. 样本选择
        anchors, positives, negatives = self.select_samples(patches)
        
        # 3. 特征提取
        anchor_feats = self.extract_features(anchors)
        positive_feats = self.extract_features(positives)
        negative_feats = self.extract_features(negatives)
        
        # 4. 损失计算
        losses = []
        for a, p in zip(anchor_feats, positive_feats):
            # 正样本相似度
            sim_pos = a @ p.T / self.temperature
            
            # 负样本相似度（同一batch的所有负样本）
            sim_neg = a @ negative_feats.T / self.temperature
            
            # InfoNCE损失
            numerator = torch.exp(sim_pos)
            denominator = numerator + torch.exp(sim_neg).sum()
            losses.append(-torch.log(numerator / denominator))
            
        return torch.mean(torch.stack(losses))


class PATS(nn.Module):
    def __init__(self, temperature=0.1, patch_size=32):
        super().__init__()
        self.temperature = temperature
        self.patch_size = patch_size
        
        # CLIP模型初始化
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        for parameter in self.clip_model.parameters():
            parameter.requires_grad_(False)
        
        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                (0.26862954, 0.26130258, 0.27577711))
        ])

    def split_into_patches(self, x):
        """将图像分割为指定大小的块"""
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, -1, C, self.patch_size, self.patch_size)
        return x  # (B, num_patches, C, 16, 16)

    def select_samples(self, all_images):
        """
        输入: 
        all_images - 形状 (4, C, H, W) 的张量，按顺序包含：
        [同位融合图像1, 错位融合图像1, 同位融合图像2, 错位融合图像2]
        """
        # 分割所有图像的块
        all_patches = self.split_into_patches(all_images)  # (4, 9, C, 16, 16)
        
        # 从图像1选择锚点
        img1_patches = all_patches[0]  # (9, C, 16, 16)
        img1_var = img1_patches.view(9, -1).var(dim=1)
        anchor_idx = img1_var.argmax()
        anchor_patch = img1_patches[anchor_idx]  # (C, 16, 16)

        # 从另一幅同位融合图像选择像素级相似的正样本
        positive_patches = all_patches[2]
        mse = ((positive_patches - anchor_patch.unsqueeze(0)) ** 2).mean(dim=(1,2,3))
        psnr = 20 * torch.log10(torch.tensor(1.0, device=mse.device)) - 10 * torch.log10(mse + 1e-8)
        psnr[anchor_idx] = -float("inf")
        num_positives = min(2, psnr.numel() - 1)
        positive_idx = psnr.topk(num_positives).indices
        positive_patch = positive_patches[positive_idx]

        # 错位融合结果中的 patch 是 PATS 的伪融合负样本候选
        mismatched_patches = torch.cat([all_patches[1], all_patches[3]], dim=0)

        return anchor_patch, positive_patch, mismatched_patches

    def extract_features(self, patches):
        """特征提取（支持单通道输入）"""
        batch_size = patches.size(0)
        features = []
        
        for i in range(batch_size):
            patch = patches[i]
            if patch.size(0) == 1:  # 单通道转三通道
                patch = patch.repeat(3, 1, 1)
            
            processed = self.preprocess(patch.unsqueeze(0).to(self.device))
            feat = self.clip_model.encode_image(processed)
            features.append(feat.squeeze().float())  # 强制使用float32
        
        return torch.stack(features)  # (N, 512)

    def forward(self, x):

        # 样本选择
        anchor, positive, negatives = self.select_samples(x)
        
        # 特征提取
        anchor = self.extract_features(anchor.unsqueeze(0))  # (1, 512)
        positive = self.extract_features(positive)
        negatives = self.extract_features(negatives)  # (9, 512)
        
        # 计算相似度（使用余弦相似度）
        sim_pos = F.cosine_similarity(anchor, positive) / self.temperature
        sim_neg = F.cosine_similarity(anchor, negatives) / self.temperature
        num_negatives = min(8, sim_neg.numel())
        hard_negative_idx = sim_neg.topk(num_negatives, largest=False).indices
        sim_neg = sim_neg[hard_negative_idx]
        
        # 数值稳定的损失计算
        max_sim = torch.max(sim_pos.max(), sim_neg.max()).detach()
        logits = torch.cat([sim_pos, sim_neg])
        return -torch.log_softmax(logits - max_sim, dim=0)[:sim_pos.numel()].mean()
