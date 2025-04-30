from __future__ import print_function
from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class GraspPointAppGroup(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        graspgroup_mlp_specs: Union[List[int], int] = None,
        group_mlp_specs: Union[List[int], int] = None,
    ):
        super().__init__()
        self.in_ch = in_ch
        assert in_ch >= 3
        if graspgroup_mlp_specs is None:
            graspgroup_mlp_specs = [16, 32]
        if group_mlp_specs is None:
            group_mlp_specs = [64, 128]
        if type(graspgroup_mlp_specs) is not list:
            graspgroup_mlp_specs = [graspgroup_mlp_specs]
        if type(group_mlp_specs) is not list:
            group_mlp_specs = [group_mlp_specs]
        self.graspgroup_featnum = graspgroup_mlp_specs[-1]
        self.group_featnum = group_mlp_specs[-1]

        self.grasp_conv1 = torch.nn.Conv1d(in_ch, graspgroup_mlp_specs[0], 1)
        self.grasp_norm1 = nn.LayerNorm(graspgroup_mlp_specs[0], eps=1e-6)
        self.grasp_conv2 = torch.nn.Conv1d(graspgroup_mlp_specs[0], graspgroup_mlp_specs[1], 1)
        self.grasp_norm2 = nn.LayerNorm(graspgroup_mlp_specs[1], eps=1e-6)

        self.group_conv1 = torch.nn.Conv1d(3 + graspgroup_mlp_specs[-1] * 2, group_mlp_specs[0], 1)
        self.group_norm1 = nn.LayerNorm(group_mlp_specs[0], eps=1e-6)
        self.group_conv2 = torch.nn.Conv1d(group_mlp_specs[0], group_mlp_specs[1], 1)
        self.group_norm2 = nn.LayerNorm(group_mlp_specs[1], eps=1e-6)

    # 아마도
    # x_init: 20개 gaussian points, x: orig grasp moved with x_init
    def forward(self, x_init, x):
        # x size: (B, N, K, 3 + C), x_init size: (B, K, 3 + C)
        # (1,40,20,3), (1,20,3)
        bat, n_pts, k_pts, feat_num = x.size()
        intra_grasp_feat = torch.cat([x_init.unsqueeze(1), x], dim=1)  # (B, N+1, K, 3+C)
        
        # 원래 grasp points에 gaussian 20개 더한게 x고, 다시 그걸 20개에 대해 평균낸게 grasp_center
        grasp_center = torch.mean(x[..., :3], dim=2)  # (B, N, 3)

        x = intra_grasp_feat.view(bat, (n_pts+1) * k_pts, feat_num).transpose(2, 1).contiguous()  # (B, 3+C, N+1 * K)
        x = F.relu(self.grasp_norm1(self.grasp_conv1(x).transpose(2, 1).contiguous()))  # (B, N+1 * K, gg[0])
        x = x.transpose(2, 1).contiguous()  # (B, gg[0], N+1 * K)
        x = F.relu(self.grasp_norm2(self.grasp_conv2(x).transpose(2, 1).contiguous()))  # (B, N+1 * K, gg[1])
        x = x.view(-1, n_pts+1, k_pts, self.graspgroup_featnum)  # (B, N+1, K, gg[-1])
        #(1, 41, 20, 32) -> (1, 41, 32) (가우시안 더해서 20개로 뿔렸던거 다시 max하나만 남기도록 pool)
        x = torch.max(x, 2)[0]  # (B, N+1, gg[-1])
        x = torch.cat([x[:, 1:], x[:, 0].unsqueeze(1).repeat(1, n_pts, 1), grasp_center], 2)  # (B, N, 2 * gg[-1]+3)

        x = x.transpose(2, 1).contiguous()  # (B, 2*gg[-1]+3, N)
        x = F.relu(self.group_norm1(self.group_conv1(x).transpose(2, 1).contiguous()))  # (B, N，g[0])
        x = x.transpose(2, 1).contiguous()  # (B, g[0], N)
        x = F.relu(self.group_norm2(self.group_conv2(x).transpose(2, 1).contiguous()))  # (B, N，g[1])
        x = x.transpose(2, 1).contiguous()  # (B, g[1], N)
        # (1, 256, 40) -> (1, 256) (40개의) grasp 중 max값인거 선택
        x = torch.max(x, 2)[0]  # (B, g[-1])
        return x


class PointNetfeat(nn.Module):
    def __init__(self, in_ch=3, mlp_specs=None, global_feat=True, xyz_transform=False, feature_transform=False):
        super(PointNetfeat, self).__init__()
        if mlp_specs is None:
            mlp_specs = [64, 128, 512]
        assert in_ch >= 3
        self.in_ch = in_ch
        self.mlp_specs = mlp_specs
        self.global_feat = global_feat
        self.xyz_transform = xyz_transform
        if self.xyz_transform:
            self.stn = STN3d()
        # self.fc = nn.Sequential(
        #     nn.Linear(mlp_specs[2], 256),
        #     nn.Linear(256, 128)
        # )
        self.conv1 = torch.nn.Conv1d(in_ch, mlp_specs[0], 1)
        self.conv2 = torch.nn.Conv1d(mlp_specs[0], mlp_specs[1], 1)
        self.conv3 = torch.nn.Conv1d(mlp_specs[1], mlp_specs[2], 1)
        # self.bn1 = nn.BatchNorm1d(mlp_specs[0])
        # self.bn2 = nn.BatchNorm1d(mlp_specs[1])
        # self.bn3 = nn.BatchNorm1d(mlp_specs[2])
        self.norm1 = nn.LayerNorm(mlp_specs[0], eps=1e-6)
        self.norm2 = nn.LayerNorm(mlp_specs[1], eps=1e-6)
        self.norm3 = nn.LayerNorm(mlp_specs[2], eps=1e-6)
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        if self.xyz_transform:
            if self.in_ch > 3:
                xyz, feat = x[:, :3], x[:, 3:]
                trans = self.stn(xyz)
                xyz = xyz.transpose(2, 1).contiguous()
                xyz = torch.bmm(xyz, trans)
                xyz = xyz.transpose(2, 1).contiguous()
                x = torch.cat((xyz, feat), dim=1)
            else:
                trans = self.stn(x)
                x = x.transpose(2, 1).contiguous()
                x = torch.bmm(x, trans)
                x = x.transpose(2, 1).contiguous()
        else:
            trans = None
            x = F.relu(self.norm1(self.conv1(x).transpose(2, 1).contiguous()))
            x = x.transpose(2, 1).contiguous()

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1).contiguous()
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1).contiguous()
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.norm2(self.conv2(x).transpose(2, 1).contiguous()))
        x = x.transpose(2, 1).contiguous()
        x = self.norm3(self.conv3(x).transpose(2, 1).contiguous())
        x = x.transpose(2, 1).contiguous()
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.mlp_specs[2])
        if self.global_feat:
            # x = self.fc(x)
            return x, trans, trans_feat
        else:
            x = x.view(-1, self.mlp_specs[2], 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
        # global_x = self.fc(x)
        # cat_x = torch.cat([x.view(-1, self.mlp_specs[2], 1).repeat(1, 1, n_pts), pointfeat], 1)
        # return global_x, cat_x, trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32, 3, 2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat()
    global_x, cat_x, _, _ = pointfeat(sim_data)
    print('global feat, point feat: ', global_x.size(), cat_x.size())

    sim_data = torch.randn(2, 40, 6, 3)
    gpag = GraspPointAppGroup(
        in_ch=3,
        graspgroup_mlp_specs=[16, 32],
        group_mlp_specs=[64, 256],
    )
    g_feat = gpag(sim_data)
    print(g_feat.shape)
