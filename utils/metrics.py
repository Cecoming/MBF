import torch
import numpy as np
import os
from utils.reranking import re_ranking
from utils.test_tsen import estimate_tsen
from utils.test_PCA import pca

def weighted_euclidean_distance(qf, gf, q_vis, g_vis, feature_dim):
    query_num, total_feature_dim = qf.shape
    assert total_feature_dim % feature_dim == 0
    num_features = total_feature_dim // feature_dim
    
    final_dist_mat = torch.zeros(query_num, gf.shape[0])
    
    vis_part = torch.zeros(query_num, gf.shape[0]) + 10e-12
    # part distance
    for i in range(0, num_features):
        start_idx = i * feature_dim
        end_idx = start_idx + feature_dim

        scaled_qf = qf[:, start_idx:end_idx]
        scaled_gf = gf[:, start_idx:end_idx]

        vis_temp = q_vis[:, i:i+1] * g_vis[:, i:i+1].T
        vis_part += vis_temp
        
        dist_mat = torch.from_numpy(euclidean_distance(scaled_qf, scaled_gf)) * vis_temp
        final_dist_mat += dist_mat

    final_dist_mat = final_dist_mat / vis_part
    
    # # gloabl distance
    # qf_global = qf[:, :feature_dim]
    # gf_global = gf[:, :feature_dim]
    # final_dist_mat += euclidean_distance(qf_global, gf_global)
        
    return final_dist_mat

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()   # 两个矩阵的平方和[2210, 17661]
    dist_mat.addmm_(1, -2, qf, gf.t())  # 矩阵乘法
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

# motified
class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False, use_visible_matrx=False, feature_dim=768):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.use_visible_matrx = use_visible_matrx
        self.feature_dim = feature_dim

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.visible_matrxs = []

    def update(self, output):  # called once for each batch
        if self.use_visible_matrx:
            feat, pid, camid, visible_matrx = output
            self.visible_matrxs.append(visible_matrx.cpu())
        else:
            feat, pid, camid = output
        
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def tsne(self, path=None):
        feats = torch.cat(self.feats, dim=0).numpy()
        gf = feats
        g_pids = np.asarray(self.pids)
        mask = (g_pids >= 20) & (g_pids < 60)
        filtered_g_pids = g_pids[mask]
        filtered_gf = gf[mask]
        datas, _= pca(filtered_gf, 30)
        X = datas.real
        Y = estimate_tsen(X,filtered_g_pids, 2, 30, plot=True, path=path)

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        if self.use_visible_matrx:
            vis = torch.cat(self.visible_matrxs, dim=0)
            q_vis = vis[:self.num_query]
            g_vis = vis[self.num_query:]

        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        else:
            if self.use_visible_matrx:
                print('=> Computing DistMat with euclidean_distance')
                distmat = weighted_euclidean_distance(qf, gf, q_vis, g_vis, self.feature_dim)
                # distmat = euclidean_distance(qf, gf)
            else:
                print('=> Computing DistMat with euclidean_distance')
                distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf

    # def compute(self):  # called after each epoch
    #     feats = torch.cat(self.feats, dim=0)    # [19871, 3840]
    #     if self.feat_norm:
    #         print("The test feature is normalized")
    #         feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel, p=2指L2范数L2 范数归一化会将每一个特征向量除以其自身的 L2 范数（也就是其元素平方和的平方根
    #     # query
    #     qf = feats[:self.num_query]
    #     q_pids = np.asarray(self.pids[:self.num_query])
    #     q_camids = np.asarray(self.camids[:self.num_query])
    #     # gallery
    #     gf = feats[self.num_query:]
    #     g_pids = np.asarray(self.pids[self.num_query:])

    #     g_camids = np.asarray(self.camids[self.num_query:])
    #     if self.reranking:
    #         print('=> Enter reranking')
    #         # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
    #         distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

    #     else:
    #         print('=> Computing DistMat with euclidean_distance')
    #         distmat = euclidean_distance(qf, gf)
    #     cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

    #     return cmc, mAP, distmat, self.pids, self.camids, qf, gf

