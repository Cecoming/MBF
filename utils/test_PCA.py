import numpy as np
import matplotlib.pyplot as plt
# data 输入数据  维度 [N，D]
# n_dim: 降维后的维度
# 返回 [N,n_dim]
def pca(data, n_dim):
    
    N,D = np.shape(data)
    
    data = data - np.mean(data, axis = 0, keepdims = True)

    C = np.dot(data.T, data)/(N-1)  # [D,D]
    
    # 计算特征值和特征向量
    eig_values, eig_vector = np.linalg.eig(C)
    
    # 将特征值进行排序选取 n_dim 个较大的特征值
    indexs_ = np.argsort(-eig_values)[:n_dim]
    
    # 选取相应的特征向量组成降维矩阵
    picked_eig_vector = eig_vector[:, indexs_] # [D,n_dim]
    
    # 对数据进行降维
    data_ndim = np.dot(data, picked_eig_vector)
    return data_ndim, picked_eig_vector

def draw_pic(datas,labs):
    plt.cla()
    unque_labs = np.unique(labs)
    colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1,len(unque_labs))]
    p=[]
    legends = []
    for i in range(len(unque_labs)):
        index = np.where(labs==unque_labs[i])
        pi = plt.scatter(datas[index, 0], datas[index, 1], c =[colors[i]] )
        p.append(pi)
        legends.append(unque_labs[i])
    
    plt.legend(p, legends)
    plt.show()
    
    
    
    


if __name__ == "__main__":
    
    # 加载数据
    data = np.loadtxt("iris.data",dtype="str",delimiter=',')
    feas = data[:,:-1]
    feas = np.float32(feas)
    labs = data[:,-1]
    
    # 进行降维
    data_2d, picked_eig_vector= pca(feas, 2)
    
    
    #绘图
    draw_pic(data_2d,labs)
    
    
   
    
    
    
    
    
    
    
    