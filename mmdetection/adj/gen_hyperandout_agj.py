import pickle
import numpy as np
import torch

def gen():

    CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                   'cat', 'chair', 'cow', 'diningtable', 'horse', 'motorbike',
                   'person', 'pottedplant', 'sheep', 'tvmonitor',
                   'car', 'dog', 'sofa', 'train']

    class2hyper = { 'background': 'background',
        'aeroplane': 'vihicles', 'bicycle': 'vihicles',
        'bird': 'animals', 'boat': 'vihicles', 'bottle': 'household', 'bus': 'vihicles',
                   'cat': 'animals', 'chair': 'household', 'cow': 'animals',
                   'diningtable': 'household', 'horse': 'animals', 'motorbike': 'vihicles',
                   'person': 'person', 'pottedplant': 'household', 'sheep': 'animals', 'tvmonitor': 'household',
                   'car': 'animals', 'dog': 'animals', 'sofa': 'household', 'train': 'vihicles'}

    cate2class = {'4-wheeled': ['car', 'bus'], '2-wheeled': ['bicycle', 'motorbike'], 'furniture': ['chair', 'sofa', 'dining table'], 'domestic': ['cat', 'dog'], 'farmyard': ['cow', 'horse', 'sheep']}
    cate3class = {'seating': ['chair', 'sofa']}

    class2cate = dict()
    class3cate = dict()

    for i in cate2class:
        for j in cate2class[i]:
            class2cate[j] = i

    for i in cate3class:
        for j in cate3class[i]:
            class3cate[j] = i

    out_adj = [[0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
    [0,	0,	100,	25,	100,	0,	100,	0,	50,	0,	40,	0,	100,	100,	0,	0,	30,	0,	0,	0,	100],
    [0,	100,	0,	0,	100,	10,	100,	0,	0,	0,	0,	25,	200,	300,	0,	0,	0,	0,	0,	0,	100],
    [0,	25,	0,	0,	0,	0,	0,	100,	0,	100,	0,	100,	0,	10,	0,	100,	0,	0,	100,	0,	0],
    [0,	100,	100,	0,	0,	0,	100,	0,	10,	0,	5,	0,	100,	200,	0,	0,	0,	100,	0,	0,	100],
    [0,	0,	10,	0,	0,	0,	0,	0,	100,	0,	300,	0,	0,	200,	0,	0,	100,	0,	0,	100,	0],
    [0,	100,	100,	0,	100,	0,	0,	0,	50,	0,	0,	0,	100,	300,	0,	0,	10,	200,	0,	0,	100],
    [0,	0,	0,	100,	0,	0,	0,	0,	5,	100,	0,	100,	0,	300,	0,	100,	0,	0,	200,	5,	0],
    [0,	50,	0,	0,	10,	100,	50,	5,	0,	0,	200,	0,	0,	300,	100,	0,	100,	0,	0,	100,	0],
    [0,	0,	0,	100,	0,	0,	0,	100,	0,	0,	0,	200,	0,	200,	0,	200,	0,	0,	100,	0,	0],
    [0,	40,	0,	0,	5,	300,	0,	0,	200,	0,	0,	0,	0,	200,	200,	0,	200,	0,	0,	200,	0],
    [0,	0,	25,	100,	0,	0,	0,	100,	0,	200,	0,	0,	25,	300,	0,	200,	0,	100,	0,	0,	100],
    [0,	100,	200,	0,	100,	0,	100,	0,	0,	0,	0,	25,	0,	300,	0,	0,	0,	100,	0,	0,	100],
    [0,	100,	300,	10,	200,	200,	300,	300,	300,	200,	200,	300,	300,	0,	100,	100,	300,	200,	300,	300,	100],
    [0,	0,	0,	0,	0,	0,	0,	0,	100,	0,	200,	0,	0,	100,	0,	0,	100,	0,	0,	100,	0],
    [0,	0,	0,	100,	0,	0,	0,	100,	0,	200,	0,	200,	0,	100,	0,	0,	0,	0,	100,	0,	0],
    [0,	30,	0,	0,	0,	100,	10,	0,	100,	0,	200,	0,	0,	300,	100,	0,	0,	0,	0,	110,	0],
    [0,	0,	0,	0,	100,	0,	200,	0,	0,	0,	0,	100,	100,	200,	0,	0,	0,	0,	0,	0,	100],
    [0,	0,	0,	100,	0,	0,	0,	200,	0,	100,	0,	0,	0,	300,	0,	100,	0,	0,	0,	0,	0],
    [0,	0,	0,	0,	0,	100,	0,	5,	100,	0,	200,	0,	0,	300,	100,	0,	110,	0,	0,	0,	0],
    [0,	100,	100,	0,	100,	0,	100,	0,	0,	0,	0,	100,	100,	100,	0,	0,	0,	100,	0,	0,	0]]

    out_adj = np.array(out_adj)

    hyp_adj = np.zeros_like(out_adj)

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i != j:
                aclass = CLASSES[i]
                bclass = CLASSES[j]

                if class2hyper[aclass] == class2hyper[bclass]:
                    hyp_adj[i][j] += 100
                if aclass in class2cate and bclass in class2cate:
                    if class2cate[aclass] == class2cate[bclass]:
                        hyp_adj[i][j] += 100
                if aclass in class3cate and bclass in class3cate:
                    if class3cate[aclass] == class3cate[bclass]:
                        hyp_adj[i][j] += 100

    # print(hyp_adj)

    # if np.transpose(out_adj) == out_adj:
        # print('true')

    # print(np.allclose(hyp_adj, hyp_adj.T, rtol=1e-05, atol=1e-08))

    with open('voc_hyp_adj.pkl', 'wb') as f:
        pickle.dump(hyp_adj,f)

    with open('voc_out_adj.pkl', 'wb') as f:
        pickle.dump(out_adj,f)

def adapt():

    # with open('voc_adj.pkl', 'rb') as f:
    #     a = pickle.load(f)
    #
    # a['nums'] = np.insert(a['nums'], 0, 1)
    # a['adj'] = np.insert(a['adj'], 0, [0]*20, axis=1)
    # a['adj'] = np.insert(a['adj'], 0, [0]*21, axis=0)

    # print(a)

    with open('voc_hyp_adj.pkl', 'rb') as f:
        b = pickle.load(f)

    b = b['adj']

    b[13][1:] = np.array([50] * 20)

    # print(b[13])
    # exit()

    real_b = dict()
    real_b['nums'] = np.max(b, axis=0)
    real_b['nums'][0] = 1
    real_b['adj'] = b



    # print(real_b)
    #
    # with open('voc_out_adj.pkl', 'rb') as f:
    #     c = pickle.load(f)
    #
    # real_c = dict()
    # real_c['nums'] = np.max(c, axis=0)
    # real_c['nums'][0] = 1
    # real_c['adj'] = c
    #
    # print(real_c)
    #
    # with open('voc_adj.pkl', 'wb') as f:
    #     pickle.dump(a,f)

    with open('voc_hyp_adj.pkl', 'wb') as f:
        pickle.dump(real_b,f)

    # with open('voc_out_adj.pkl', 'wb') as f:
    #     pickle.dump(real_c,f)

def inspect():
    with open('voc_adj.pkl', 'rb') as f:
        a = pickle.load(f)

    with open('voc_hyp_adj.pkl', 'rb') as f:
        b = pickle.load(f)

    with open('voc_out_adj.pkl', 'rb') as f:
        c = pickle.load(f)

    print(b)

    print(gen_A(21, 0.4, b))

def gen_A(num_classes, t, result):
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = torch.Tensor(_adj) + torch.Tensor(np.identity(num_classes, np.int))
    return _adj

if __name__ == '__main__':
    # gen()
    adapt()
    inspect()