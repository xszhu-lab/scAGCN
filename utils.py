import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

import tensorflow as tf
# import tensorflow.keras.backend as K
import collections
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.manifold import TSNE

# MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-4, 1e4)
# DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)

MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-3, 1e4)

# based on  https://github.com/XifengGuo/IDEC-toy/blob/master/DEC.py
# def cluster_acc(y_true, y_pred):
#     """
#     Calculate clustering accuracy. Require scikit-learn installed
#     # Arguments
#         y: true labels, numpy.array with shape `(n_samples,)`
#         y_pred: predicted labels, numpy.array with shape `(n_samples,)`
#     # Return
#         accuracy, in [0,1]
#     """
#     y_true = y_true.astype(np.int64)
#     assert y_pred.size == y_true.size
#     D = max(y_pred.max(), y_true.max()) + 1
#     w = np.zeros((D, D), dtype=np.int64)
#     for i in range(y_pred.size):
#         w[y_pred[i], y_true[i]] += 1
#     ind = linear_sum_assignment(w.max() - w)
def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    # ind = linear_assignment(w.max() - w)
    # return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    """
    DeprecationWarning: The linear_assignment function 
    is deprecated in 0.21 and will be removed from 0.23. Use scipy.optimize.linear_sum_assignment instead.
    """

    indices = linear_sum_assignment(w.max() - w)
    indices = np.asarray(indices)
    indices = np.transpose(indices)
    return sum([w[i, j] for i, j in indices]) * 1.0 / y_pred.size

def show_tSNE(latent, name, tSNE=True, labels=None, cmap=plt.get_cmap("tab20", 15)):
    
    if tSNE:
        tsne = TSNE().fit_transform(latent)
        
    else:
        tsne = latent

    classes = np.unique(labels)
    #tab10, tab20, husl, hls
    if cmap is not None:
        cmap = cmap
    elif len(classes) <= 10:
        cmap = 'tab10'
    elif len(classes) <= 20:
        cmap = 'tab20'

    if len(classes) > 20:
        cmap = 'hsv'

    plt.figure(figsize=(10, 10))

    #隐藏边框线
    ax1=plt.gca()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # ax1.spines['bottom'].set_visible(False)
    # ax1.spines['left'].set_visible(False)
    # ax1.xaxis.set_major_locator(plt.NullLocator())
    # ax1.yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)

    plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, \
                                   cmap=cmap, edgecolors='none')


    plt.tick_params(labelsize=30) #刻度字体大小13
    # plt.axis("off")
    # plt.title("tSNE for latent space " + name, fontsize=16)

    # plt.savefig(name + ".png", dpi=300, pad_inches = 0)
    
    # plt.savefig(name, format='png', transparent=False, dpi=300, pad_inches = 0)
    
    plt.savefig(name + ".png", format='png', bbox_inches='tight',dpi=400)

    return tsne

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__




def _nan2zero(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

def _nan2inf(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x)+np.inf, x)

def _nelem(x):
    nelem = tf.reduce_sum(tf.cast(~tf.math.is_nan(x), tf.float32))
    return tf.cast(tf.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)

def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return tf.divide(tf.reduce_sum(x), nelem)

# def NB(theta, y_true, y_pred, mask = False, debug = True, mean = False):


#     eps = 1e-10
#     scale_factor = 1.0
#     y_true = tf.cast(y_true, tf.float32)
#     y_pred = tf.cast(y_pred, tf.float32) * scale_factor

#     # y_pred =tf.maximum(y_pred, 1e-15) ####

#     # y_pred = tf.clip_by_value(y_pred, 1e-15, 1e4)  #####
#     if mask:
#         nelem = _nelem(y_true)
#         y_true = _nan2zero(y_true)

#     # print(y_true.numpy())
#     # print(y_pred.numpy())
#     # print(theta.numpy())


#     theta = tf.minimum(theta, 1e6)
#     # theta =tf.maximum(theta, 1e-15) ####
#     # y_true =tf.maximum(y_true, 1e-15)###
#     # y_true = tf.minimum(y_true, 0.1)
    
#     # theta = tf.clip_by_value(theta, 1e-6, 1e6)  #####
#     t1 = tf.math.lgamma(theta + eps) + tf.math.lgamma(y_true + 1.0) - tf.math.lgamma(y_true + theta + eps)
#     t2 = (theta + y_true) * tf.math.log(1.0 + (y_pred / (theta + eps))) + (y_true * (tf.math.log(theta + eps) - tf.math.log(y_pred + eps)))
#     if debug:
#         assert_ops = [tf.compat.v1.verify_tensor_all_finite(y_pred, 'y_pred has inf/nans'),
#                       tf.compat.v1.verify_tensor_all_finite(t1, 't1 has inf/nans'),
#                       tf.compat.v1.verify_tensor_all_finite(t2, 't2 has inf/nans')]
#         with tf.control_dependencies(assert_ops):
#             final = t1 + t2
#     else:
#         final = t1 + t2
#     final = _nan2inf(final)
#     if mean:
#         if mask:
#             final = tf.divide(tf.reduce_sum(final), nelem)
#         else:
#             final = tf.reduce_mean(final)
#     return final


# # def ZINB(x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
# #         eps = 1e-10
# #         scale_factor = scale_factor[:, None]
# #         mean = mean * scale_factor
        
# #         t1 = tf.math.lgamma(disp+eps) + tf.math.lgamma(x+1.0) - tf.math.lgamma(x+disp+eps)
# #         t2 = (disp+x) * tf.math.log(1.0 + (mean/(disp+eps))) + (x * (tf.math.log(disp+eps) - tf.math.log(mean+eps)))
# #         nb_final = t1 + t2

# #         nb_case = nb_final - tf.math.log(1.0-pi+eps)
# #         zero_nb = tf.math.pow(disp/(disp+mean+eps), disp)
# #         zero_case = -tf.math.log(pi + ((1.0-pi)*zero_nb)+eps)
# #         result = tf.where(tf.less(x, 1e-8), zero_case, nb_case)
        
# #         if ridge_lambda > 0:
# #             ridge = ridge_lambda*tf.square(pi)
# #             result += ridge
        
# #         result = tf.mean(result)
# #         return result

# # https://github.com/ttgump/ZINBAE/blob/master/ZINBAE.py
# def ZINB(pi, theta, y_true, y_pred, ridge_lambda, mean = True, mask = False, debug = False):
#     eps = 1e-10
#     scale_factor = 1.0
#     nb_case = NB(theta, y_true, y_pred, mean=False, debug=debug) - tf.math.log(1.0 - pi + eps)
#     y_true = tf.cast(y_true, tf.float32)
#     y_pred = tf.cast(y_pred, tf.float32) * scale_factor
#     theta = tf.minimum(theta, 1e6)

#     zero_nb = tf.math.pow(theta / (theta + y_pred + eps), theta)
#     zero_case = -tf.math.log(pi + ((1.0 - pi) * zero_nb) + eps)
#     result = tf.where(tf.less(y_true, 1e-8), zero_case, nb_case)
#     ridge = ridge_lambda * tf.math.square(pi)
#     result += ridge
#     if mean:
#         if mask:
#             result = _reduce_mean(result)
#         else:
#             result = tf.reduce_mean(result)

#     result = _nan2inf(result)
#     return result

def NB(theta, y_true, y_pred, mask = False, debug = False, mean = False):
    eps = 1e-10
    scale_factor = 1.0
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32) * scale_factor
    if mask:
        nelem = _nelem(y_true)
        y_true = _nan2zero(y_true)
    theta = tf.minimum(theta, 1e6)
    t1 = tf.math.lgamma(theta + eps) + tf.math.lgamma(y_true + 1.0) - tf.math.lgamma(y_true + theta + eps)
    t2 = (theta + y_true) * tf.math.log(1.0 + (y_pred / (theta + eps))) + (y_true * (tf.math.log(theta + eps) - tf.math.log(y_pred + eps)))
    if debug:
        assert_ops = [tf.verify_tensor_all_finite(y_pred, 'y_pred has inf/nans'),
                      tf.verify_tensor_all_finite(t1, 't1 has inf/nans'),
                      tf.verify_tensor_all_finite(t2, 't2 has inf/nans')]
        with tf.control_dependencies(assert_ops):
            final = t1 + t2
    else:
        final = t1 + t2
    final = _nan2inf(final)
    if mean:
        if mask:
            final = tf.divide(tf.reduce_sum(final), nelem)
        else:
            final = tf.reduce_mean(final)
    return final

def ZINB(pi, theta, y_true, y_pred, ridge_lambda, mean = True, mask = False, debug = False):
    eps = 1e-10
    scale_factor = 1.0
    nb_case = NB(theta, y_true, y_pred, mean=False, debug=debug) - tf.math.log(1.0 - pi + eps)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32) * scale_factor
    theta = tf.minimum(theta, 1e6)

    zero_nb = tf.math.pow(theta / (theta + y_pred + eps), theta)
    zero_case = -tf.math.log(pi + ((1.0 - pi) * zero_nb) + eps)
    result = tf.where(tf.less(y_true, 1e-8), zero_case, nb_case)
    ridge = ridge_lambda * tf.math.square(pi)
    result += ridge
    if mean:
        if mask:
            result = _reduce_mean(result)
        else:
            result = tf.reduce_mean(result)

    result = _nan2inf(result)
    return result
# def ZINB(pi, disp, y_true, y_pred, ridge_lambda, mean = True, mask = False, debug = False):
#     x = y_true
#     eps = 1e-10
#     scale_factor=1.0
#     mean = mean * scale_factor

#     t1 = tf.math.lgamma(disp+eps) + tf.math.lgamma(x+1.0) - tf.math.lgamma(x+disp+eps)
#     t2 = (disp+x) * tf.math.log(1.0 + (mean/(disp+eps))) + (x * (tf.math.log(disp+eps) - tf.math.log(mean+eps)))
#     # print(t1)
#     # print(t2)  
#     nb_final = t1 + t2

#     nb_case = nb_final - tf.math.log(1.0-pi+eps)
#     zero_nb = tf.math.pow(disp/(disp+mean+eps), disp)
#     zero_case = -tf.math.log(pi + ((1.0-pi)*zero_nb)+eps)
#     result = tf.where(tf.less(x, 1e-8), zero_case, nb_case)
        
#     if ridge_lambda > 0:
#         ridge = ridge_lambda*tf.math.square(pi)
#         result += ridge
    
#     result = tf.reduce_mean(result)
#     return result


class DataDict(collections.OrderedDict):

    def shuffle(self, random_state=np.random):
        shuffled = DataDict()
        shuffle_idx = None
        for item in self:
            shuffle_idx = random_state.permutation(self[item].shape[0]) \
                if shuffle_idx is None else shuffle_idx
            shuffled[item] = self[item][shuffle_idx]
        return shuffled

    @property
    def size(self):
        data_size = set([item.shape[0] for item in self.values()])
        assert len(data_size) == 1
        return data_size.pop()

    @property
    def shape(self):  # Compatibility with numpy arrays
        return [self.size]

    def __getitem__(self, fetch):
        if isinstance(fetch, (slice, np.ndarray)):
            return DataDict([
                (item, self[item][fetch]) for item in self
            ])
        return super(DataDict, self).__getitem__(fetch)


def densify(arr):
    if scipy.sparse.issparse(arr):
        return arr.toarray()
    return arr


def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn


decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
encode = empty_safe(np.vectorize(lambda _x: str(_x).encode("utf-8")), "S")
upper = empty_safe(np.vectorize(lambda x: str(x).upper()), str)
lower = empty_safe(np.vectorize(lambda x: str(x).lower()), str)
tostr = empty_safe(np.vectorize(str), str)

# import tensorflow_probability as tfp

# def js_divergence(probs_a,probs_b):

#     dist_a = tfp.distributions.Categorical(probs=probs_a)
#     dist_b = tfp.distributions.Categorical(probs=probs_b)
    
#     probs_m = (probs_a + probs_b)/2.0
#     dist_m = tfp.distributions.Categorical(probs=probs_m)

#     js = 0.5*(tfp.distributions.kl_divergence(dist_a,dist_m) + tfp.distributions.kl_divergence(dist_b,dist_m))
#     return js
# def loss_js(attn_weights, batch_size=64):
    
#     js_loss = tf.zeros(batch_size)

#     for i in range(len(attn_weights)-1):
#         js_loss += js_divergence(attn_weights[i],attn_weights[i+1])
    
#     js_loss += js_divergence(attn_weights[0], attn_weights[-1])
#     final_loss = tf.reduce_mean(js_loss/ (len(attn_weights)+1)) 
#     return final_loss

#  def js_divergence(a, b):
#     a, b = style_matrix(a), style_matrix(b)
#     a = a / tf.reduce_sum(a) + 0.001
#     b = b / tf.reduce_sum(b) + 0.001
#     m = (a + b) / 2

#     return tf.reduce_mean(a * tf.log(a / m) + b * tf.log(b / m)) * 10000000000000
# def pearson(x, y):
#     x, y = style_matrix(x), style_matrix(y)

#     x_average = tf.reduce_mean(x)
#     y_average = tf.reduce_mean(y)
#     x_variance = tf.reduce_mean(tf.square(x - x_average))
#     y_variance = tf.reduce_mean(tf.square(y - y_average))

#     covariance = tf.reduce_sum((x - x_average) * (y - y_average))

#     return -covariance / tf.sqrt(x_variance * y_variance)

def cost_matrix(x,y):
    x_col = tf.expand_dims(x,-2)
    y_lin = tf.expand_dims(y,-3)
    c = tf.reduce_sum((tf.abs(x_col-y_lin))**2,axis=-1)
    return c

def sinkhorn_loss(x,y,epsilon,niter,reduction = 'mean'):
    '''
    Parameters
    ----------
    x : 输入A
    y : 输入B
    epsilon :缩放参数
    n : A/B中元素个数
    niter : 迭代次数
    Return:
        返回结果和sinkhorn距离
    '''
    def M(C,u,v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + tf.expand_dims(u,-1) + tf.expand_dims(v,-2)) / epsilon
    def lse(A):
        return tf.reduce_logsumexp(A,axis=1,keepdims=True)


    x_points = tf.shape(x)[-2]
    y_points = tf.shape(y)[-2]

    
    dim = len( tf.shape(x) )
    if dim == 2:
        batch = 1
    else:
        batch = tf.shape(x)[0]
        
    mu = tf.ones([batch,x_points] , dtype = tf.float32)*(1.0/tf.cast(x_points,tf.float32))
    nu = tf.ones([batch,y_points] , dtype = tf.float32)*(1.0/tf.cast(y_points,tf.float32))
    mu = tf.squeeze(mu)
    nu = tf.squeeze(nu)
    
    
    u, v = 0. * mu, 0. * nu
    C = cost_matrix(x, y)  # Wasserstein cost function    
    for i in range(niter):
        
        u1 = u  #保存上一步U值
        
        u = epsilon * (tf.math.log(mu+1e-8) - tf.squeeze(lse(M(C,u, v)) )  ) + u
        v = epsilon * (tf.math.log(nu+1e-8) - tf.squeeze( lse(tf.transpose(M(C,u, v))) ) ) + v
        
        err = tf.reduce_mean( tf.reduce_sum( tf.abs(u - u1),-1) )
        # print("err",err)
        if err.numpy() < 1e-1: #如果U值没有在更新，则结束
            break

    u_final,v_final = u,v
    pi = tf.exp(M(C, u_final,v_final))
    if reduction == 'mean':
        cost = tf.reduce_mean(pi*C) 
    elif reduction == 'sum':
        cost = tf.reduce_sum(pi*C)

    return cost,pi, C