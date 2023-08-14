import numpy as np
import numpy.typing as npt
import cv2
import matplotlib.pyplot as plt
import importlib
skcolor = importlib.import_module("skimage.color")  # ignore import issue with skimage
from sklearn.cluster import KMeans
import os
from scipy.ndimage.interpolation import zoom
from typing import Any
from abc import ABC, abstractmethod

@staticmethod
def create_temp_directory(path_template: str, N: int = int(1e8)) -> str:
    print(f"Path template: {path_template}")
    cur_path = path_template % np.random.randint(0, N)
    while(os.path.exists(cur_path)):
        cur_path = path_template % np.random.randint(0, N)
    print(f"Creating directory: {cur_path}")
    os.mkdir(cur_path)
    return cur_path

@staticmethod
def lab2rgb_transpose(img_l: np.ndarray, img_ab: np.ndarray) -> np.ndarray:
    ''' INPUTS
            img_l     1xXxX     [0,100]
            img_ab     2xXxX     [-100,100]
        OUTPUTS
            returned value is XxXx3 '''
    pred_lab = np.concatenate((img_l, img_ab), axis=0).transpose((1, 2, 0))
    pred_rgb = (np.clip(skcolor.lab2rgb(pred_lab), 0, 1) * 255).astype('uint8')
    return pred_rgb

@staticmethod
def rgb2lab_transpose(img_rgb: np.ndarray) -> np.ndarray:
    ''' INPUTS
            img_rgb XxXx3
        OUTPUTS
            returned value is 3xXxX '''
    return skcolor.rgb2lab(img_rgb).transpose((2, 0, 1))


class Data():
    def __init__(self, input_image_size: int = 256, max_input_image_size: int = 10000):
        # image metadata
        self.input_image_size = input_image_size
        self.img_l_set = False
        self.max_input_image_size = max_input_image_size  # maximum size of maximum dimension
        self.img_just_set = False  # this will be true whenever image is just loaded
        # net_forward can set this to False if they want

        # variables
        self.l_norm = 1.0
        self.ab_norm = 1.0
        self.l_mean = 50.0
        self.ab_mean = 0.0
        self.mask_mult = 1.0

        # images
        self.input_ab = np.array([], np.float64)
        self.input_ab_mc = np.array([], np.float64)
        self.input_mask = np.array([], np.float64)
        self.input_mask_mult = np.array([], np.float64)
        self.output_rgb = np.array([], np.float64)

    def load_image(self, input_path: str) -> None:
        # rgb image [C x input_image_size x input_image_size]
        im = cv2.cvtColor(cv2.imread(input_path, 1), cv2.COLOR_BGR2RGB)
        self.img_rgb_fullres = im.copy()
        self._set_img_lab_fullres_()

        im = cv2.resize(im, (self.input_image_size, self.input_image_size))
        self.img_rgb = im.copy()
        # self.img_rgb = sp.misc.imresize(plt.imread(input_path),(self.input_image_size,self.input_image_size)).transpose((2,0,1))

        self.img_l_set = True

        # convert into lab space
        self._set_img_lab_()
        self._set_img_lab_mc_()

    def set_image(self, input_image: np.ndarray) -> None:
        self.img_rgb_fullres = input_image.copy()
        self._set_img_lab_fullres_()

        self.img_l_set = True

        self.img_rgb = input_image
        # convert into lab space
        self._set_img_lab_()
        self._set_img_lab_mc_()

    def get_result_PSNR(self, result: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.array((result)).flatten()[0] == -1:
            cur_result = self.get_img_forward()
        else:
            cur_result = result.copy()
        SE_map = (1. * self.img_rgb - cur_result)**2
        cur_MSE = np.mean(SE_map)
        cur_PSNR = 20 * np.log10(255. / np.sqrt(cur_MSE))
        return(cur_PSNR, SE_map)

    def get_img_forward(self) -> npt.NDArray[np.float64]:
        """get image with point estimate"""
        return self.output_rgb

    def get_img_gray(self) -> np.ndarray:
        """Get black and white image"""
        return lab2rgb_transpose(self.img_l, np.zeros((2, self.input_image_size, self.input_image_size)))

    def get_img_gray_fullres(self) -> np.ndarray:
        """Get black and white image"""
        return lab2rgb_transpose(self.img_l_fullres, np.zeros((2, self.img_l_fullres.shape[1], self.img_l_fullres.shape[2])))

    def get_img_fullres(self) -> np.ndarray:
        """This assumes self.img_l_fullres, self.output_ab are set.
        Typically, this means that set_image() and net_forward() have been called.
        
        bilinear upsample"""
        zoom_factor = (1, 1. * self.img_l_fullres.shape[1] / self.output_ab.shape[1], 1. * self.img_l_fullres.shape[2] / self.output_ab.shape[2])
        output_ab_fullres = zoom(self.output_ab, zoom_factor, order=1)

        return lab2rgb_transpose(self.img_l_fullres, output_ab_fullres)

    def get_input_img_fullres(self) -> np.ndarray:
        zoom_factor = (1, 1. * self.img_l_fullres.shape[1] / self.input_ab.shape[1], 1. * self.img_l_fullres.shape[2] / self.input_ab.shape[2])
        input_ab_fullres = zoom(self.input_ab, zoom_factor, order=1)
        return lab2rgb_transpose(self.img_l_fullres, input_ab_fullres)

    def get_input_img(self) -> np.ndarray:
        return lab2rgb_transpose(self.img_l, self.input_ab)

    def get_img_mask(self) -> np.ndarray:
        """Get black and white image"""
        return lab2rgb_transpose(100. * (1 - self.input_mask), np.zeros((2, self.input_image_size, self.input_image_size)))

    def get_img_mask_fullres(self) -> np.ndarray:
        """Get black and white image"""
        zoom_factor = (1, 1. * self.img_l_fullres.shape[1] / self.input_ab.shape[1], 1. * self.img_l_fullres.shape[2] / self.input_ab.shape[2])
        input_mask_fullres = zoom(self.input_mask, zoom_factor, order=0)
        return lab2rgb_transpose(100. * (1 - input_mask_fullres), np.zeros((2, input_mask_fullres.shape[1], input_mask_fullres.shape[2])))

    def get_sup_img(self) -> np.ndarray:
        return lab2rgb_transpose(50 * self.input_mask, self.input_ab)

    def get_sup_fullres(self) -> np.ndarray:
        zoom_factor = (1, 1. * self.img_l_fullres.shape[1] / self.output_ab.shape[1], 1. * self.img_l_fullres.shape[2] / self.output_ab.shape[2])
        input_mask_fullres = zoom(self.input_mask, zoom_factor, order=0)
        input_ab_fullres = zoom(self.input_ab, zoom_factor, order=0)
        return lab2rgb_transpose(50 * input_mask_fullres, input_ab_fullres)

    # ***** Private functions *****
    def _set_run_input_(self, input_ab: np.ndarray, input_mask: np.ndarray) -> None:
        self.input_ab = input_ab
        self.input_ab_mc = (input_ab - self.ab_mean) / self.ab_norm
        self.input_mask = input_mask
        self.input_mask_mult = input_mask * self.mask_mult
    
    def _set_img_lab_fullres_(self) -> None:
        """adjust full resolution image to be within maximum dimension is within max_input_image_size"""
        Xfullres = self.img_rgb_fullres.shape[0]
        Yfullres = self.img_rgb_fullres.shape[1]
        if Xfullres > self.max_input_image_size or Yfullres > self.max_input_image_size:
            if Xfullres > Yfullres:
                zoom_factor = 1. * self.max_input_image_size / Xfullres
            else:
                zoom_factor = 1. * self.max_input_image_size / Yfullres
            self.img_rgb_fullres = zoom(self.img_rgb_fullres, (zoom_factor, zoom_factor, 1), order=1)

        self.img_lab_fullres = skcolor.rgb2lab(self.img_rgb_fullres).transpose((2, 0, 1))
        self.img_l_fullres = self.img_lab_fullres[[0], :, :]
        self.img_ab_fullres = self.img_lab_fullres[1:, :, :]

    def _set_img_lab_(self) -> None:
        """set self.img_lab from self.im_rgb"""
        self.img_lab = skcolor.rgb2lab(self.img_rgb).transpose((2, 0, 1))
        self.img_l = self.img_lab[[0], :, :]
        self.img_ab = self.img_lab[1:, :, :]

    def _set_img_lab_mc_(self) -> None:
        """set self.img_lab_mc from self.img_lab
        
        lab image, mean centered [XxYxX]"""
        self.img_lab_mc = self.img_lab / np.array((self.l_norm, self.ab_norm, self.ab_norm))[:, np.newaxis, np.newaxis] - np.array(
            (self.l_mean / self.l_norm, self.ab_mean / self.ab_norm, self.ab_mean / self.ab_norm))[:, np.newaxis, np.newaxis]
        self._set_img_l_()

    def _set_img_l_(self) -> None:
        self.img_l_mc = self.img_lab_mc[[0], :, :]
        self.img_l_set = True

    def _set_img_ab_(self) -> None:
        self.img_ab_mc = self.img_lab_mc[[1, 2], :, :]

    def _set_out_ab_(self) -> None:
        self.output_lab = rgb2lab_transpose(self.output_rgb)
        self.output_ab = self.output_lab[1:, :, :]


class Forward(ABC):
    @abstractmethod
    def forward(self, data: Data) -> npt.NDArray[np.float64]:
        """Returns output_rgb"""
        pass


class ForwardDist(ABC):
    @abstractmethod
    def forward(self, data: Data) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        pass
    
    @abstractmethod
    def get_pts_grid(self) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def get_in_hull(self) -> npt.NDArray[np.bool_]:
        pass

    @abstractmethod
    def get_pts_in_hull(self) -> np.ndarray:
        pass


class ForwardGlobDist(ABC):
    @abstractmethod
    def forward(self, data: Data, glob_dist: int = -1) -> npt.NDArray[np.float64]:
        pass


class ModelTorch(Forward):
    def __init__(self, gpu_id: int | None = None, path: str = '', dist: bool = False, mask_cent: bool = False):
        self.mask_cent = 0.5 if mask_cent else 0.0

        # prep net
        import torch
        import models.pytorch.model as model
        print(f"ColorizeTorch: path = {path}")
        print(f"ColorizeTorch: dist mode = {dist}")
        self.net = model.SIGGRAPHGenerator(use_gpu=gpu_id is not None, dist=dist)
        state_dict = torch.load(path)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, self.net, key.split('.'))
        self.net.load_state_dict(state_dict)
        if gpu_id is not None:
            self.net.cuda(gpu_id)
        self.net.eval()

    @staticmethod
    def __patch_instance_norm_state_dict(state_dict: dict, module: Any, keys: list, i: int = 0) -> None:
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            ModelTorch.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def forward(self, data: Data) -> npt.NDArray[np.float64]:
        output_ab = self.net.forward(data.img_l_mc, data.input_ab_mc, data.input_mask_mult, self.mask_cent)[0][0, :, :, :].cpu().data.numpy()
        return lab2rgb_transpose(data.img_l, output_ab)


class ModelDistTorch(ForwardDist):
    def __init__(self, gpu_id: int | None = None, path: str = '', mask_cent: bool = False):
        self.pts_grid = np.array(np.meshgrid(np.arange(-110, 120, 10), np.arange(-110, 120, 10))).reshape((2, 529)).T
        self.in_hull = np.ones(529, dtype=bool)
        self.pts_in_hull = np.array(np.meshgrid(np.arange(-110, 120, 10), np.arange(-110, 120, 10))).reshape((2, 529)).T

        # prep net
        self.torch = ModelTorch(gpu_id, path, True, mask_cent)

    def forward(self, data: Data) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # set distribution
        function_return, dist_ab = self.torch.net.forward(data.img_l_mc, data.input_ab_mc, data.input_mask_mult, self.torch.mask_cent)
        function_return = function_return[0, :, :, :].cpu().data.numpy()
        dist_ab = dist_ab[0, :, :, :].cpu().data.numpy()
        return function_return, dist_ab

    def get_pts_grid(self) -> npt.NDArray[np.float64]:
        return self.pts_grid

    def get_in_hull(self) -> npt.NDArray[np.bool_]:
        return self.in_hull

    def get_pts_in_hull(self) -> np.ndarray:
        return self.pts_in_hull


class ModelCaffe(Forward):
    def __init__(self, gpu_id: int = -1, prototxt_path: str = '', caffemodel_path: str = ''):
        self.pred_ab_layer = 'pred_ab'  # predicted ab layer

        # Load grid properties
        self.pts_in_hull_path = './data/color_bins/pts_in_hull.npy'
        self.pts_in_hull = np.load(self.pts_in_hull_path)  # 313x2, in-gamut

        # prep net
        import caffe
        print(f"ColorizeCaffe: gpu_id {gpu_id}")
        print(f"ColorizeCaffe: net_path {prototxt_path}")
        print(f"ColorizeCaffe: model_path {caffemodel_path}")
        if gpu_id == -1:
            caffe.set_mode_cpu()
        else:
            caffe.set_device(gpu_id)
            caffe.set_mode_gpu()
        self.net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)

        # automatically set cluster centers
        if len(self.net.params[self.pred_ab_layer][0].data[...].shape) == 4 and self.net.params[self.pred_ab_layer][0].data[...].shape[1] == 313:
            print(f"ColorizeImageCaffe: Setting ab cluster centers in layer: {self.pred_ab_layer}")
            self.net.params[self.pred_ab_layer][0].data[:, :, 0, 0] = self.pts_in_hull.T

        # automatically set upsampling kernel
        for layer in self.net._layer_names:
            if layer[-3:] == '_us':
                print(f"ColorizeImageCaffe: Setting upsampling layer kernel: {layer}")
                self.net.params[layer][0].data[:, 0, :, :] = np.array(((.25, .5, .25, 0), (.5, 1., .5, 0), (.25, .5, .25, 0), (0, 0, 0, 0)))[np.newaxis, :, :]

    def forward(self, data: Data) -> npt.NDArray[np.float64]:
        net_input_prepped = np.concatenate((data.img_l_mc, data.input_ab_mc, data.input_mask_mult), axis=0)

        self.net.blobs['data_l_ab_mask'].data[...] = net_input_prepped
        self.net.forward()

        # return prediction
        output_rgb = lab2rgb_transpose(data.img_l, self.net.blobs[self.pred_ab_layer].data[0, :, :, :])
        return output_rgb


class ModelDistCaffe(ForwardDist):
    def __init__(self, gpu_id: int = -1, prototxt_path: str = '', caffemodel_path: str = '', S: float = 0.2):
        self.scale_S_layer = 'scale_S'
        self.dist_ab_S_layer = 'dist_ab_S'  # softened distribution layer
        self.pts_grid = np.load('./data/color_bins/pts_grid.npy')  # 529x2, all points
        self.in_hull = np.load('./data/color_bins/in_hull.npy')  # 529 bool

        # prep net
        self.caffe = ModelCaffe(gpu_id, prototxt_path, caffemodel_path)
        self.S = S
        self.caffe.net.params[self.scale_S_layer][0].data[...] = S

    def forward(self, data: Data) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        function_return = self.caffe.forward(data)
        dist_ab = self.caffe.net.blobs[self.dist_ab_S_layer].data[0, :, :, :]
        return function_return, dist_ab

    def get_pts_grid(self) -> npt.NDArray[np.float64]:
        return self.pts_grid

    def get_in_hull(self) -> npt.NDArray[np.bool_]:
        return self.in_hull

    def get_pts_in_hull(self) -> np.ndarray:
        return self.caffe.pts_in_hull


class ModelGlobDistCaffe(ForwardGlobDist):
    def __init__(self, gpu_id: int = -1, prototxt_path: str = '', caffemodel_path: str = ''):
        self.glob_mask_mult = 1.
        self.glob_layer = 'glob_ab_313_mask'

        # prep net
        self.caffe = ModelCaffe(gpu_id, prototxt_path, caffemodel_path)

    def forward(self, data: Data, glob_dist: int = -1) -> npt.NDArray[np.float64]:
        # glob_dist is 313 array, or -1
        if np.array(glob_dist).flatten()[0] == -1:  # run without this, zero it out
            self.caffe.net.blobs[self.glob_layer].data[0, :-1, 0, 0] = 0.
            self.caffe.net.blobs[self.glob_layer].data[0, -1, 0, 0] = 0.
        else:  # run conditioned on global histogram
            self.caffe.net.blobs[self.glob_layer].data[0, :-1, 0, 0] = glob_dist
            self.caffe.net.blobs[self.glob_layer].data[0, -1, 0, 0] = self.glob_mask_mult
        return self.caffe.forward(data)


class Colorizer():
    def __init__(self, model: Forward, input_image_size: int = 256, max_input_image_size: int = 10000):
        self.model = model
        self.data = Data(input_image_size, max_input_image_size)

    def run(self, input_ab: np.ndarray, input_mask: np.ndarray) -> npt.NDArray[np.float64] | Exception:
        if(not self.data.img_l_set):
            return Exception("I need to have an image!")
        self.data._set_run_input_(input_ab, input_mask)

        self.data.output_rgb = self.model.forward(self.data)
        self.data._set_out_ab_()
        return self.data.output_rgb


class ColorizerDist():
    def __init__(self, model: ForwardDist, input_image_size: int = 256, max_input_image_size: int = 10000):
        self.model = model
        self.data = Data(input_image_size, max_input_image_size)
        self.input_image_size = input_image_size

        self.dist_ab_set = False
        self.AB = self.model.get_pts_grid().shape[0]  # 529
        self.A = int(np.sqrt(self.AB))  # 23
        self.B = int(np.sqrt(self.AB))  # 23
        self.dist_ab_full = np.zeros((self.AB, self.input_image_size, self.input_image_size))
        self.dist_ab_grid = np.zeros((self.A, self.B, self.input_image_size, self.input_image_size))
        self.dist_entropy = np.zeros((self.input_image_size, self.input_image_size))

    def run(self, input_ab: np.ndarray, input_mask: np.ndarray) -> npt.NDArray[np.float64] | Exception:
        if(not self.data.img_l_set):
            return Exception("I need to have an image!")
        self.data._set_run_input_(input_ab, input_mask)

        # set distribution
        # in-gamut, CxXxX, C = 313
        (function_return, self.dist_ab) = self.model.forward(self.data)
        self.dist_ab_set = True

        # full grid, ABxXxX, AB = 529
        self.dist_ab_full[self.model.get_in_hull(), :, :] = self.dist_ab

        # gridded, AxBxXxX, A = 23
        self.dist_ab_grid = self.dist_ab_full.reshape((self.A, self.B, self.input_image_size, self.input_image_size))

        # return
        return function_return

    def get_ab_reccs(self, h: int, w: int, K: int = 5, N: int = 25000) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | Exception:
        """Recommended colors at point (h,w)

        Call this after calling net_forward

        Returns (cluster centers, percentage of points within each cluster)"""

        if not self.dist_ab_set:
            return Exception("Need to set prediction first!")

        # randomly sample from pdf
        cmf = np.cumsum(self.dist_ab[:, h, w])  # CMF
        cmf = cmf / cmf[-1]
        cmf_bins = cmf

        # randomly sample N points
        rnd_pts = np.random.uniform(low=0, high=1.0, size=N)
        inds = np.digitize(rnd_pts, bins=cmf_bins)
        rnd_pts_ab = self.model.get_pts_in_hull()[inds, :]

        # run k-means
        kmeans = KMeans(n_clusters=K).fit(rnd_pts_ab)

        # sort by cluster occupancy
        k_label_cnt = np.histogram(kmeans.labels_, np.arange(0, K + 1))[0]
        k_inds = np.argsort(k_label_cnt, axis=0)[::-1]

        cluster_percentages = 1. * k_label_cnt[k_inds] / N  # percentage of points within cluster
        cluster_centers = kmeans.cluster_centers_[k_inds, :]  # cluster centers

        # cluster_centers = np.random.uniform(low=-100,high=100,size=(N,2))
        return (cluster_centers, cluster_percentages)

    def compute_entropy(self) -> None:
        # compute the distribution entropy (really slow right now)
        self.dist_entropy = np.sum(self.dist_ab * np.log(self.dist_ab), axis=0)

    def plot_dist_grid(self, h: int, w: int) -> None:
        # Plots distribution at a given point
        plt.figure()
        plt.imshow(self.dist_ab_grid[:, :, h, w], extent=[-110, 110, 110, -110], interpolation='nearest')
        plt.colorbar()
        plt.ylabel('a')
        plt.xlabel('b')

    def plot_dist_entropy(self) -> None:
        # Plots distribution at a given point
        plt.figure()
        plt.imshow(-self.dist_entropy, interpolation='nearest')
        plt.colorbar()


class ColorizerGlobDist():
    def __init__(self, model: ForwardGlobDist, input_image_size: int = 256, max_input_image_size: int = 10000):
        self.model = model
        self.data = Data(input_image_size, max_input_image_size)

    def run(self, input_ab: np.ndarray, input_mask: np.ndarray, glob_dist: int = -1) -> npt.NDArray[np.float64] | Exception:
        if(not self.data.img_l_set):
            return Exception("I need to have an image!")
        self.data._set_run_input_(input_ab, input_mask)

        return self.model.forward(self.data, glob_dist)
