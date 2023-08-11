from __future__ import print_function
import sys
import argparse
import importlib
import importlib.util
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from ui import gui_design
from data import colorize_image as CI

sys.path.append('./caffe_files')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='iDeepColor: deep interactive colorization')
    # basic parameters
    parser.add_argument('--win_size', dest='win_size', help='the size of the main window', type=int, default=512)
    parser.add_argument('--image_file', dest='image_file', help='input image', type=str, default='test_imgs/mortar_pestle.jpg')
    parser.add_argument('--gpu', dest='gpu', help='gpu id', type=int, default=0)
    parser.add_argument('--cpu_mode', dest='cpu_mode', help='do not use gpu', action='store_true')

    # Caffe - Main colorization model
    parser.add_argument('--color_prototxt', dest='color_prototxt', help='colorization caffe prototxt', type=str, default='./models/reference_model/deploy_nodist.prototxt')
    parser.add_argument('--color_caffemodel', dest='color_caffemodel', help='colorization caffe prototxt', type=str, default='./models/reference_model/model.caffemodel')

    # Caffe - Distribution prediction model
    parser.add_argument('--dist_prototxt', dest='dist_prototxt', type=str, help='distribution net prototxt', default='./models/reference_model/deploy_nopred.prototxt')
    parser.add_argument('--dist_caffemodel', dest='dist_caffemodel', type=str, help='distribution net caffemodel', default='./models/reference_model/model.caffemodel')

    # PyTorch (same model used for both)
    parser.add_argument('--color_model', dest='color_model', help='colorization model', type=str, default='./models/pytorch/caffemodel.pth')
    parser.add_argument('--dist_model', dest='color_model', help='colorization distribution prediction model', type=str, default='./models/pytorch/caffemodel.pth')

    parser.add_argument('--backend', dest='backend', type=str, help='caffe or pytorch', default='caffe')
    parser.add_argument('--pytorch_maskcent', dest='pytorch_maskcent', help='need to center mask (activate for siggraph_pretrained but not for converted caffemodel)', action='store_true')

    # ***** DEPRECATED *****
    parser.add_argument('--load_size', dest='load_size', help='image size', type=int, default=256)

    # extra parameters
    parser.add_argument('--dark_style', dest='dark_style', help='use qdarkstyle ui', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    for arg in vars(args):
        print(f"[{arg}] = {getattr(args, arg)}")
    print("\n")

    args.win_size = int(args.win_size / 4.0) * 4  # make sure the width of the image can be divided by 4

    # initialize the colorization model
    if args.backend == 'caffe':
        if args.cpu_mode:
            args.gpu = -1
        colorModel = CI.ColorizeImageCaffe(args.load_size)
        colorModel.prep_net(args.gpu, args.color_prototxt, args.color_caffemodel)

        distModel = CI.ColorizeImageCaffeDist(args.load_size)
        distModel.prep_net(args.gpu, args.dist_prototxt, args.dist_caffemodel)
    elif args.backend == 'pytorch':
        if args.cpu_mode:
            args.gpu = None
        colorModel = CI.ColorizeImageTorch(args.load_size,args.pytorch_maskcent)
        colorModel.prep_net(args.gpu, args.color_model)

        distModel = CI.ColorizeImageTorchDist(args.load_size,args.pytorch_maskcent)
        distModel.prep_net(args.gpu, args.color_model, True)
    else:
        print(f"Backend type [{args.backend}] unknown")
        sys.exit()
    print("\n")

    # initialize application
    app = QApplication(sys.argv)
    if args.dark_style and importlib.util.find_spec("qdarkstyle") is not None:
        qdarkstyle = importlib.import_module("qdarkstyle")
        app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    window = gui_design.GUIDesign(colorModel, distModel, args.image_file, args.load_size, args.win_size)
    app.setWindowIcon(QIcon('imgs/logo.png'))  # load logo
    window.setWindowTitle('iColor')
    window.setWindowFlags(window.windowFlags())
    window.showMaximized()
    window.show()
    app.exec_()
