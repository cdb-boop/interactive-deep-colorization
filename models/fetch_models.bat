curl http://colorization.eecs.berkeley.edu/siggraph/models/model.caffemodel --output-dir ./models/reference_model/ --output model.caffemodel
curl http://colorization.eecs.berkeley.edu/siggraph/models/global_model.caffemodel --output-dir ./models/global_model/ --output global_model.caffemodel
curl http://colorization.eecs.berkeley.edu/siggraph/models/dummy.caffemodel --output-dir ./models/global_model/ --output dummy.caffemodel
curl http://colorization.eecs.berkeley.edu/siggraph/models/pytorch.pth --output-dir ./models/pytorch/ --output pytorch_trained.pth
curl http://colorization.eecs.berkeley.edu/siggraph/models/caffemodel.pth --output-dir ./models/pytorch/ --output caffemodel.pth
pause