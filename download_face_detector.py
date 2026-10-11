import urllib.request

# URLs of the files
prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
caffemodel_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

# Download deploy.prototxt
urllib.request.urlretrieve(prototxt_url, "deploy.prototxt")
print("✅ deploy.prototxt downloaded")

# Download caffemodel
urllib.request.urlretrieve(caffemodel_url, "res10_300x300_ssd_iter_140000.caffemodel")
print("✅ res10_300x300_ssd_iter_140000.caffemodel downloaded")
