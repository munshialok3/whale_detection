import caffe
import cv2
import sys

def deploy(img_path):

    caffe.set_mode_gpu()
    
    net = caffe.Classifier('/dli/data/digits/20200514-092522-38e6/deploy.prototxt', '/dli/data/digits/20200514-092522-38e6/snapshot_iter_54.caffemodel',
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))
                       
    input_image= caffe.io.load_image(img_path)
    input_image = cv2.resize(input_image, (256,256))
    mean_image = caffe.io.load_image('/dli/data/digits/20200514-085933-e186/mean.jpg')
    input_image = input_image-mean_image

    prediction = net.prediction = net.predict([ready_image])

    if prediction.argmax()==0:
        print "Whale"
    else:
        print "Not whale"
    
if __name__ == '__main__':
    print(deploy(sys.argv[1]))

