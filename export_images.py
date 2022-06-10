import cv2
import argparse
print(cv2.__version__)

def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        #vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line
        success,image = vidcap.read()
        print ('Read a new frame: ', success)

        try:
            resized = resize(image, 500, 375)
            cv2.imwrite( pathOut + "\\frame%d.ppm" % count, resized)     # save frame as JPEG file
            count = count + 1
        except:pass

def resize(src,w,h):
    width = int(w)
    height = int(h)
    dim = (width, height)

    # resize image
    return  cv2.resize(src, dim, interpolation=cv2.INTER_AREA)

if __name__=="__main__":

    video_path='C:/Users/mom44/Google Drive (mohammed@clearimageai.com)/projects/4-loss-prevention/media/raw_videos/input.mp4'
    image_folder='data/sampleFasterRCNN/faster-rcnn'
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to video",default=video_path)
    a.add_argument("--pathOut", help="path to images",default=image_folder)
    args = a.parse_args()
    print(args)

    extractImages(args.pathIn, args.pathOut)