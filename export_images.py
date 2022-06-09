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
            cv2.imwrite( pathOut + "\\frame%d.ppm" % count, image)     # save frame as JPEG file
            count = count + 1
        except:pass



if __name__=="__main__":

    video_path='C:/Users/mom44/Google Drive (mohammed@clearimageai.com)/projects/4-loss-prevention/media/raw_videos/input.mp4'
    image_folder='data/ppm_folder'
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to video",default=video_path)
    a.add_argument("--pathOut", help="path to images",default=image_folder)
    args = a.parse_args()
    print(args)

    extractImages(args.pathIn, args.pathOut)