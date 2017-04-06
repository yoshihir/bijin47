'''Cut face.'''

import os
import cv2

def main():
    '''
    Cut face
    '''
    for srcpath, _, files in os.walk('bijinwatch'):
        if len(_):
            continue
        dstpath = srcpath.replace('bijinwatch', 'faces')
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        for filename in files:
            if filename.startswith('.'):  # Pass .DS_Store
                continue
            try:
                detect_faces(srcpath, dstpath, filename)
            except:
                continue


def detect_faces(srcpath, dstpath, filename):
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    image = cv2.imread('{}/{}'.format(srcpath, filename))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray_image)
    # Extract when just one face is detected

    for i in range(len(faces)):
        (x, y, w, h) = faces[i]
        image = image[y:y+h, x:x+w]
        image = cv2.resize(image, (100, 100))
        if i == 0:
            cv2.imwrite('{}/{}'.format(dstpath, filename), image)
            print('{}/{}'.format(dstpath, filename) + ' output')
        else:
             cv2.imwrite(dstpath + '/' + str(i) + '_' + filename, image)
             print(dstpath + '/' + str(i) + '_' + filename + ' output')

if __name__ == '__main__':
    main()