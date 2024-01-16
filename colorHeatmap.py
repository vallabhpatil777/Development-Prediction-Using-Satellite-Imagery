import cv2
import matplotlib.pyplot as plt
import operator

try:
    fname="abc.png"
    img = cv2.imread(str(fname), 1)
    color = ('b','g','r')
    qtdBlue = 0
    qtdGreen = 0
    qtdRed = 0
    totalPixels = 0
    

    for channel,col in enumerate(color):
        histr = cv2.calcHist([img],[channel],None,[256],[1,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
        totalPixels+=sum(histr)
        #print histr
        if channel==0:
            qtdBlue = sum(histr)
        elif channel==1:
            qtdGreen = sum(histr)
        elif channel==2:
            qtdRed = sum(histr)

    qtdBlue = (qtdBlue/totalPixels)*100
    qtdGreen = (qtdGreen/totalPixels)*100
    qtdRed = (qtdRed/totalPixels)*100

    qtdBlue = filter(operator.isNumberType, qtdBlue)
    qtdGreen = filter(operator.isNumberType, qtdGreen)
    qtdRed = filter(operator.isNumberType, qtdRed)

    plt.title("Color Heatmap")
    plt.show()
    cv2.waitKey(27)
    cv2.destroyAllWindows
except Exception as e:
    print "Exception is",e
