import numpy as np
import sys
import cv2
import math
import os
import random
from glob import glob
"""
Class for HOG feature extraction
"""
class HOG:
    """
    function to convert image from blue-gree-red to grayscale
    img = the image matrix
    """
    def bgr2gray(self, img):
        rows = img.shape[0]
        cols = img.shape[1]
        gray = np.zeros((rows,cols))
        for i in range(0,rows):
            for j in range(0,cols):
                bgr = img[i,j]
                r = bgr[2]
                g = bgr[1]
                b = bgr[0]
                gray[i,j] = round(0.299*r+0.587*g+0.114*b)
#         print(gray.shape)
        return gray

    """
    function to perform prewitts operation
    img = the image matrix
    """
    def prewitt(self, img):
        #Mask to calculate x gradient
        px = (1.0/3.0)*np.array([(-1,0,1),
                    (-1,0,1),
                    (-1,0,1)])
        #Mask to calculate y gradient
        py = (1.0/3.0)*np.array([(1,1,1),
                    (0,0,0),
                    (-1,-1,-1)])

        #initialize matrices to store the value of x & y gradient and gradient angle
        gradientx = np.zeros(shape=img.shape)
        gradienty = np.zeros(shape=img.shape)
        gradient = np.zeros(shape=img.shape)
        gradient_angle = np.zeros(shape=img.shape)
        rows = img.shape[0]
        cols = img.shape[1]
        #find the gradient values by perfoeming convolution
        for row in range(0,rows-2):
            for col in range(0,cols-2):
                gx = 0
                gy = 0
                #perform convolution with the 3x3 matrix starting at (i,j)
                for i in range (0,3):
                    for j in range (0,3):
                        gx = gx + img[row+i,col+j]*px[i,j]
                        gy = gy + img[row+i,col+j]*py[i,j]
                gradientx[row+1,col+1] = gx
                gradienty[row+1,col+1] = gy
                #normalize the gradient magnitude, divide by sqrt(2)
                gradient[row+1,col+1]=(((gx**2+gy**2)**(0.5))/np.sqrt(2))
                #calculate the gradient angle
                angle = 0
                if(gx == 0):
                    if(gy == 0):
                        angle = 0
                    else:
                        if( gy > 0):
                            angle = 90
                        else:
                            angle = -90
                else:
                    angle = round(math.degrees(np.arctan(gy/gx)))
                if (angle < 0):
                    angle = angle + 180
                gradient_angle[row+1,col+1]  = angle
        #return the gradient magnitude and gradient angle matrix
        return [gradient, gradient_angle]

    """
    function to get histogram for each 8x8 cell
    gradient = gradient magnitude for each pixel
    gradient_angle = gradient angle for each pixel
    """ 
    def getCellHistogram(self, gradient, gradient_angle):
        cellSize = 8
        rows = gradient.shape[0]
        cols = gradient.shape[1]
        #initialize the number of cell rows and cell columns
        cellRows = round(rows/cellSize)
        cellCols = round(cols/cellSize)
        cellHistogram = np.zeros((cellRows,cellCols,9))
        for i in range (0,cellRows-1):
            for j in range (0,cellCols-1):
                for row in range (i*8,i*8+8):
                    for col in range (j*8,j*8+8):
                        angle = gradient_angle[row,col]
                        mag = gradient[row,col]
                        if(angle%20 == 0):
                            if(angle == 180):
                                cellHistogram[i,j,0] += mag
                                continue
                            bin = int(angle/20)
                            cellHistogram[i,j,bin] += mag
                            continue
                        bin_l = int(angle/20)
                        #calculate the vote for left and right bins.
                        if(bin_l == 8):
                            bin_r = 0
                            cellHistogram[i,j,bin_l] += ((180-angle)/20)*mag
                            cellHistogram[i,j,bin_r] += ((angle-160)/20)*mag
                        else:
                            bin_r = bin_l+1
                            cellHistogram[i,j,bin_l] += (((bin_r*20)-angle)/20)*mag
                            cellHistogram[i,j,bin_r] += ((angle-(bin_l*20))/20)*mag
#                         print(bin_l,angle,'ba',row,col)
                break
        cellHistogramSuared = np.square(cellHistogram)
        return [cellHistogram, cellHistogramSuared] 

    """
    function to get the hog descriptor
    """
    def getHogDescriptor(self, cellHistogram, cellHistogramSquared):
        rows = cellHistogram.shape[0]
        cols = cellHistogram.shape[1]
        descriptor = np.array([])
        for row in range(0,rows-1):
            for col in range(0,cols-1):
                block = np.array([])
                temp = np.array([])
                block = np.append(block,cellHistogram[row,col])
                block = np.append(block,cellHistogram[row,col+1])
                block = np.append(block,cellHistogram[row+1,col])
                block = np.append(block,cellHistogram[row+1,col+1])
                temp = np.append(temp,cellHistogramSquared[row,col])
                temp = np.append(temp,cellHistogramSquared[row,col+1])
                temp = np.append(temp,cellHistogramSquared[row+1,col])
                temp = np.append(temp,cellHistogramSquared[row+1,col+1])
                temp = np.sum(temp)
                if(temp>0):
                    #normalize the block descriptor
                    norm = np.sqrt(temp)
                    block = (1/norm)*block
                descriptor = np.append(descriptor, block)
        return descriptor

    """
    function to read image
    """
    def readImage(self, path):
        return cv2.imread(path,cv2.IMREAD_COLOR)
    
    """
    function to save the image and hog descriptor
    """
    def saveImageAndHog(self, img, hog, path, append):
        pathSplit = path.split('/')
        currImage = pathSplit[-1]
        imageName, imageExt = currImage.split('.')
        updatedImage = '.'.join([imageName+append,imageExt])
        imageFolder = '/'.join(pathSplit[:-1])
        imageFolder += "_res"
        if not os.path.exists(imageFolder):
            os.makedirs(imageFolder)
        print(imageFolder)
        finalPath = '/'.join([imageFolder,updatedImage])
        cv2.imwrite(finalPath, img)
        filename = imageFolder+"/"+imageName+"_hog.txt"
        f = open(imageFolder+"/"+imageName+"_hog.txt",'ab+')
        np.savetxt(f, hog, delimiter = '\n')

    """
    function to get the hog descriptor for the given list of images
    """
    def hog(self, im_path):
        features = []
        for path in im_path:
            print(path)
            img = self.readImage(path)
            img = self.bgr2gray(img)
            gradients = self.prewitt(img)
            magnitude = gradients[0]
            print(magnitude.shape)
            gradient_angle = gradients[1]
            Histogram = self.getCellHistogram(magnitude, gradient_angle)
            cellHistogram = Histogram[0]
            cellHistogramSquared = Histogram[1]
            print(cellHistogram.shape)
            HOGdescriptor = self.getHogDescriptor(cellHistogram, cellHistogramSquared)
            HOGdescriptor = HOGdescriptor.reshape(-1,1)
            print(HOGdescriptor.shape)
            break
            #save gradient magnitude image and hog descriptor
            self.saveImageAndHog(magnitude, HOGdescriptor, path, "_mag")
            features.append(HOGdescriptor)
        return features

"""
Class for Neural Network
"""
class NeuralNet:

    """
    initialize the neural network
    graph = the shape of the neural network
    ep = the number of epocs
    lr = learning rate
    patience = number of epocs to continue after error changes very little 
    """
    def __init__(self, graph = (7524, 1000, 1), ep = 100, lr = 0.1, patience = 3):
        
        self.graph = graph
        #initialize the weights randomly
        np.random.seed(1)
        self.w1 = 0.1 * (2 * np.random.random((graph[1], graph[0])) - 1)
        np.random.seed(2)
        self.w2 = 0.1 * (2 * np.random.random((graph[2], graph[1])) - 1)

        #initialize bias for hidden layer and output to zero
        self.b1 = np.zeros((graph[1], 1))
        self.b2 = np.zeros((graph[2], 1))

        #intermediate values to remember
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.output = None

        self.ep = ep
        self.lr = lr
        self.patience = patience
    
    """
    feed forward, calculate the output of the neural net for given input
    """
    def ff(self, td):
        #calculate the input for the hidden layer [input*w1 + b1] 
        self.z1 = self.w1.dot(td) + self.b1
        #activation for hidden layer using ReLU
        self.a1 = self.ReLU(self.z1)
        #calculate the input for output layer [activation1*w2 + b2]
        self.z2 = self.w2.dot(self.a1) + self.b2
        #activation for output nodes using sigmoid
        self.output = self.sigmoid(self.z2)

    """
    function to calculate squared error from the output and expected value
    """
    def err(self, expected_output):
        return np.square(self.output - expected_output).sum()
    
    """
    function to perform back propogation
    """
    def bp(self, td, expedted_output):
        diff =  self.output - expedted_output
        t2 = 2 * diff * self.dsigmoid(self.output)
        self.d_w2 = np.dot(t2 ,self.a1.T)

        t1 = np.dot(self.w2.T,t2) * self.dReLU(self.a1)
        self.d_w1 = np.dot(t1,td.T)

        self.d_b2 = np.sum(t2, axis = 1, keepdims = True)
        self.d_b1 = np.sum(t1, axis = 1, keepdims = True)
    
    """
    update the weights and bias of the neural net
    """
    def update(self):
        self.w1 = self.w1 - (self.d_w1 * self.lr)
        self.b1 = self.b1 - (self.d_b1 * self.lr)

        self.w2 = self.w2 - (self.d_w2 * self.lr)
        self.b2 = self.b2 - (self.d_b2 * self.lr)
    
    """
    function to train the neural net
    training_data = input training data
    label = expected output
    """
    def train(self, training_data, label):
        dataLen = len(training_data)
        sn = np.arange(dataLen)
        random.shuffle(sn)
        prev_err = sys.maxsize
        for epoch in range(self.ep):
            ep_err = 0.0 #initialize error for current epoch to zero
            #train the network for each image and update the weights accordingly
            for count in sn:
                train_data = training_data[count]
                self.ff(train_data)
                error = self.err(label[count])
                ep_err += error
                self.bp(train_data, label[count])
                self.update()
            ep_err = ep_err/dataLen
            print("Epoch Count: " + str(epoch), "Average Error: ", ep_err)
            if(ep_err < prev_err):
                print("error decreased by ", prev_err-ep_err)
            else:
                if(ep_err > prev_err):
                    print("error increased by ", ep_err-prev_err)
                else:
                    print("error stayed same")
            #check for the change in error if very less we can stop training
            if(prev_err - ep_err < 0.000000001):
                self.patience -= 1
                if(self.patience == 0):
                    print("training complete....")
                    break
            #save the error for comparison with next epoch
            prev_err = ep_err
        #save the weights and bias of the network to a file
        self.saveState()

    """
    funciton to test the network with the testing data
    """
    def test(self, testImages, testing_data, label):
        misclassify = 0
        positiveList = []
        negativeList = []
        for count, test_data in enumerate(testing_data):
            self.ff(test_data)
            cPrediction = np.round(self.output[0])
            print("image: ", testImages[count])
            print("Predicted Probability: " + str(self.output.sum()), "Actual Probability Value: " + str(label[count]))
            if cPrediction:
                positiveList.append([testImages[count], str(self.output.sum())])
            else:
                negativeList.append([testImages[count], str(self.output.sum())])
            if(cPrediction - label[count] != 0):
                print("misclassified!!!!!!!!!!!!!!!!")
                misclassify += 1
        print(str(float(len(label)-misclassify) / float(len(label)) * 100) + " % Prediction Accuracy")

    """
    function to save the weights and bias of the network
    """
    def saveState(self):
        np.savetxt("weights1.csv", self.w1, delimiter=',')
        np.savetxt("weights2.csv", self.w2, delimiter=',')
        np.savetxt("bias1.csv", self.b1, delimiter=',')
        np.savetxt("bias2.csv", self.b2, delimiter=',')
    
    #sigmoid activation function
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    def dsigmoid(self, x):
        return x*(1-x)

    #ReLU activation function
    def ReLU(self,x):
        return x*(x>0)
    def dReLU(self, x):
        return 1*(x>0)


"""
function to get all the files in the given folder and expected lable
"""
def getImagesWithLable(dataFile, delimeter):
    PathList = []
    dataOut = []
    extension = "*.bmp"
    for dataFolder in dataFile.keys():
        for directoryName, subDirectory, fileL in os.walk(dataFolder):
            for imageFile in glob(os.path.join(directoryName, extension)):
                imageP = imageFile
                PathList.append(imageP)
                dataOut.append([dataFile[dataFolder]])
    return PathList, dataOut

"""
The entry point for our program
"""
if __name__ == "__main__":
    #define the path to folder that contains the test and train images 
    #main
    root = 'ImageData/' 
    train_pos_dir = root + 'Training_images_Pos'
    train_neg_dir = root + 'Trainingimages_Neg' 
    test_pos_dir = root + 'Test images (Pos)' 
    test_neg_dir = root + 'Test images (Neg)' 

    train_data_dir_with_output = {
        train_pos_dir:1,
        train_neg_dir:0
    }
    test_data_dir_with_output = {
        test_neg_dir:0,
        test_pos_dir:1
    }
    #get the complete list of train and test images along with expected label
    train_image_path_list, train_data_output = getImagesWithLable(train_data_dir_with_output,'/')
    test_image_path_list, test_data_output = getImagesWithLable(test_data_dir_with_output,'/')
    h = HOG()
    #get the hog feature descriptor for train and test images
    train_data_input = np.array(h.hog(train_image_path_list))
    test_data_input = np.array(h.hog(test_image_path_list))

    #initialize the neural network
#     neuralNet = NeuralNet(graph = (7524, 200, 1), ep = 100, lr = 0.1, patience = 3)
#     neuralNet.train(train_data_input, train_data_output)
#     neuralNet.test(test_image_path_list, test_data_input, test_data_output)