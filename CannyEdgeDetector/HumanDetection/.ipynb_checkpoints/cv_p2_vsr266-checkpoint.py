
# coding: utf-8



# IMPORTING ALL THE LIBRARIES NEEDED

import glob                # For getting all filenames of the imagess in the given folder
import cv2
import numpy as np         # For array manipulations
import math as m                       # For math functions, square root.




# Function to convert the image to grayscale image

def convert_gray(img):
    conversion = np.array([0.229,0.587,0.114])                          # List to multiply to get grayscale image
    gray_img_array = np.around(np.dot(img,conversion))                  # Taking dot product and then rounding off to get grayscale image
    return gray_img_array                                               




# FUNCTION FOR CALCULATING GRADIENT

def gradient(img,fd,sd):
    # DEFINING PREWITT OPERATORS FOR GRADIENT CALCULATION

    prewitt_x = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])       # Prewitt x operator defined
    prewitt_y = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])       # Prewitt y operator defined
    prfd, prsd = prewitt_x.shape                             # Storing the dimensions of prewitt masks into variables
    gx = np.zeros((fd,sd),dtype = np.float)                    # Defining gradient x array 
    gy = np.zeros((fd,sd),dtype = np.float)                    # Defining gradient y array 
    gxn = np.zeros((fd,sd),dtype = np.float)                   # Defining the normalized gradient x array
    gyn = np.zeros((fd,sd),dtype = np.float)                   # Defining the normalized gradient y array
    gm = np.zeros((fd,sd),dtype = np.float)                    # Defining the gradient magnitude array
    temp_gradient = np.zeros((prfd,prsd),dtype = np.float)     # Defining temporary array that will store the slice of smoothed image array for direct matrix multiplication
    for i in range(fd-prfd+1):
        for j in range(sd-prfd+1):
            temp_gradient = img[(i):(3+i),(j):(3+j)]       # Storing the slice of smoothed image array in temporary array
            gx[1+i,1+j] = np.sum(np.multiply(temp_gradient, prewitt_x))  # Applying convolution for gradient x by directly multpilying the slice of matrix with prewitt x operator
            gy[1+i,1+j] = np.sum(np.multiply(temp_gradient, prewitt_y))  # Applying convolution for gradient y by directly multiplying the slice of matrix with prewitt y operator
    gxn = np.absolute(gx)/3                 # Forming normalized gradient x matrix from gradient x matrix by taking absolute value using np.absolute() and dividing by three
    gyn = np.absolute(gy)/3                 # Forming normalized gradient y matrix from gradient y matrix by taking absolute value using np.absolute() and dividing by three
    gm = np.hypot(gxn,gyn)/np.sqrt(2)       # Forming the normalized gradient magnitude array by using np.hypot() which takes under root of sum of squares of normalized gradient x and normalized gradient y and then dividing by square root of 2 for normalization
    return np.around(gx),np.around(gy),np.around(gxn),np.around(gyn),np.around(gm)       # Returning gradient x, gradient y, normalized gradient x, normalized gradient y and normalized gradient magnitude




# Function to compute gradient angle, and then wrapping it around -10 to 170

def angle(gy,gx):
    ga = np.degrees(np.arctan2(gy,gx))                # To compute the gradient angle and then converting it into degrees
    for i in range(ga.shape[0]):                     
        for j in range(ga.shape[1]):
            if ga[i,j]<-10:                           # Mapping negative angles
                ga[i,j]+=180
            elif ga[i,j]>=170:                        # Mapping angles greater than 170
                ga[i,j]-=180
    return ga




# Function to proportionately divide gradient magnitude into histogram bins

def divide(mag, ang, x):
    c = abs(x-ang)/20                       # Dividing the magnitude and then returning
    return c*mag,(1-c)*mag




# Function to calculate the histogram of each 8x8 pixel cell, calling divide to split the gradient magnitude proportionally and
# then adding it to bins
def cell_histo(ga,gm):
    histogram = [0]*9
    for i in range(ga.shape[0]):
        for j in range(ga.shape[1]):
            if ga[i,j]<=0:
                mag1,mag2=divide(gm[i,j],ga[i,j],0)
                histogram[8]+=mag1
                histogram[0]+=mag2
            elif ga[i,j]>=0 and ga[i,j]<=20:
                mag1,mag2=divide(gm[i,j],ga[i,j],20)
                histogram[0]+=mag1
                histogram[1]+=mag2
            elif ga[i,j]>=20 and ga[i,j]<=40:
                mag1,mag2=divide(gm[i,j],ga[i,j],40)
                histogram[1]+=mag1
                histogram[2]+=mag2
            elif ga[i,j]>=40 and ga[i,j]<=60:
                mag1,mag2=divide(gm[i,j],ga[i,j],60)
                histogram[2]+=mag1
                histogram[3]+=mag2
            elif ga[i,j]>=60 and ga[i,j]<=80:
                mag1,mag2=divide(gm[i,j],ga[i,j],80)
                histogram[3]+=mag1
                histogram[4]+=mag2
            elif ga[i,j]>=80 and ga[i,j]<=100:
                mag1,mag2=divide(gm[i,j],ga[i,j],100)
                histogram[4]+=mag1
                histogram[5]+=mag2
            elif ga[i,j]>=100 and ga[i,j]<=120:
                mag1,mag2=divide(gm[i,j],ga[i,j],120)
                histogram[5]+=mag1
                histogram[6]+=mag2
            elif ga[i,j]>=120 and ga[i,j]<=140:
                mag1,mag2=divide(gm[i,j],ga[i,j],140)
                histogram[6]+=mag1
                histogram[7]+=mag2
            elif ga[i,j]>=140 and ga[i,j]<=160:
                mag1,mag2=divide(gm[i,j],ga[i,j],160)
                histogram[7]+=mag1
                histogram[8]+=mag2
            elif ga[i,j]>=160:
                mag1,mag2=divide(gm[i,j],ga[i,j],160)
                histogram[0]+=mag1
                histogram[8]+=mag2
    return histogram
        




# Function to calculate the L2 Norm of each histogram, taking 2x2 cells and returning 36xq vector


def normalize(histo):
    sqsum = 0                                                     # To store square sum
    norm_histo = []                                               # To store 36x1 histogram
    for i in range(2):
        for j in range(2):
            for k in range(9):
                sqsum += (histo[i,j,k]*histo[i,j,k])                # Taking square sum of each value
                norm_histo.append(histo[i,j,k])
    lval = m.sqrt(sqsum)                                           # Taking sqaure root of square sum
    norm_histo = np.array(norm_histo)                              # Converting list to numpy array for easier calculations
    if lval!=0:                                                    # If not zero only then divide else let it be, it will remain 0
        norm_histo = norm_histo/lval
    return norm_histo




# Function to calculate descriptor of all images, it calls cell_histo to calculate histograms of all 8x8 cells, normalize to do L2
# normalization


def hog_descriptor(ga,gm):
    x = int(ga.shape[0]/8)
    y = int(ga.shape[1]/8)
    histogram = np.zeros((x,y,9))
    index = [0,0]
    for i in range(x):
        for j in range(y):
            temp = cell_histo(ga[index[0]:(index[0]+8),index[1]:(index[1]+8)],gm[index[0]:(index[0]+8),index[1]:(index[1]+8)])
            histogram[i,j]=temp
            index[1] += 8
        index[0] += 8
        index[1] = 0
    norm_histo = []
    for i in range(x-1):
        for j in range(y-1):
            temp = normalize(histogram[i:(i+2),j:(j+2)])
            temp = temp.tolist()
            norm_histo.extend(temp)
    norm_histo = np.array(norm_histo)
    return norm_histo


def bp(block):
    histogram={}
    allowed=[0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30,
              31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 
              126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 
              193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 
              240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255]
    histogram = {el:0 for el in allowed}
    for r in range(block.shape[0]):
        for c in range(block.shape[1]):
            barray=[]
            if (r == 0 or c ==0 or r == block.shape[0]-1 or c == block.shape[1]-1):
                if histogram[5] == 0:
                    histogram[5]=1
                else:
                    histogram[5]+=1
            else:
                for i in range(r-1, r+2):
                    for j in range(c-1, c+2):
                        if block[i][j] > block[r][c]:
                            barray.append(1)
                        else:
                            barray.append(0)
                barray.pop(4)
                barray.reverse()
                wherebarray=np.where(barray)[0]
                if len(wherebarray)>=1:
                    num=0
                    for n in wherebarray:
                        num+=2**n
                else:
                    num=0
                if num in allowed and num != 5:
                    if histogram[num]==0:
                        histogram[num]=1
                    else:
                        histogram[num]+=1
    return histogram
    

                
def lbp(img, r, c):
    x=int(r/16)
    y=int(c/16)
    index=[0,0]
    histogram=np.zeros((x,y,59))
    for i in range(x):
        for j in range(y):
            temp=bp(img[index[0]:(index[0]+16), index[1]:index[1]+16])
            temp=list(temp.values())
            histogram[i,j]=temp
            index[1]+=16
        index[0]+=16
        index[1]=0
    flt = []
    for i in range(x):
        for j in range(y):
            ssum=0
            norm_histo=[]
            for k in range(59):
                ssum+=(histogram[i,j,k]*histogram[i,j,k])
                norm_histo.append(histogram[i,j,k])
            lval=m.sqrt(ssum)
            norm_histo=np.array(norm_histo)
            if lval!=0:
                norm_histo=norm_histo/lval
            norm_histo = norm_histo.tolist()
            flt.extend(norm_histo)
    flt = np.array(flt)
    return flt
    


# Function to calculate RELU


def relu(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i,j]<0:
                x[i,j]=0
    return x




# Function to calculate sigmoid

def sigmoid(x):
    return 1/(1+np.exp(-x))




# Function to implament neural network, to train it.

def neural_net(inp,out,hidden):
    aplha = 0.1                                               # Initializing the learning rate
    col = inp.shape[1]                                         
    col2 = 1
    w1 = np.random.randn(col,hidden)                          # Weight for layer 1
    w1 = np.multiply(w1,m.sqrt(2/int(col+hidden)))            # Faactoring the weight
    w2 = np.random.randn(hidden,col2)                         # Weight for layer 2
    w2 = np.multiply(w2,m.sqrt(2/int(hidden+col2)))           # Factoring the weight
    w1bias = np.random.randn(hidden)                          # Bias for layer 1
    w1bias = np.multiply(w1bias,m.sqrt(2/int(hidden)))
    w2bias = np.random.randn(col2)                           # Bias for layer 2
    w2bias = np.multiply(w2bias,m.sqrt(2/int(col2)))
    err_curve=np.zeros((100,1))                              # Error array for each epoch
    epoch = 0
    while epoch<100:                                         # Doing forward and backward propogation for each epoch
        for i in range(inp.shape[0]):
            x = inp[i,:].reshape([1,-1])
            z = relu((x.dot(w1)+w1bias))                     # Computing values for hidden layer
            y = sigmoid((z.dot(w2)+w2bias))                  # Computing values for output layer
            err = out[i]-y                                   # Error for output layer             
            sqerr = 0.5*err*err                              # Square error                      
            del_out=(-1*err)*(1-y)*y                         
            del_layer2=z.T.dot(del_out)
            del_layer20=np.sum(del_out,axis=0)
            zz=np.zeros_like(z)
            for k in range(hidden):
            
                if(z[0][k]>0):
                    zz[0][k]=1
                else:
                    zz[0][k]=0                       
            del_hidden= del_out.dot(w2.T)*zz
            del_layer1=x.T.dot(del_hidden)
            delta_layer10=np.sum(del_hidden,axis=0)
            
            w2-= aplha*del_layer2
            w2bias-= aplha*del_layer20
            w1-= aplha*del_layer1
            w1bias-= aplha*delta_layer10
            err_curve[epoch] = sqerr/inp.shape[0]
        print('Epoch %d: err %f'%(epoch,np.mean(sqerr)/inp.shape[0]))
        epoch +=1
    return w1,w1bias,w2,w2bias,err_curve




# Function to predict values for my neural network

def predict(w,wb,v,vb,Output_descriptor):
    Number_of_test_image,number_of_attribute=Output_descriptor.shape
    predict=[]
    for k in range(Number_of_test_image):
            x=Output_descriptor[k,:].reshape([1,-1])
            z=relu((x.dot(w)+wb))
            y=sigmoid(z.dot(v)+vb)
            predict.append(y)
    return predict




# Main function that calls every other function

def main():
    
    trainx = []                    # List to store all training images
    trainy = []                    # List to store all training output
    train_p_path = '/Users/vvviren/Desktop/kp2_vsr266/train_p'
    train_n_path = '/Users/vvviren/Desktop/kp2_vsr266/train_n'
    test_p_path = '/Users/vvviren/Desktop/kp2_vsr266/test_p'
    test_n_path = '/Users/vvviren/Desktop/kp2_vsr266/test_n'
    trn = 0
    
    for filename in glob.glob(train_p_path+'/*.bmp'):          # Getting all the filenames of positive images
        img = np.array(cv2.imread(filename, cv2.IMREAD_COLOR))             # Opening the image and converrting to numpy array
        trainx.append(img)                               # Appending to the train image array
        trainy.append(1)                                 # Appending to the test array, the value 1 for positive
        trn += 1

    for filename in glob.glob(train_n_path+'/*.bmp'):          # Getting all file names of negative images
        img = np.array(cv2.imread(filename, cv2.IMREAD_COLOR))             # Opening the image and converting to numpy array
        trainx.append(img)                               # Appending to train image array
        trainy.append(0)                                 # Appending to the test array, the value 0 for negative


    testx = []                                       # List to store all testing images 
    testy = []                                       # List to store all training output
    tst = 0
    for filename in glob.glob(test_p_path+'/*.bmp'):   # Getting all the filenames of positive images
        img = np.array(cv2.imread(filename, cv2.IMREAD_COLOR))      # Opening the image and converrting to numpy array
        testx.append(img)                         # Appending to the train image array
        testy.append(1)                           # Appending to the test array, the value 1 for positive
        tst += 1

    for filename in glob.glob(test_n_path+'/*.bmp'):  # Getting all file names of negative images
        img = np.array(cv2.imread(filename, cv2.IMREAD_COLOR))     # Opening the image and converting to numpy array
        testx.append(img)                        # Appending to train image array
        testy.append(0)                           # Appending to the test array, the value 0 for negative
            

    for i in range(len(trainx)):
        trainx[i] = convert_gray(trainx[i])         # Converting train images to grayscale
    for i in range(len(testx)):
        testx[i] = convert_gray(testx[i])           # Converting test images to grayscale

    fvector = np.zeros((20,7524))                   # Creating HOG descriptor for training images
    fvector2 = np.zeros((10,7524))                  # Creating HOG descriptor for test images
    newfvector = np.zeros((20,11064))
    newfvector2=np.zeros((10,11064))


    for i in range(len(trainx)):
        gx,gy,gxn,gyn,gm = gradient(trainx[i],trainx[i].shape[0],trainx[i].shape[1])    # Calculating gradient magnitude of all training images
        ga = angle(gy,gx)                          # Calculating gradient angle of all training images
        fvector[i] = hog_descriptor(ga,gm)         # Storing HOG descriptor for all images
        lvector = lbp(trainx[i] ,trainx[i].shape[0],trainx[i].shape[1])
        newfvector[i] = np.concatenate((fvector[i],lvector))


    for i in range(len(testx)):
        gx,gy,gxn,gyn,gm = gradient(testx[i],testx[i].shape[0],testx[i].shape[1])  # Calculating gradient magnitude of all testing images
        ga = angle(gy,gx)                        # Calculating gradient angle of all training images
        cv2.imwrite('/Users/vvviren/Desktop/kp2_vsr266/gradients{}.bmp'.format(i),gm) # Writing the images to directory
        fvector2[i] = hog_descriptor(ga,gm)       # Storing HOG descriptor for all images
        lvector2 = lbp(testx[i] ,testx[i].shape[0],testx[i].shape[1])
        newfvector2[i] = np.concatenate((fvector2[i],lvector2))
    

    hidden = [200,400]    # List with values of Hidden neurons


    for i in range(len(hidden)):   # running neural networks for different values of hidden neurons
        print('HIDDEN LAYER = %d'%(hidden[i]))
        print('\n\n')
        w1,w1bias,w2,w2bias,err_curve = neural_net(newfvector,np.array(trainy),hidden[i])
        predicted_output=predict(w1,w1bias,w2,w2bias,newfvector2)
        pre=[]

        for check in predicted_output:
            if(check >=0.5):
                pre.append(1)
            else:
                pre.append(0)
            print(check)

        print(len(pre))

        correct=0
        wrong=0

        for i in range(len(pre)):
            if(pre[i]==testy[i]):
                correct+=1
            else:
                wrong+=1

        print('correct = %d'%(correct))
        print('wrong = %d'%(wrong))

        print(pre)
        print(testy)
        print('\n\n\n')
    
    
    
#    f = open('D:\\CV\\Project 2\\Project 2 report\\HOG_crop001278a.txt','w+')
#    f2 = open('D:\\CV\\Project 2\\Project 2 report\\HOG_crop001045b.txt','w+')
#    print('The HOG descriptor for the image crop001278a.bmp\n\n')
#    for i in range(len(fvector[5])):
#        f.write('%.17f\n'%(fvector[5,i]))
#    print('\n\nThe HOG descriptor for the image crop001045b.bmp\n\n')
#    for i in range(len(fvector2[3])):
#        f2.write('%.17f\n'%(fvector2[3,i]))
#    f.close()
#    f2.close()




# Function for calling main function

if __name__=="__main__":
    main()

