from keras.layers import Conv2D, Flatten, Lambda, MaxPooling2D, Dropout
from keras.models import Model, Sequential
import helper as aux
import glob
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split as trainTestSplit
import pickle
import os
from keras.callbacks import ModelCheckpoint


# Fully convolutional neural network model
def poolerPico(inputShape=(64, 64, 3)):
    """
    So-called 'Fully-convolutional Neural Network' (FCNN). Single filter in the top layer
    used for binary classification of 'vehicle/non-vehicle'
    :param inputShape: 
    :return: Keras model, model name
    """
    model = Sequential()
    # Center and normalize our data
    model.add(Lambda(lambda x: x / 255., input_shape=inputShape, output_shape=inputShape))
    # Block 0
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', name='cv0',
                     input_shape=inputShape, padding="same"))
    model.add(Dropout(0.5))

    # Block 1
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', name='cv1', padding="same"))
    model.add(Dropout(0.5))

    # block 2
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='cv2', padding="same"))
    model.add(MaxPooling2D(pool_size=(8, 8)))
    model.add(Dropout(0.5))

    # binary 'classifier'
    model.add(Conv2D(filters=1, kernel_size=(8, 8), name='fcn', activation="sigmoid"))

    return model, 'ppico'


def generator(samples, batchSize=32, useFlips=False, resize=False):
    """
    Generator to supply batches of sample images and labels
    :param samples: list of sample images file names
    :param batchSize: 
    :param useFlips: adds horizontal flips if True (effectively inflates training set by a factor of 2)
    :param resize: Halves images widths and heights if True
    :return: batch of images and labels
    """
    samplesCount = len(samples)

    while True:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, samplesCount, batchSize):
            batchSamples = samples[offset:offset + batchSize]

            xTrain = []
            yTrain = []
            for batchSample in batchSamples:
                y = float(batchSample[1])

                fileName = batchSample[0]

                image = aux.rgbImage(fileName, resize=resize)

                xTrain.append(image)
                yTrain.append(y)

                if useFlips:
                    flipImg = aux.flipImage(image)
                    xTrain.append(flipImg)
                    yTrain.append(y)

            xTrain = np.array(xTrain)
            yTrain = np.expand_dims(yTrain, axis=1)

            yield shuffle(xTrain, yTrain)  # Since we added flips, better shuffle again


def createSamples(x, y):
    """
    Returns a list of tuples (x, y)
    :param x: 
    :param y: 
    :return: 
    """
    assert len(x) == len(y)

    return [(x[i], y[i]) for i in range(len(x))]


def getData():
    """
    Creates dataset where x are image file names, y - labels (0 for non-vehicles / 1 for vehicles)
    :return: 
    """
    dataFile = 'data.p'

    if not os.path.isfile(dataFile):
        tryGenerateNew = aux.promptForInputCategorical(message='data file not found. Attempt to generate?',
                                                       options=['y', 'n']) == 'y'

        if tryGenerateNew:
            vehicleFolder = 'samples/vehicles/'
            nonVehiclesFolder = 'samples/non-vehicles/'

            if not os.path.isdir(vehicleFolder) or not os.path.isdir(nonVehiclesFolder):
                print('No samples found.')
                return None, None, None, None, None, None
            else:
                vehicleFiles = glob.glob('{}*/*.png'.format(vehicleFolder), recursive=True)
                nonVehicleFiles = glob.glob('{}*/*.png'.format(nonVehiclesFolder), recursive=True)

                imageSamplesFiles = vehicleFiles + nonVehicleFiles
                y = np.concatenate((np.ones(len(vehicleFiles)), np.zeros(len(nonVehicleFiles))))

                imageSamplesFiles, y = shuffle(imageSamplesFiles, y)

                # Using skLearn utils to split data to train and test sets
                xTrain, xTest, yTrain, yTest = trainTestSplit(imageSamplesFiles, y, test_size=0.2, random_state=42)

                # Further split train data to train and validation
                xTrain, xVal, yTrain, yVal = trainTestSplit(xTrain, yTrain, test_size=0.2, random_state=42)

                data = {'xTrain': xTrain, 'xValidation': xVal, 'xTest': xTest,
                        'yTrain': yTrain, 'yValidation': yVal, 'yTest': yTest}

                pickle.dump(data, open(dataFile, 'wb'))

                return xTrain, xVal, xTest, yTrain, yVal, yTest

        else:
            return None, None, None, None, None, None
    else:
        with open(dataFile, mode='rb') as f:
            data = pickle.load(f)

            xTrain = data['xTrain']
            xValidation = data['xValidation']
            xTest = data['xTest']
            yTrain = data['yTrain']
            yValidation = data['yValidation']
            yTest = data['yTest']

            return xTrain, xValidation, xTest, yTrain, yValidation, yTest


def main():
    xTrain, xVal, xTest, yTrain, yVal, yTest = getData()

    trainSamples = createSamples(xTrain, yTrain)
    validationSamples = createSamples(xVal, yVal)

    # batchSize = 32
    # useFlips = True
    # epochCount = 3

    batchSize = aux.promptForInt(message='Please specify the batch size (32, 64, etc.): ')
    useFlips = aux.promptForInputCategorical('Use flips?', options=['y', 'n']) == 'y'
    epochCount = aux.promptForInt(message='Please specify the number of epochs: ')

    inflateFactor = 2 if useFlips else 1

    # Keras generator params computation
    stepsPerEpoch = len(trainSamples) * inflateFactor / batchSize
    print('steps per epoch: {}'.format(stepsPerEpoch))

    validationSteps = len(validationSamples) * inflateFactor / batchSize
    print('validation steps per epoch: {}'.format(validationSteps))

    proceed = aux.promptForInputCategorical('Proceed?', ['y', 'n']) == 'y'

    if proceed:

        sourceModel, modelName = poolerPico()

        # Adding fully-connected layer to train the 'classifier'
        x = sourceModel.output
        x = Flatten()(x)
        model = Model(inputs=sourceModel.input, outputs=x)

        print(model.summary())

        confirm = aux.promptForInputCategorical('Confirm?', ['y', 'n']) == 'y'

        if not confirm:
            return
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        # Instantiating train and validation generators
        trainGen = generator(samples=trainSamples, useFlips=useFlips)
        validGen = generator(samples=validationSamples, useFlips=useFlips)

        timeStamp = aux.timeStamp()
        weightsFile = '{}_{}.h5'.format(modelName, timeStamp)

        checkpointer = ModelCheckpoint(filepath=weightsFile,
                                       monitor='val_acc', verbose=0, save_best_only=True)

        _ = model.fit_generator(trainGen,
                                steps_per_epoch=stepsPerEpoch,
                                validation_data=validGen,
                                validation_steps=validationSteps,
                                epochs=epochCount, callbacks=[checkpointer])

        print('Training complete. Weights for best validation accuracy have been saved to {}.'
              .format(weightsFile))

        # Evaluating accuracy on test set
        print('Evaluating accuracy on test set.')
        testSamples = createSamples(xTest, yTest)
        testGen = generator(samples=testSamples, useFlips=False)
        testSteps = len(testSamples) / batchSize
        accuracy = model.evaluate_generator(generator=testGen, steps=testSteps)

        print('test accuracy: ', accuracy)

if __name__ == '__main__':
    main()
