import numpy as np
import model
from scipy.ndimage.measurements import label
import cv2
import helper as aux


class VehicleScanner:
    """
    pointSize defines the size of the samples those the CNN has been trained on.
    """
    def __init__(self, imgInputShape=(720, 1280, 3), crop=(400, 660), pointSize=64,
                 confidenceThrd=.7, veHiDepth=30,
                 groupThrd=10, groupDiff=.1):

        self.crop = crop
        self.detectionPointSize = pointSize
        self.confidenceThrd = confidenceThrd

        bottomClip = imgInputShape[0] - crop[1]
        inH = imgInputShape[0] - crop[0] - bottomClip
        inW = imgInputShape[1]
        inCh = imgInputShape[2]

        self.cnnModel, cnnModelName = model.poolerPico(inputShape=(inH, inW, inCh))

        self.cnnModel.load_weights('{}.h5'.format(cnnModelName))

        self.veHiDepth = veHiDepth
        self.vehicleBoxesHistory = []
        self.groupThrd = groupThrd
        self.groupDiff = groupDiff

        # Custom kernel to consolidate diagonally-touching areas
        self.diagKernel = [[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]]

    # noinspection PyTypeChecker
    def vehicleScan(self, img):

        # Cropping to the region of interest
        roi = img[self.crop[0]:self.crop[1], :]

        roiW, roiH = roi.shape[1], roi.shape[0]

        # Going 4-D
        roi = np.expand_dims(roi, axis=0)

        # Single-Feature top convolutional layer, which represents a
        # miniaturized (25x153) version of the ROI with the vehicle's probability at each point
        detectionMap = self.cnnModel.predict(roi)

        predictionMapH, predictionMapW = detectionMap.shape[1], detectionMap.shape[2]

        ratioH, ratioW = roiH / predictionMapH, roiW / predictionMapW

        # Prediction output is 4-D tensor: (1, H, W, 1) in this particular case,
        # thus converting to 2-D, effectively represents it as a single-channel image
        detectionMap = detectionMap.reshape(detectionMap.shape[1], detectionMap.shape[2])

        # Thresholding by the confidence. The predictions are actually VERY polarized,
        # sticking to mostly Zero for non-vehicle points and mostly Ones for vehicles.
        # That said, midpoint of 0.5 for confidence threshold seems to be a reliable choice.
        detectionMap = detectionMap > self.confidenceThrd

        labels = label(detectionMap, structure=self.diagKernel)

        hotPoints = []

        # Considering obtained labels as vehicles.
        for vehicleID in range(labels[1]):
            nz = (labels[0] == vehicleID + 1).nonzero()
            nzY = np.array(nz[0])
            nzX = np.array(nz[1])

            # +/-'s are manually derived adjustments for more appropriate boxes visualization
            xMin = np.min(nzX) - 32
            xMax = np.max(nzX) + 32

            yMin = np.min(nzY)
            yMax = np.max(nzY) + 64

            # This used to keep generated bounding boxes within a range of the label (a.k.a. vehicle) boundaries
            spanX = xMax - xMin
            spanY = yMax - yMin

            for x, y in zip(nzX, nzY):

                # Adjustment offsets for a box starting point.
                # Ranges from 0 for the left(upper)-most to detectionPointSize for right(bottom)-most
                offsetX = (x - xMin) / spanX * self.detectionPointSize
                offsetY = (y - yMin) / spanY * self.detectionPointSize

                # Getting boundaries in ROI coordinates scale (multiplying by ratioW, ratioH)
                topLeftX = int(round(x * ratioW - offsetX, 0))
                topLeftY = int(round(y * ratioH - offsetY, 0))
                bottomLeftX = topLeftX + self.detectionPointSize
                bottomLeftY = topLeftY + self.detectionPointSize

                topLeft = (topLeftX, self.crop[0] + topLeftY)
                bottomRight = (bottomLeftX, self.crop[0] + bottomLeftY)

                hotPoints.append((topLeft, bottomRight))

        return hotPoints

    @staticmethod
    def addHeat(mask, bBoxes):
        """
        Creates the actual heat map. Overlaps build-up the 'heat'
        :param mask: the image where the 'heat' being projected
        :param bBoxes: bounding boxes formatted as list of tuples of tuples: [((x, y), (x, y)),...]
        :return: 1-channel Heat map image
        """
        for box in bBoxes:
            # box as ((x, y), (x, y))
            topY = box[0][1]
            bottomY = box[1][1]
            leftX = box[0][0]
            rightX = box[1][0]

            mask[topY:bottomY, leftX:rightX] += 1

            mask = np.clip(mask, 0, 255)

        return mask

    def getHotRegions(self, src):
        """
        Getting hot regions as a labeled features within a current frame + heat map for visualization
        :param src: current frame (3-channel RGB image)
        :return: hot regions as a labeled features, heat map
        """
        hotPoints = self.vehicleScan(img=src)
        sampleMask = np.zeros_like(src[:, :, 0]).astype(np.float)
        heatMap = self.addHeat(mask=sampleMask, bBoxes=hotPoints)

        # Filtering outliers
        #heatMap[heatMap <= 3] = 0
        currentFrameBoxes = label(heatMap, structure=self.diagKernel)

        return currentFrameBoxes, heatMap

    def updateHistory(self, currentLabels):
        """
        Converting hot regions to bounding boxes and saving them to boxes history list
        :param currentLabels: current frame hot regions as labeled features
        :return: void
        """
        for i in range(currentLabels[1]):
            nz = (currentLabels[0] == i + 1).nonzero()
            nzY = np.array(nz[0])
            nzX = np.array(nz[1])

            tlX = np.min(nzX)
            tlY = np.min(nzY)
            brX = np.max(nzX)
            brY = np.max(nzY)

            self.vehicleBoxesHistory.append([tlX, tlY, brX, brY])

            # Keeping last elements in history. Number defined by self.VeHiDepth
            self.vehicleBoxesHistory = self.vehicleBoxesHistory[-self.veHiDepth:]

    def relevantBoxes(self, src):
        """
        Extracts vehicles bounding boxes
        :param src: currentFrame
        :return: boxes, color heat map (for visualization)
        """
        currentLabels, heatMapGray = self.getHotRegions(src=src)

        heatColor = aux.colorHeatMap(heatMapMono=heatMapGray, cmap=cv2.COLORMAP_JET)

        self.updateHistory(currentLabels=currentLabels)

        boxes, _ = cv2.groupRectangles(rectList=np.array(self.vehicleBoxesHistory).tolist(),
                                       groupThreshold=self.groupThrd, eps=self.groupDiff)
        return boxes, heatColor
