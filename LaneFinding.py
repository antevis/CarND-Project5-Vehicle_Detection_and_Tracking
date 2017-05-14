import numpy as np
from enum import Enum
import helper as aux
import imageProcessing as ip
import cv2


class LineSpace(Enum):
    """
    for clarity
    """
    LEFT = 0
    RIGHT = 1


class LaneFinding:
    @staticmethod
    def getWindowPoints(imgHeight, winHeight, winIdx, xCurr, leftMargin, rightMargin):
        """
        Used for box search to obtain coordinates of window for current iteration (winIdx).
        Almost replicates function introduced in the lesson with minor changes. 
        :param imgHeight: image height
        :param winHeight: window height
        :param winIdx: window index
        :param xCurr: current x coordinate
        :param leftMargin: left detectionPointSize
        :param rightMargin: right detectionPointSize
        :return: dictionary with 4 named values
        """
        xCurr = int(round(xCurr, 0))
        winTop = imgHeight - (winIdx + 1) * winHeight
        winBottom = winTop + winHeight
        winLeft = xCurr - leftMargin
        winRight = xCurr + rightMargin

        # print(winTop, winBottom, winLeft, winRight, xCurr)

        return {'winTop': winTop, 'winBottom': winBottom,
                'winLeft': winLeft, 'winRight': winRight}

    @staticmethod
    def getFit(lineInds, nzX, nzY):

        """
        generates x and y spaces according to given params and fits 2nd order polynomial (a,b,c)
        :param lineInds: 
        :param nzX: 
        :param nzY: 
        :return: 
        """
        x = nzX[lineInds]
        y = nzY[lineInds]

        if len(x) > 0 and len(y) > 0:
            return np.polyfit(y, x, 2)
        else:
            return None

    def primarySearchPolyMargin(self, src, lineSpace, winCount, margin, minpix,
                                nzX, nzY, windowSplit=2, xPrimary=None):
        """
        primary search when no previous fits available for guidance.
        Margin size varies depending of the slope
        :param src: source bird-eye binary
        :param lineSpace: line space: LineSpace.LEFT or LineSpace.RIGHT
        :param winCount: number of windows to use vertically
        :param margin: initial base detectionPointSize width in pixels
        :param minpix: number of valid pixels to shift window by X axis
        :param nzX: nonZero X
        :param nzY: nonZero Y
        :param windowSplit: lower 1 / this param used for histogram evaluation
        :param xPrimary: used in case this is not an initial search
        :return: fit + image with search process depicted
        """
        imgH = src.shape[0]

        imgW = src.shape[1]

        srcRgb = np.dstack((src, src, src)) * 255

        if xPrimary is None:
            lowHalf = src[int((windowSplit - 1) * imgH / windowSplit):, :]
            histogram = np.sum(lowHalf, axis=0)

            midpoint = int(histogram.shape[0] / 2)

            if lineSpace == LineSpace.LEFT:

                xPrimary = np.argmax(histogram[:midpoint])
            else:
                xPrimary = np.argmax(histogram[midpoint:]) + midpoint

        winHeight = np.int(imgH / winCount)

        lineInds = []

        lMargin = margin
        rMargin = margin

        for idx in range(winCount):
            winPoints = self.getWindowPoints(imgHeight=imgH, winHeight=winHeight, winIdx=idx,
                                             xCurr=xPrimary, leftMargin=lMargin, rightMargin=rMargin)
            winTop = winPoints['winTop']
            winBottom = winPoints['winBottom']
            winLeft = winPoints['winLeft']
            winRight = winPoints['winRight']

            winCenter = int((winLeft + winRight) / 2)

            if lMargin / 2 <= winCenter < imgW - 1 - int(rMargin / 2):

                aux.drawRect(img=srcRgb,
                             lx=winLeft, ly=winTop,
                             rx=winRight, ry=winBottom, color=(0, 255, 0))

                # Identify the nonzero pixels in x and y within the window
                validInds = ((nzY >= winTop) & (nzY < winBottom) &
                             (nzX >= winLeft) & (nzX < winRight)).nonzero()[0]

                lineInds.append(validInds)

                adj = 5
                #

                # If more indices than minpix pixels found, recentering next window on their mean position
                if len(validInds) > minpix:
                    # noinspection PyTypeChecker
                    newMean = np.mean(nzX[validInds]).astype(np.int)

                    deltaShift = newMean - xPrimary
                    shiftRatio = adj * deltaShift / imgW
                    lMargin -= lMargin * shiftRatio
                    rMargin += rMargin * shiftRatio
                    lMargin = int(lMargin)
                    rMargin = int(rMargin)

                    xPrimary = newMean

        if len(lineInds) > 0:
            lineInds = np.concatenate(lineInds)

        fit = LaneFinding.getFit(lineInds, nzX, nzY)

        srcRgb = ip.Drawing.addLine(src=srcRgb, fit=fit, color=[255, 0, 255], thickness=10)

        return fit, srcRgb

    def secondarySearch(self, imgH, previousFit, nzX, nzY, margin, src, ratio=1., lineData=None):
        """
        Look Ahead search, when previous fits exist for guidance
        :param imgH: image height
        :param previousFit: previous fit used as a guidance
        :param nzX: nonZero X
        :param nzY: nonZero Y
        :param margin: detectionPointSize used for creation of search area
        :param src: source bird-eye binary
        :param ratio: used to limit search area by height
        :param lineData: used for visualization of available fits
        :return: fit + image with search process depicted
        """

        ySplit = int(imgH * (1 - ratio))

        filterKey = nzY >= ySplit

        nzY = nzY[filterKey]
        nzX = nzX[filterKey]

        srcRgb = np.dstack((src, src, src)) * 255

        leftFit, rightFit = self.borderFit(imgH=imgH, centralFit=previousFit, margin=margin)

        leftX = aux.funcSpace(argSpace=nzY, fitParams=leftFit)
        rightX = aux.funcSpace(argSpace=nzY, fitParams=rightFit)

        lineInds = ((nzX > leftX) & (nzX < rightX))

        fit = LaneFinding.getFit(lineInds, nzX, nzY)

        if lineData is not None:
            color = [255, 0, 0] if lineData['lineSpace'] == LineSpace.LEFT else [0, 0, 255]

            for lineFit in lineData['fits']:
                srcRgb = ip.Drawing.addLine(src=srcRgb, fit=lineFit, color=color, thickness=3, stepCount=20)

        srcRgb = ip.Drawing.addLine(src=srcRgb, fit=fit, color=[255, 0, 255], thickness=10, stepCount=20)

        winImg = np.zeros_like(srcRgb)

        y = np.linspace(start=0, stop=imgH - 1, num=imgH)

        xl = aux.funcSpace(argSpace=y, fitParams=leftFit)
        xr = aux.funcSpace(argSpace=y, fitParams=rightFit)

        line_window1 = np.array([np.transpose(np.vstack([xl, y]))])
        line_window2 = np.array([np.flipud(np.transpose(np.vstack([xr, y])))])
        line_pts = np.hstack((line_window1, line_window2))

        cv2.fillPoly(winImg, np.int_([line_pts]), (0, 255, 0))

        srcRgb = cv2.addWeighted(srcRgb, 1, winImg, 0.5, 0)

        return fit, srcRgb

    @staticmethod
    def borderFit(imgH, centralFit, margin):
        """
        Generates a region of interest outstanding to a given detectionPointSize from the center fit 
        as a perpendicular to the tangent line at each point
        :param imgH: image height
        :param centralFit: central fit
        :param margin: 
        :return: left and right fits approximating area tangentially equidistant 
        from central fit at both sides
        """

        a = centralFit[0]
        b = centralFit[1]

        y = np.linspace(start=0, stop=imgH - 1, num=imgH)
        x = aux.funcSpace(argSpace=y, fitParams=centralFit)

        dy = 2 * a * y + b  # Line slope as the derivative

        alpha = np.arctan(dy)  # Negative for negative slopes
        deltaY = abs(margin * np.sin(alpha))  # Negative for negative alphas
        deltaX = margin * np.cos(alpha)

        xLeft = x - deltaX
        xRight = x + deltaX

        yLeft = y + deltaY * np.sign(dy)
        yRight = y - deltaY * np.sign(dy)

        leftFit = np.polyfit(x=yLeft, y=xLeft, deg=2)
        rightFit = np.polyfit(x=yRight, y=xRight, deg=2)

        return leftFit, rightFit

    @staticmethod
    def convolutional_search(bin_wrap_img, win_width, win_height, margin, init_ratio):

        img_h = bin_wrap_img.shape[0]
        img_w = bin_wrap_img.shape[1]

        ys = []
        lxs = []
        rxs = []
        window = np.ones(win_width)

        height_share = 1 - init_ratio

        top = int(img_h * height_share)
        bottom = img_h - 1
        left = 0
        mid = int(img_w / 2)
        right = img_w - 1

        l_sum = np.sum(bin_wrap_img[top:bottom, left:mid], axis=0)
        r_sum = np.sum(bin_wrap_img[top:bottom, mid:right], axis=0)

        l_center = np.argmax(np.convolve(window, l_sum)) - int(win_width / 2)
        r_center = np.argmax(np.convolve(window, r_sum)) - int(win_width / 2) + int(img_w / 2)

        ys.append(int((top + bottom) / 2))
        lxs.append(l_center)
        rxs.append(r_center)

        for idx in range(1, int(img_h / win_height)):
            top = img_h - (idx + 1) * win_height
            bottom = top + win_height

            image_slice = np.sum(bin_wrap_img[top:bottom, :], axis=0)

            conv_signal = np.convolve(window, image_slice)

            offset = int(win_width / 2)

            # noinspection PyTypeChecker
            l_min_index = int(max(l_center + offset - margin, 0))
            l_max_index = int(min(l_center + offset + margin, img_w))

            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset

            # noinspection PyTypeChecker
            r_min_index = int(max(r_center + offset - margin, 0))
            r_max_index = int(min(r_center + offset + margin, img_w))

            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset

            ys.append(int((top + bottom) / 2))
            lxs.append(l_center)
            rxs.append(r_center)

        return np.polyfit(ys, lxs, 2), np.polyfit(ys, rxs, 2)
