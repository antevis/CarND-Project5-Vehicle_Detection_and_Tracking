import pickle
import cv2
import os
import numpy as np
import math
import helper as aux


class Processing:

    def __init__(self):
        calibrationFile = 'calibration_data.p'

        if not os.path.isfile(calibrationFile):
            self.camMtx = None
            self.distortionCoeffs = None
        else:
            with open(calibrationFile, mode='rb') as f:
                data = pickle.load(f)

            self.camMtx = data['cameraMatrix']
            self.distortionCoeffs = data['distCoeffs']

    def undistort(self, src):
        """
        Distortion correction using available camera matrix and distortion coefficients
        :param src: source image captured with the same camera as available matrix and distortion
        coefficients for
        :return: image with corrected distortion
        """
        if self.camMtx is not None and self.distortionCoeffs is not None:

            return cv2.undistort(src, self.camMtx, self.distortionCoeffs, None, self.camMtx)
        else:
            return None

    @staticmethod
    def histEq(src):
        """
        Histogram equalization
        :param src: gray or rgb image. The latter to be converted to HLS and histogram
        equalization applied to Luminance space, then converted back to RGB
        :return: histogram-equalized image
        """

        if len(src.shape) > 2:

            src = Thresholding.hls(src)

            src[:, :, 1] = cv2.equalizeHist(src[:, :, 1])

            return cv2.cvtColor(src, cv2.COLOR_HLS2RGB)

        else:
            return cv2.equalizeHist(src)

    @staticmethod
    def resize(src, ratio, interpolation=cv2.INTER_AREA):
        """
        Convenience wrapper for OpenCV resize function
        :param src: 
        :param ratio: 
        :param interpolation: 
        :return: 
        """
        return cv2.resize(src=src, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=interpolation)


class Thresholding:
    def __init__(self):
        pass

    @staticmethod
    def gradAbsolute(img, orient='x', sobelKernel=3, threshold=(0, 255)):

        """
        absolute gradient with sobel operator
        :param img: source image
        :param orient: direction of the gradient
        :param sobelKernel: 
        :param threshold: 
        :return: binary imaged obtained by applying Sobel with given params
        """

        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        else:
            gray = img

        dx = orient == 'x'
        dy = not dx
        sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=sobelKernel)
        absSobel = np.absolute(sobel)
        scaledSobel = np.uint8(255 * absSobel / np.max(absSobel))
        sBinary = np.zeros_like(gray)
        sBinary[(scaledSobel > threshold[0]) & (scaledSobel < threshold[1])] = 1

        return sBinary

    @staticmethod
    def gradMagnitude(img, sobelKernel=3, threshold=(0, 255)):
        """
        overall magnitude of the gradient, in both x and y.
        :param img: 
        :param sobelKernel: 
        :param threshold: 
        :return: 
        """

        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        else:
            gray = img

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobelKernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobelKernel)
        sobelAbs = np.sqrt(np.square(sobelx) + np.square(sobely))
        scaledSobel = np.uint8(255 * sobelAbs / np.max(sobelAbs))
        sBinary = np.zeros_like(gray)
        sBinary[(scaledSobel > threshold[0]) & (scaledSobel < threshold[1])] = 1

        return sBinary

    @staticmethod
    def gradDirectional(img, sobel_kernel=3, threshold=(0, np.pi / 2)):
        """
        Directional gradient using Sobel
        :param img: 
        :param sobel_kernel: 
        :param threshold: 
        :return: 
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        absSobelX = np.absolute(sobelX)
        absSobelY = np.absolute(sobelY)
        gradDir = np.arctan2(absSobelY, absSobelX)
        sBinary = np.zeros_like(gradDir)
        sBinary[(gradDir > threshold[0]) & (gradDir < threshold[1])] = 1

        return sBinary

    @staticmethod
    def binaryImage(gray, threshold=(0, 255)):
        """
        Converts single channel to binary according to given thresholding
        :param gray: single-channel image
        :param threshold: thresholds to be applied for binary clamping
        :return: binary image
        """
        binary = np.zeros_like(gray)
        binary[(gray > threshold[0]) & (gray <= threshold[1])] = 1

        return binary

    @staticmethod
    def hls(src):
        """
        Converts RGB to HLS
        :param src: source image
        :return: image in HLS color space
        """
        return cv2.cvtColor(src, cv2.COLOR_RGB2HLS)

    @staticmethod
    def yuv(src):
        """
        Converts RGB to YUV
        :param src: 
        :return: 
        """
        return cv2.cvtColor(src, cv2.COLOR_RGB2YUV)

    @staticmethod
    def combiThreshold(src, redThrsd=(200, 255), satThrsd=(90, 255),
                       absThrsd=(20, 255), orient='x'):
        """
        Returns combined binary image according to rule: 
        (redBinary & saturationBinary | abs binary)
        :param src: source image
        :param redThrsd: red channel thresholds
        :param satThrsd: HL(S)aturation channel thresholds
        :param absThrsd: absolute sobel thresholds
        :param orient: sobel orientation
        :return: combined binary thresholded image
        """
        hlsImg = Thresholding.hls(src=src)

        rChannel = src[:, :, 0]
        sChannel = hlsImg[:, :, 2]

        rBinary = Thresholding.binaryImage(gray=rChannel, threshold=redThrsd)
        sBinary = Thresholding.binaryImage(gray=sChannel, threshold=satThrsd)
        absBinary = Thresholding.gradAbsolute(img=src, orient=orient, threshold=absThrsd)

        combined = np.zeros_like(rBinary)
        combined[((rBinary == 1) & (sBinary == 1) | absBinary == 1)] = 1

        return combined


class Warping:
    def __init__(self):
        """
           0--3
          /    \
         1______2
        """
        self.src = np.array([[596, 450],
                             [204, 719],
                             [1108, 719],
                             [684, 450]], dtype=np.float32)

        self.dst = np.array([[204, 0],
                             [204, 719],
                             [1108, 719],
                             [1108, 0]], dtype=np.float32)

        self.M = cv2.getPerspectiveTransform(self.src, self.dst)

        self.mInv = cv2.getPerspectiveTransform(self.dst, self.src)

    @staticmethod
    def warp(srcImg, matrix):
        """
        Wrapper for warper:)
        :param srcImg: 
        :param matrix: 
        :return: 
        """
        return cv2.warpPerspective(srcImg, matrix, (srcImg.shape[1], srcImg.shape[0]),
                                   flags=cv2.INTER_LINEAR)

    def birdEye(self, img, leftShift=0):
        """
        perspective to bird-eye projection
        :param img: 
        :param leftShift: 
        :return: 
        """
        if leftShift == 0:
            return self.warp(srcImg=img, matrix=self.M)
        else:
            src = self.src.copy()
            src[:, 0] = src[:, 0] + leftShift

            dst = self.dst.copy()
            dst[:, 0] = dst[:, 0] + leftShift

            mInv = cv2.getPerspectiveTransform(src, dst)
            return self.warp(srcImg=img, matrix=mInv)

    def perspective(self, img, leftShift=0):
        """
        the inverse of bird-eye.
        Could be re-factored to reuse birdEye(), but that won't add clarity in this case
        :param img: 
        :param leftShift: 
        :return: 
        """
        if leftShift == 0:
            return self.warp(srcImg=img, matrix=self.mInv)
        else:

            src = self.src.copy()
            src[:, 0] = src[:, 0] + leftShift

            dst = self.dst.copy()
            dst[:, 0] = dst[:, 0] + leftShift

            mInv = cv2.getPerspectiveTransform(dst, src)
            return self.warp(srcImg=img, matrix=mInv)


class Drawing:
    @staticmethod
    def addPolygon(srcShape, lFit, rFit, stepCount=10, color=(0, 255, 0)):
        """
        Adds polygon to empty image with a shape of a source
        :param srcShape: shape of a source image
        :param lFit: left 2-nd order polynomial line params
        :param rFit: right 2-nd order polynomial line params
        :return: image with polygon between lines
        :param stepCount: number of steps
        :param color: 
        """

        imgH = srcShape[0]
        imgW = srcShape[1]

        cL = lFit[2]

        cR = rFit[2]

        mask = np.zeros(shape=srcShape, dtype=np.uint8)

        rOutstand = max(0, (int(math.ceil(cR)) - (imgW - 1)))

        if rOutstand > 0:
            r_filler = np.zeros((imgH, rOutstand), dtype=np.uint8)
            mask = np.hstack((mask, r_filler))

        lOutstand = abs(min(0, int(math.floor(cL))))

        if lOutstand > 0:
            lFiller = np.zeros((imgH, lOutstand), dtype=np.uint8)
            mask = np.hstack((lFiller, mask))

        outImg = np.dstack((mask, mask, mask))

        y = np.linspace(0, imgH - 1, stepCount)

        xl = aux.funcSpace(argSpace=y, fitParams=lFit)
        xr = aux.funcSpace(argSpace=y, fitParams=rFit)

        leftLinePoints = np.array([np.transpose(np.vstack([xl + lOutstand, y]))])
        rightLinePoints = np.array([np.flipud(np.transpose(np.vstack([xr + lOutstand, y])))])

        points = np.hstack((leftLinePoints, rightLinePoints))

        cv2.fillPoly(img=outImg, pts=np.int_(points), color=color)

        return outImg, lOutstand, rOutstand

    @staticmethod
    def addLine(src, fit, color, stepCount=10, thickness=2):

        if fit is not None:
            imgH = src.shape[0]

            y = np.linspace(0, imgH - 1, stepCount)

            x = aux.funcSpace(argSpace=y, fitParams=fit)

            for i in range(stepCount - 1):
                x1 = int(round(x[i], 0))
                y1 = int(round(y[i], 0))

                x2 = int(round(x[i + 1], 0))
                y2 = int(round(y[i + 1], 0))
                pt1 = (x1, y1)
                pt2 = (x2, y2)
                src = cv2.line(img=src, pt1=pt1, pt2=pt2, color=color, thickness=thickness)

        return src
