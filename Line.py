import numpy as np
from LaneFinding import LaneFinding
import helper as aux


class Line:
    def __init__(self, lineSpace, historyDepth=5, margin=100, windowSplit=2, winCount=9, searchPortion=1.):
        self.lineSpace = lineSpace
        self.fits = np.empty((0, 3), float)
        self.depth = historyDepth
        self.margin = margin
        self.windowSplit = windowSplit
        self.winCount = winCount
        self.searchPortion = searchPortion

    def resetFits(self):
        """
        removes all previously found fits
        :return: void
        """
        self.fits = np.empty((0, 3), float)

    def reScanWithPrimary(self, src):
        """
        Allows to initiate box search with xPrimary obtained from previously found fits.
        :param src: binary bird-eye image
        :return: fit + fitType + image with search process depicted
        fitType used for debugging purposes (may be ignored)
        """

        imgH = src.shape[0]

        yBottom = imgH - 1

        self.fits = self.fits[:len(self.fits) - 1]  # Remove newest fit

        finder = LaneFinding()

        nzX, nzY = self.nz(src=src)

        currFit = self.currentFit()

        if currFit is not None:
            x_primary = aux.funcSpace(yBottom, currFit)
        else:
            x_primary = None

        fit, srcRgb = finder.primarySearchPolyMargin(src=src, lineSpace=self.lineSpace,
                                                     winCount=self.winCount, margin=self.margin, minpix=50,
                                                     nzX=nzX, nzY=nzY, windowSplit=self.windowSplit,
                                                     xPrimary=x_primary)
        if fit is not None:
            self.addFit(fit)

        return self.currentFit(), 'reScan', srcRgb

    def addFit(self, fit):
        """
        keeps the number of newest fits not exceeding the defined history depth
        :param fit: fit to add
        :return: void
        """
        self.fits = np.vstack((self.fits, np.array(fit)))
        self.fits = self.fits[-self.depth:]

    def currentFit(self):
        """
        Computes current line fit as a weighted average
        :return: fit (a, b, c)
        """

        weights = self.getWeights()

        if weights is not None:

            a = np.sum(self.fits[:, 0] * weights) / np.sum(weights)
            b = np.sum(self.fits[:, 1] * weights) / np.sum(weights)
            c = np.sum(self.fits[:, 2] * weights) / np.sum(weights)

            return [a, b, c]
        else:
            return None

    def getWeights(self):
        """
        computes weights according to the following logic:
        * Slope weights take steepness of curvature (A in polynomial coefficients) and give more weight 
            to more vertical lines
        * Vertex weights are the ```y``` coordinate where parabola turns.
            Considered that the lower it is, the higher the weight should be
        * Age weights - the younger the fit, the heavier its weight.

        Then I compute cumulative weight as a product of all three and normalize it between 0 and 1
        :return: average normalized weights
        """

        if len(self.fits) > 0:
            slope_weights = 1 / abs(self.fits[:, 0])
            vertex_weights = -self.fits[:, 1] / (self.fits[:, 0] * 2)

            age_weights = 1 / np.flipud(np.linspace(1, len(self.fits), len(self.fits)))

            w = slope_weights * vertex_weights * age_weights

            normal_w = (w - np.min(w)) / (np.max(w) - np.min(w)) if np.max(w) != np.min(w) else w

            return normal_w
        else:
            return None

    @staticmethod
    def nz(src, full=False, ratio=1):
        """
        Convenience wrapper for numpy.nonzero() function
        :param src: source binary image
        :param full: flag to determine whether to return all or just nzX and nzY
        :param ratio: allows to take partial nonzeros along zeroth axis ('y' in terms of numpy convention)
        :return: 
        """

        imgH = src.shape[0]

        vSplit = int(imgH * (1 - ratio))

        nonZero = src[vSplit:, :].nonzero()
        nzY = np.array(nonZero[0])
        nzX = np.array(nonZero[1])

        if full:
            return nonZero, nzX, nzY
        else:
            return nzX, nzY

    def getFit(self, src):
        """
        Performs one of 2 types of search: initial box search or look ahead, depending on available fits
        :param src: binary bird-eye
        :return: fit + fitType + image with search process depicted
        """

        fitType = 'primary'

        finder = LaneFinding()

        current_fit = self.currentFit()

        nzX, nzY = self.nz(src=src)

        if current_fit is None:

            """
            def primarySearchPolyMargin(self, src, lineSpace, winCount, detectionPointSize, minpix,
                                nzX, nzY, windowSplit=2, xPrimary=None):
                                """

            fit, src_rgb = finder.primarySearchPolyMargin(src=src, lineSpace=self.lineSpace,
                                                          winCount=self.winCount, margin=self.margin, minpix=50,
                                                          nzX=nzX, nzY=nzY, windowSplit=self.windowSplit)
        else:
            fit, src_rgb = finder.secondarySearch(imgH=src.shape[0], previousFit=current_fit, nzX=nzX, nzY=nzY,
                                                  margin=self.margin, src=src, ratio=self.searchPortion,
                                                  lineData={'lineSpace': self.lineSpace, 'fits': self.fits})
            fitType = 'lookAhead'
        if fit is not None:
            self.addFit(fit)

        return self.currentFit(), fitType, src_rgb

    def reScanJustified(self):

        """
        Determines whether it makes sense to perform re-scan with box search
        :return: bool
        """

        if len(self.fits) > 1:
            fitParamsFull = self.fits.T
            fitParamsNoLast = self.fits[:-1].T

            for i in range(len(fitParamsFull)):
                stdFull = np.std(fitParamsFull[i])
                stdNoLast = np.std(fitParamsNoLast[i])

                if (stdNoLast != 0 and stdFull / stdNoLast >= 1.5) or stdNoLast == 0:
                    return True

            return False
        return False
