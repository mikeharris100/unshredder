#!/usr/bin/env python

from PIL import Image
import numpy as np
from itertools import permutations, chain
import os
import sys


def unshred(filename):
    pilimg = Image.open(filename)
    img = np.asarray(pilimg)

    shredwidth = find_shred_width(img)
    num_shreds = img.shape[1] / shredwidth

    # column indexes of leftmost pixels of each shred
    leftinds = range(0, img.shape[1], shredwidth)

    # and the rightmost...
    rightinds = range(shredwidth - 1, img.shape[1], shredwidth)

    diffs = []

    # compare each shred with all possible neighbours
    for (left, right) in permutations(range(num_shreds), 2):
        lpix = img[:, rightinds[left], :].flatten(1)
        rpix = img[:, leftinds[right], :].flatten(1)
        # calculate a distance metric, Euclid will do for now
        diff = 10 * (sum((lpix - rpix) ** 2) ** 0.5)
        diffs.append((left, right, diff))

    pairs = []
    # loop through all shreds
    for shred in xrange(num_shreds):
        # find most likely neighbours
        pair = sorted(diffs, key=lambda x: x[2])[0]
        # remove those neighbours (on the same side) from future comparisons
        diffs = filter(lambda x: x[0] != pair[0] and x[1] != pair[1], diffs)
        pairs.append(pair)

    # initialise shreds with the least likely neighbours
    # (will hopefully be the edges)
    shreds = [sorted(pairs, key=lambda x:x[2])[-1][1]]
    # walk through pairs to generate l-r shreds
    for shred in xrange(1, num_shreds):
        shreds.append(filter(lambda x: x[0] == shreds[-1], pairs)[0][1])

    # convert shred indexes to column indexes
    cols = list(chain(*[range(shredwidth * shred, shredwidth * (shred + 1))
        for shred in shreds]))

    # arrange image
    joined = img[:, cols, :]

    # write out
    output = Image.fromarray(joined)
    fname, fext = os.path.splitext(filename)
    unshreddedfile = fname + "_UNSHREDDED" + fext
    output.save(unshreddedfile)
    print("Unshreddified %s" % unshreddedfile)


def find_shred_width(img):
    h, w, c = img.shape

    stacked = img.transpose((2, 0, 1)).reshape(h * c, w)

    diffs = np.sum((stacked[:, 1:] - stacked[:, :-1]) ** 2, axis=0) ** 0.5

    peaks = diffs[1:] - diffs[:-1]  # first derivative
    peaks2 = peaks[:-1] - peaks[1:]  # second

    # we care about onset
    peaks2[peaks2 < 0] = 0

    # normalise
    peaks2 = peaks2 / peaks2.max()

    # pick some threshold
    peakpositions = np.nonzero(peaks2 > 0.5)[0]

    # correct positions
    peakpositions = peakpositions + 2

    # explicitly add image width to search criteria
    peakpositions = np.append(peakpositions, w)

    # hopefully shred_width = gcd of peaks
    return _noisygcd(peakpositions)


def _noisygcd(nums):
    """
    Find the best fitting GCD for noisy data
    """
    candidates = []
    for i in range(nums.size):
        candidate = float(nums[i])
        goodness = 999  # initialise to some high value
        n = 0
        while goodness > 0.1:
            n += 1
            n_candidate = candidate / n
            fit = np.concatenate((n_candidate / nums[nums < n_candidate],
                nums[nums > n_candidate] / n_candidate))
            goodness = (fit - fit.round()).std()

        candidates.append(n_candidate)
    return int(np.mean(candidates).round())


def main():
    if len(sys.argv) != 2:
        print("Yo dawg, Ima need a file to process...")
        sys.exit(1)

    unshred(sys.argv[1])
    return 0

if __name__ == "__main__":
    status = main()
    sys.exit(status)
