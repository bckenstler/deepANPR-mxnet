"""
The MIT License (MIT)

Copyright (c) 2015 matthewearl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import mxnet as mx
from mxnet import nd, gluon
import numpy as np
import os
from os.path import join as path_join
from PIL import Image, ImageDraw, ImageFont
import cv2
import itertools
import math
import random
import sys
import glob
import collections

DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS = LETTERS + DIGITS
FCHARS = CHARS + " "
FONT_HEIGHT = 32
OUTPUT_SHAPE = (64, 128)
FCHARS = CHARS + " "
UK_FONT = "UKNumberPlate.ttf"
WINDOW_SHAPE = (64, 128)


def make_char_ims(font_path, output_height):
    font_size = output_height * 4
    font = ImageFont.truetype(font_path, font_size)

    height = max(font.getsize(c)[1] for c in FCHARS)

    for c in FCHARS:
        width = font.getsize(c)[0]
        im = Image.new("RGBA", (width, height), (0, 0, 0))

        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, (255, 255, 255), font=font)
        scale = float(output_height) / height
        im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
        yield c, np.array(im)[:, :, 0].astype(np.float32) / 255.

def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = np.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = np.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = np.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M

    return M

def pick_colors():
    first = True
    while first or plate_color - text_color < 0.3:
        text_color = random.random()
        plate_color = random.random()
        if text_color > plate_color:
            text_color, plate_color = plate_color, text_color
        first = False
    return text_color, plate_color

def make_affine_transform(from_shape, to_shape, 
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    out_of_bounds = False

    from_size = np.array([[from_shape[1], from_shape[0]]]).T
    to_size = np.array([[to_shape[1], to_shape[0]]]).T

    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)
    if scale > max_scale or scale < min_scale:
        out_of_bounds = True
    roll = random.uniform(-0.3, 0.3) * rotation_variation
    pitch = random.uniform(-0.2, 0.2) * rotation_variation
    yaw = random.uniform(-1.2, 1.2) * rotation_variation

    # Compute a bounding box on the skewed input image (`from_shape`).
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h, w = from_shape
    corners = np.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    skewed_size = np.array(np.max(M * corners, axis=1) -
                              np.min(M * corners, axis=1))

    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    scale *= np.min(to_size / skewed_size)

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    trans = (np.random.random((2,1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if np.any(trans < -0.5) or np.any(trans > 0.5):
        out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = np.hstack([M, trans + center_to - M * center_from])

    return M, out_of_bounds

def generate_code():
    return "{}{}{}{} {}{}{}".format(
        random.choice(LETTERS),
        random.choice(LETTERS),
        random.choice(DIGITS),
        random.choice(DIGITS),
        random.choice(LETTERS),
        random.choice(LETTERS),
        random.choice(LETTERS))

def rounded_rect(shape, radius):
    out = np.ones(shape)
    out[:radius, :radius] = 0.0
    out[-radius:, :radius] = 0.0
    out[:radius, -radius:] = 0.0
    out[-radius:, -radius:] = 0.0

    cv2.circle(out, (radius, radius), radius, 1.0, -1)
    cv2.circle(out, (radius, shape[0] - radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, shape[0] - radius), radius, 1.0, -1)

    return out

def generate_plate(font_height, char_ims):
    h_padding = random.uniform(0.2, 0.4) * font_height
    v_padding = random.uniform(0.1, 0.3) * font_height
    spacing = font_height * random.uniform(-0.05, 0.05)
    radius = 1 + int(font_height * 0.1 * random.random())

    code = generate_code()
    text_width = sum(char_ims[c].shape[1] for c in code)
    text_width += (len(code) - 1) * spacing

    out_shape = (int(font_height + v_padding * 2),
                 int(text_width + h_padding * 2))

    text_color, plate_color = pick_colors()
    
    text_mask = np.zeros(out_shape)
    
    x = h_padding
    y = v_padding 
    for c in code:
        char_im = char_ims[c]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + spacing

    plate = (np.ones(out_shape) * plate_color * (1. - text_mask) +
             np.ones(out_shape) * text_color * text_mask)

    return plate, rounded_rect(out_shape, radius), code.replace(" ", "")

def generate_bg(path, num_bg_images):
    found = False
    while not found:
        fname = "bgs/{:08d}.jpg".format(random.randint(0, num_bg_images - 1))
        bg = cv2.imread(path_join(path,fname), cv2.IMREAD_GRAYSCALE) / 255.
        if (bg.shape[1] >= OUTPUT_SHAPE[1] and
            bg.shape[0] >= OUTPUT_SHAPE[0]):
            found = True

    x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1])
    y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0])
    bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]

    return bg

def generate_im(path, char_ims, num_bg_images):
    bg = generate_bg(path, num_bg_images)

    plate, plate_mask, code = generate_plate(FONT_HEIGHT, char_ims)
    
    M, out_of_bounds = make_affine_transform(
                            from_shape=plate.shape,
                            to_shape=bg.shape,
                            min_scale=0.6,
                            max_scale=0.875,
                            rotation_variation=1.0,
                            scale_variation=1.5,
                            translation_variation=1.2)
    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))

    out = plate * plate_mask + bg * (1.0 - plate_mask)

    out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))

    out += np.random.normal(scale=0.05, size=out.shape)
    out = np.clip(out, 0., 1.)

    return out, code, not out_of_bounds

def load_fonts(folder_path):
    font_char_ims = {}
    fonts = [f for f in os.listdir(folder_path) if f.endswith('.ttf')]
    for font in fonts:
        font_char_ims[font] = dict(make_char_ims(os.path.join(folder_path,
                                                              font),
                                                 FONT_HEIGHT))
    return fonts, font_char_ims

def generate_ims(path, fonts = [UK_FONT]):
    """
    Generate number plate images.
    :return:
        Iterable of number plate images.
    """
    font_path = path_join(path, "fonts")
    variation = 1.0
    if not fonts:
        fonts, font_char_ims = load_fonts(font_path)
    else:
        font_char_ims = {UK_FONT:dict(make_char_ims(os.path.join(font_path, UK_FONT), FONT_HEIGHT))}
    num_bg_images = len(os.listdir(path_join(path,"bgs")))
    while True:
        yield generate_im(path, font_char_ims[random.choice(fonts)], num_bg_images)

def code_to_vec(p, code):
    def char_to_vec(c):
        y = np.zeros((len(CHARS),))
        y[CHARS.index(c)] = 1.0
        return y

    c = np.vstack([char_to_vec(c) for c in code])

    return np.concatenate([[1. if p else 0], c.flatten()])


def read_data(img_glob, batch_size):
    i = 0
    ims = []
    labels = []
    for fname in sorted(glob.glob(img_glob)):
        im = cv2.imread(fname)[:, :, 0].astype(np.float32) / 255.
        code = fname.split("/")[-1][9:16]
        p = fname.split("/")[-1][17] == '1'
        ims.append(nd.expand_dims(nd.array(im), 0))
        labels.append(nd.array(code_to_vec(p, code)))
        i += 1
        if i == batch_size:
            yield nd.stack(*ims), nd.stack(*labels)
            ims = []
            labels = []
            i = 0


def unzip(b):
    xs, ys = zip(*b)
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys


def batch(it, batch_size):
    out = []
    for x in it:
        out.append(x)
        if len(out) == batch_size:
            yield out
            out = []
    if out:
        yield out
        
def read_batches(path, batch_size, iterations):
    g = generate_ims(path)
    i = 0
    def gen_vecs():
        for im, c, p in itertools.islice(g, batch_size):
            yield np.expand_dims(im, 0), code_to_vec(p, c)

    while True:
        x, y = unzip(gen_vecs())
        yield nd.array(x), nd.array(y)
        i += 1
        if i == iterations:
            raise StopIteration

def make_scaled_ims(im, min_shape):
    ratio = 1. / 2 ** 0.5
    shape = (im.shape[0] / ratio, im.shape[1] / ratio)

    while True:
        shape = (int(shape[0] * ratio), int(shape[1] * ratio))
        if shape[0] < min_shape[0] or shape[1] < min_shape[1]:
            break
        yield cv2.resize(im, (shape[1], shape[0]))

def detect(im, net, ctx):
    """
    Detect number plates in an image.
    :param im:
        Image to detect number plates in.
    :param param_vals:
        Model parameters to use. These are the parameters output by the `train`
        module.
    :returns:
        Iterable of `bbox_tl, bbox_br, letter_probs`, defining the bounding box
        top-left and bottom-right corners respectively, and a 7,36 matrix
        giving the probability distributions of each letter.
    """

    # Convert the image to various scales.
    scaled_ims = list(make_scaled_ims(im, WINDOW_SHAPE))

    # Execute the model at each scale.
    y_vals = []
    for scaled_im in scaled_ims:
        scaled_im = nd.expand_dims(nd.expand_dims(nd.array(scaled_im), 0), 0)
        y_vals.append(np.moveaxis(net(scaled_im.as_in_context(ctx)).asnumpy(),[0,1,2,3],[0,-1,1,2]))

    # Interpret the results in terms of bounding boxes in the input image.
    # Do this by identifying windows (at all scales) where the model predicts a
    # number plate has a greater than 50% probability of appearing.
    #
    # To obtain pixel coordinates, the window coordinates are scaled according
    # to the stride size, and pixel coordinates.
    for i, (scaled_im, y_val) in enumerate(zip(scaled_ims, y_vals)):
        for window_coords in np.argwhere(y_val[0, :, :, 0] >-math.log(1./0.99 - 1)):

            letter_probs = (y_val[0,
                                  window_coords[0],
                                  window_coords[1], 1:].reshape(
                                    7, len(CHARS)))

            letter_probs = gluon.ndarray.softmax(nd.array(letter_probs)).asnumpy()
            img_scale = float(im.shape[0]) / scaled_im.shape[0]

            bbox_tl = window_coords * (8, 4) * img_scale
            bbox_size = np.array(WINDOW_SHAPE) * img_scale

            present_prob = gluon.ndarray.sigmoid(
                               nd.array([y_val[0, window_coords[0], window_coords[1], 0]])).asscalar()

            yield bbox_tl, bbox_tl + bbox_size, present_prob, letter_probs

def letter_probs_to_code(letter_probs):
    return "".join(CHARS[i] for i in np.argmax(letter_probs, axis=1))

def _overlaps(match1, match2):
    bbox_tl1, bbox_br1, _, _ = match1
    bbox_tl2, bbox_br2, _, _ = match2
    return (bbox_br1[0] > bbox_tl2[0] and
            bbox_br2[0] > bbox_tl1[0] and
            bbox_br1[1] > bbox_tl2[1] and
            bbox_br2[1] > bbox_tl1[1])


def _group_overlapping_rectangles(matches):
    matches = list(matches)
    num_groups = 0
    match_to_group = {}
    for idx1 in range(len(matches)):
        for idx2 in range(idx1):
            if _overlaps(matches[idx1], matches[idx2]):
                match_to_group[idx1] = match_to_group[idx2]
                break
        else:
            match_to_group[idx1] = num_groups 
            num_groups += 1

    groups = collections.defaultdict(list)
    for idx, group in match_to_group.items():
        groups[group].append(matches[idx])

    return groups

def post_process(matches):
    """
    Take an iterable of matches as returned by `detect` and merge duplicates.
    Merging consists of two steps:
      - Finding sets of overlapping rectangles.
      - Finding the intersection of those sets, along with the code
        corresponding with the rectangle with the highest presence parameter.
    """
    groups = _group_overlapping_rectangles(matches)

    for group_matches in groups.values():
        mins = np.stack(np.array(m[0]) for m in group_matches)
        maxs = np.stack(np.array(m[1]) for m in group_matches)
        present_probs = np.array([m[2] for m in group_matches])
        letter_probs = np.stack(m[3] for m in group_matches)

        yield (np.max(mins, axis=0).flatten(),
               np.min(maxs, axis=0).flatten(),
               np.max(present_probs),
               letter_probs[np.argmax(present_probs)])

def predict(im, detector, ctx):
    im = np.copy(im)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255.
    for pt1, pt2, present_prob, letter_probs in post_process(
                                                  detect(im_gray, detector.net, ctx)):
        pt1 = tuple(reversed(map(int, pt1)))
        pt2 = tuple(reversed(map(int, pt2)))

        code = letter_probs_to_code(letter_probs)
        color = (0.0, 255.0, 0.0)
        cv2.rectangle(im, pt1, pt2, color)

        cv2.putText(im,
                    code,
                    pt1,
                    cv2.FONT_HERSHEY_PLAIN, 
                    1.5,
                    (0, 0, 0),
                    thickness=5)

        cv2.putText(im,
                    code,
                    pt1,
                    cv2.FONT_HERSHEY_PLAIN, 
                    1.5,
                    (255, 255, 255),
                    thickness=2)

    return im

def predict_io(in_file, out_file, detector, ctx):
    im = cv2.imread(in_file)
    im = predict(im, detector, ctx)
    cv2.imwrite(out_file, im)

