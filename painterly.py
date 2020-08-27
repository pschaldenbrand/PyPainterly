import cv2
import numpy as np

#import matplotlib.pyplot as plt

MAX_ITERATIONS = 100

def normal_x(x, width):
    return (int)(x * (width - 1) + 0.5)
def normal_y(y, height):
    return (int)(y * (height - 1) + 0.5)

def draw(f, width=128, height=128):
    """ Draw a stroke onto a blank canvas
    Parameters
    ----------
    f : []
        Definition of bezier curve: x0, y0, x1, y1, x2, y2, width_start, width_end, opacity_start, opacity_end
    width : int, optional
        Width of canvas. (Default 128)
    height : int, optional
        Height of canvas. (Default 128)
    
    Returns
    -------
    np.array[width, height]
        matrix (boolean map) with the stroke drawn on it.
    """
    x0, y0, x1, y1, x2, y2, z0, z2, w0, w2 = f

    frac = 1. / MAX_ITERATIONS

    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1
    x0 = normal_x(x0, width * 2)
    x1 = normal_x(x1, width * 2)
    x2 = normal_x(x2, width * 2)
    y0 = normal_y(y0, height * 2)
    y1 = normal_y(y1, height * 2)
    y2 = normal_y(y2, height * 2)
    z0 = (int)(1 + z0 * width // 2)
    z2 = (int)(1 + z2 * width // 2)
    canvas = np.zeros([width * 2, height * 2]).astype('float32')
    
    for i in range(MAX_ITERATIONS):
        t = i * frac
        x = (int)((1-t) * (1-t) * x0 + 2 * t * (1-t) * x1 + t * t * x2)
        y = (int)((1-t) * (1-t) * y0 + 2 * t * (1-t) * y1 + t * t * y2)
        z = (int)((1-t) * z0 + t * z2)
        w = (1-t) * w0 + t * w2
        cv2.circle(canvas, (y, x), z, w, -1)
    return 1 - cv2.resize(canvas, dsize=(height, width))

def draw_spline_stroke(K, r, width=128, height=128):
    """
    Paint a stroke defined by a list of points onto a canvas
    
    args:
        K (List[Tup(int, int)]) : a nested list of points to draw. [(x_pixel, y_pixel),...]
        r (int) : radius in pixels of stroke
    
    kwargs:
        width (int) :  Width of canvas. (Default 128)
        height (int) :  Height of canvas. (Default 128)
    
    return:
        np.array[width, height] : matrix (boolean map) with the stroke drawn on it.
    """
    canvas = np.zeros([width, height]).astype('float32')

    for f in K:
        x = f[0]
        y = f[1]
        z = r
        w = 1.
        cv2.circle(canvas, (y, x), z, w, -1)
    return 1 - cv2.resize(canvas, dsize=(height, width))

def make_stroke(r, x0, x1, y0, y1, width, height):
    """
    Draw a straight line on a canvas
    
    args:
        r (int) : radius in pixels of stroke
        x0 (int) : starting x pixel
        x1 (int) : ending x pixel
        y0 (int) : starting y pixel
        y1 (int) : ending y pixel
        width (int) :  Width of canvas. (Default 128)
        height (int) :  Height of canvas. (Default 128)
    
    return:
        np.array[width, height] : matrix (boolean map) with the stroke drawn on it.
    """
    # f is (x0, y0, x1, y1, x2, y2, width_start, width_end, opacity_start, opacity_end)
    f = (x0, y0, (x1-x0)/2 + x0, (y1-y0)/2 + y0, x1, y1, r, r, 1., 1.)

    return draw(f, width=width, height=height)

# cache gradients
gradient, grad_x, grad_y = None, None, None

def make_spline_stroke(x0, y0, R, ref_image, canvas, max_stroke_length=None, fc=1):
    """
    Draw a curved line on a canvas from a starting point based on gradients
    
    args:
        x0 (int) : Starting x pixel
        x1 (int) : Ending x pixel
        R (int) : Radius in pixels of stroke
        ref_image (np.array[width, height, 3]) :  Reference image 0-255 RGB
        canvas (np.array[width, height, 3]) :  Current painting canvas 0-1 RGB
    
    kwargs:
        max_stroke_length (int) : Maximum length of a stroke in pixels.
        fc (float) : Curvature filter - used to limit or exaggerate stroke curvature. Default 1
    
    return:
        np.array[width, height] : Matrix (boolean map) with the stroke drawn on it.
    """
    stroke_color = ref_image[x0,y0,:]
    K = [(x0,y0)]
    
    x, y = x0, y0
    last_dx, last_dy = 0, 0
    
    global gradient, grad_x, grad_y
    if gradient is None:
        gradient = cv2.Laplacian(ref_image,cv2.CV_64F)
        ksize = min(R+1 if R%2 == 0 else R, 31)
        grad_x, grad_y = cv2.Sobel(ref_image,cv2.CV_64F,1,0,ksize=ksize), cv2.Sobel(ref_image,cv2.CV_64F,0,1,ksize=ksize)
    
    # default max stroke length is 1/3rd of canvas width
    max_stroke_length = max_stroke_length if max_stroke_length is not None else int(ref_image.shape[0] * 0.1)
    min_stroke_length = int(ref_image.shape[0] * 0.02)
    
    width, height, _ = ref_image.shape
    
    for i in range(1, max_stroke_length):
        x = max(min(x, ref_image.shape[0]-1), 0)
        y = max(min(y, ref_image.shape[1]-1), 0)
        
        if (i > min_stroke_length) and \
                (np.sum(np.abs(ref_image[x,y,:] - canvas[x,y,:]*255.)) < np.sum(np.abs(ref_image[x,y,:] - stroke_color))):
            break
        
        # detect vanishing gradient
        grad = np.sum(gradient[x,y])
        if grad < 1e-4 and grad > -1e-4:
            break
        
        # get unit vector of gradient
        gx, gy = np.sum(grad_x[x,y]),  np.sum(grad_y[x,y])

        # compute a normal direction
        dx, dy = -1.*gy, gx

        # if necessary, reverse direction
        if (last_dx * dx + last_dy * dy) < 0:
            dx, dy = -dx, -dy

        # filter the stroke direction
        dx, dy = fc*dx + (1-fc)*last_dx, fc*dy + (1-fc)*last_dy
        
        if (dx**2 + dy**2) != 0:
            dx, dy = dx / (dx**2 + dy**2)**(.5), dy / (dx**2 + dy**2)**(.5)
        else:
            break
        x, y = int(x + R*dx), int(y + R*dy)
        last_dx, last_dy = dx, dy
        
        K.append((x,y))

    return draw_spline_stroke(K, R, width=width, height=height)

def apply_stroke(canvas, stroke, color):
    """
    Apply a given stroke to the canvas with a given color
    
    args:
        canvas (np.array[width, height, 3]) : Current painting canvas 0-1 RGB
        stroke (np.array[width, height]) :  Stroke boolean map
        color (np.array[3]) : RGB color to use for the brush stroke
    
    return:
        np.array[width, height, 3] : Painting with additional stroke in 0-1 RGB format
    """
    s_expanded = np.tile(stroke[:,:, np.newaxis], (1,1,3))
    s_color = s_expanded * color[None, None, :]

    return canvas * (1 - s_expanded) + s_color

def paint_layer(canvas, reference_image, r, f_g, T, curved):
    """
    Go through the pixels and paint a layer of strokes with a given radius
    
    args:
        canvas (np.array[width, height, 3]) : Current painting canvas 0-1 RGB
        reference_image (np.array[width, height, 3]) :  Reference image 0-255 RGB
        r (int) : Brush radius to use
        f_g (float) : Grid size - controls spacing of brush strokes
        T (int) : Approximation threshold - how close the painting should be to target
                  In terms of pixel values.
        curved (bool) : Whether to use curved or straight brush strokes
    
    return:
        np.array[width, height, 3] : Painting in 0-1 RGB format
    """
    S = []
    
    # create a pointwise difference image
    D = np.sum(np.abs(canvas*255. - reference_image), axis=2)
    grid = int(f_g * r)
    
    width, height, _ = canvas.shape
    
    for x in range(0, width, grid):
        for y in range(0, height, grid):
            # sum the error near (x,y)
            D = np.sum(np.abs(canvas*255. - reference_image), axis=2)
            region = D[max(x-grid//2, 0):x+grid//2, max(y-grid//2, 0):y+grid//2]
            areaError = np.sum(region) / (region.shape[0] * region.shape[1])
            
            if areaError > T:
                if curved:
                    s = 1 - make_spline_stroke(x, y, r, reference_image, canvas)
                else:
                    noise = np.random.rand(region.shape[0], region.shape[1])*0.0001
                    x1, y1 = np.unravel_index((region + noise).argmax(), region.shape)
                    x1 += max(x - grid//2, 0)
                    y1 += max(y - grid//2, 0)
                    s = 1 - make_stroke(r/width*2, x/width, x1/width, y/height, y1/height, width, height)
                color = reference_image[x,y,:] / 255.
                canvas = apply_stroke(canvas, s, color)
        
    # plt.imshow(canvas)
    # plt.show()
    return canvas

def paint(source_image, R, T=100, curved=True, f_g=1):
    """
    Paint a given image
    
    args:
        source_image (np.array[width, height, 3]) : Target image 0-255 RGB
        R (list(int)) : List of brush radii to use
    kwargs:
        T (int) : Approximation threshold - how close the painting should be to target
                  Default 100. In terms of pixel values.
        curved (bool) : Whether to use curved or straight brush strokes
        f_g (float) : Grid size - controls spacing of brush strokes
    
    return:
        np.array[width, height, 3] : Painting in 0-1 RGB format
    """
    global gradient, grad_x, grad_y
    canvas = np.ones(source_image.shape)
    
    # paint the canvas
    for r in sorted(R, reverse=True): # largest to smallest
        # apply Gaussian blur
        reference_image = cv2.GaussianBlur(source_image, (r,r) if r%2 == 1 else (r+1, r+1), 0)
        # reset gradiant cache
        gradient, grad_x, grad_y = None, None, None
        # paint a layer
        canvas = paint_layer(canvas, reference_image, r, T=T, curved=curved, f_g=f_g)
        
    return canvas

def resize_img(img, max_size=300):
    h, w, _ = img.shape
    if w > max_size and w > h:
        img = cv2.resize(img, (max_size, int((max_size/w) * h)))
    elif h > max_size and h >= w:
        img = cv2.resize(img, (int((max_size/h) * w), max_size))
    return img, w, h

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Paint an image')

    parser.add_argument('img', type=str, help='path to an image to paint')

    parser.add_argument('--r', nargs='+', type=int, default=[8,4,2], help='radii to use for brushes. Usage --r 8 4 2')
    parser.add_argument('--output', type=str, default='./output.jpg', help='output file name and path')
    parser.add_argument('--T', type=float, default=100., help='Approximation threshold - how close the painting should be to target')
    parser.add_argument('--straight', action='store_true', default=False, help='Use straight brush strokes. Default False=curved strokes.')
    parser.add_argument('--f_g', type=float, default=1., help='Grid size - controls spacing of brush strokes')

    args = parser.parse_args()

    img = cv2.imread(args.img, cv2.IMREAD_COLOR)[:,:,::-1]
    img, original_width, original_height = resize_img(img)

    painting = paint(img, args.r, T=args.T, curved=not args.curved, f_g=args.f_g) * 255.

    painting = cv2.resize(painting, (original_width, original_height))
    cv2.imwrite(args.output, painting[:,:,::-1])