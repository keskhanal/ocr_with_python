import cv2 as cv
import numpy as np

def order_points(pts):
    """
    Rearrange coordinates to order: top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)

    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]
 
    diff = np.diff(pts, axis=1)
    
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    
    # return the ordered coordinates
    return rect.astype('int').tolist()


def find_dest(pts):
    (tl, tr, br, bl) = pts
    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Final destination co-ordinates.
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
 
    return order_points(destination_corners)


def scan(img):
    # Resize image to workable size
    dim_limit = 1080
    max_dim = max(img.shape)
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        img = cv.resize(img, None, fx=resize_scale, fy=resize_scale)

    # Create a copy of resized original image for later use
    orig_img = img.copy()
    
    # Repeated Closing operation to remove text from the document.
    kernel = np.ones((5, 5), np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=3)
    
    # GrabCut
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (20, 20, img.shape[1] - 20, img.shape[0] - 20)
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
 
    #increase contrast/brightness
    alpha = 1.5 #Contrast control (1.0-3.0)
    beta = 0    #Brightness control (0-100)
    img_adjusted = cv.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    gray = cv.cvtColor(img_adjusted, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (11, 11), 0)
    
    # Edge Detection.
    canny = cv.Canny(gray, 0, 200)
    canny = cv.dilate(canny, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
 
    # Finding contours for the detected edges.
    contours, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    
    # Keeping only the largest detected contour.
    page = sorted(contours, key=cv.contourArea, reverse=True)[:5]
 
    # Detecting Edges through Contour approximation.
    # Loop over the contours.
    if len(page) == 0:
        return orig_img
 
    for c in page:
        # Approximate the contour.
        epsilon = 0.02 * cv.arcLength(c, True)
        corners = cv.approxPolyDP(c, epsilon, True)
        # If our approximated contour has four points.
        if len(corners) == 4:
            break
    
    # Sorting the corners and converting them to desired shape.
    corners = sorted(np.concatenate(corners).tolist())
    
    # For 4 corner points being detected.
    corners = order_points(corners)
 
    destination_corners = find_dest(corners)
 
    h, w = orig_img.shape[:2]
    # Getting the homography.
    M = cv.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
  
    # Perspective transform using homography.
    final = cv.warpPerspective(orig_img, M, (destination_corners[2][0], destination_corners[2][1]),
                                flags=cv.INTER_LINEAR)

    return final



def main(filepath:str):
    img_original = cv.imread(filepath)
    # resized_org = cv.resize(img_original, (500,500))
    cv.imshow("original image", img_original)

    img_rotated = scan(img_original)
    # resized_rot = cv.resize(img_rotated, (500,500))
    cv.imshow("rotated image", img_rotated)

    cv.waitKey(0)


if __name__=="__main__":
    filepath = input("enter imagepath: ")
    main(filepath)