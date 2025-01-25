import cv2
import numpy as np


def mask_out_surroundings(image):
    # mask out non-workspace
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    # Create mask for contours
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Draw contours on mask
    contour_image = image.copy()
    cv2.drawContours(mask, contours, -1, 255, -1)

    # Apply contour mask to original image
    masked_image = cv2.bitwise_and(contour_image, contour_image, _,
                                     mask)

    return masked_image

def detect_and_draw_circles(gray_masked_img, circles_image, color):
    # Detect circles
    circles = cv2.HoughCircles(
            gray_masked_img, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
            param1=50, param2=30, minRadius=10, maxRadius=100
            )

    # Draw detected circles
    if circles is not None:
        print("circles not none")
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            
            # Draw the outer circle
            cv2.circle(circles_image, center, radius, (0, 255, 0), 2)
            # Draw the center point
            cv2.circle(circles_image, center, 3, (0, 0, 255), -1)
            # Add radius information
            cv2.putText(circles_image, f'{color}, R: {radius}', 
                       (center[0]-20, center[1]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return circles_image

def detect_colored_spheres(img_path):
    image = cv2.imread(img_path)
    circles_image = image.copy()
    
    masked_image = mask_out_surroundings(image)
    hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
    #gray_masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    color_masks = {}

    # Red wraps around the hue spectrum, so we need two ranges
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    color_masks['red'] = red_mask

    # other colors:
    color_ranges = {'blue' : ([95, 10, 0], [130, 255, 255]),
                    'orange' : ([10, 100, 100], [25, 255, 255]),
                    'yellow' : ([25, 100, 100], [50, 255, 255])}
    
    for color in color_ranges:
        color_mask = cv2.inRange(hsv, np.array(color_ranges[color][0]),
                                    np.array(color_ranges[color][1]))
        color_masks[color] = color_mask

    color_circles_image = image.copy()
    for color in color_masks:
        # Create masks
        color_result = cv2.bitwise_and(hsv, hsv, mask=color_masks[color])
        color_gray_masked_image = cv2.cvtColor(color_result, cv2.COLOR_BGR2GRAY)
        _, color_gray_masked_image = cv2.threshold(color_gray_masked_image, 45, 255, cv2.THRESH_BINARY)
            
        color_blurred_image = cv2.GaussianBlur(color_gray_masked_image, (5, 5), 0)

        if color == 'blue':
            kernel = np.ones((6, 6), np.uint8)
            color_blurred_image = cv2.morphologyEx(color_blurred_image, cv2.MORPH_OPEN, kernel)

        detect_and_draw_circles(color_blurred_image, color_circles_image, color)

        #cv2.imshow(color + ' blurred image', color_blurred_image)
        cv2.imshow(color + ' color_circles_image', color_circles_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.destroyAllWindows()
    
    return

def main():
    img_path = 'image_prelab8.jpg'
    detect_colored_spheres(img_path)

if __name__ == '__main__':
    main()
