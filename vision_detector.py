import cv2
import ImageProcessor_new as ip

boxes = []
frame = None


def _lightmode_detection_event_handler(confidence_list=None, rectangles=None, final_image=None):
    global frame, boxes
    # print("detection event fired")
    print(rectangles)
    boxes = rectangles
    frame = final_image


def _lightmode_track_event_handler(rectangle, final_image, record_image):
    global frame
    frame = final_image


def point_in_box(point, box):
    """ returns the clicked region of interest
    params: point - a tuple consisting of the mouseevent pixel coordinate
    return: True/False - a boolean"""
    (x, y) = point
    if (x > box[0] and x < box[2] and y > box[1] and y < box[3]):
        return True
    else:
        return False


def _light_left_click(x, y):
    global boxes
    point = (x, y)
    IoU_data = []
    _point_in_box_flag = False
    tracking_roi = None
    # self.count += 1

    if len(boxes) > 1:
        roi_list = boxes[:-1]
        timestamp = boxes[-1]

        atgmState = {}
        atgmState['zoom'] = 1
        atgmState['azimuth'] = 0
        atgmState['elevation'] = 0

        for roi in roi_list:

            if point_in_box(point, roi):
                tracking_roi = roi
                _point_in_box_flag = True

        if _point_in_box_flag:
            IoU_data.append(roi_list)
            IoU_data.append(timestamp)
            IoU_data.append(tracking_roi)
            IoU_data.append(atgmState)

            _imageProcessor.td_initializer_data = IoU_data
            _imageProcessor.mode = ip.ImgProMode.TRACKING_WITH_DETECTION
        else:
            print("point not in bbox")
            return


def get_mouse_coord(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _light_left_click(x, y)
        print("Track coordinates", x, y)


if __name__ == '__main__':
    # detector_obj = Vision_Detector()

    _imageProcessor = ip.ImageProcessor(1,
                                        tracking_with_detection=True,
                                        debug=True)

    _imageProcessor.DetectionEvent += _lightmode_detection_event_handler
    _imageProcessor.TrackEvent += _lightmode_track_event_handler

    _imageProcessor.mode = ip.ImgProMode.VIDEO
    _imageProcessor.detect_run()

    cv2.namedWindow("Filtered detector", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Filtered detector", get_mouse_coord)

    while True:
        if frame is not None:
            print(frame.shape)
            cv2.imshow("Filtered detector", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                _imageProcessor._detectloop_stopevent.set()
                break