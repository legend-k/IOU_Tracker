# TODO: pipeline detection output into Tracker
# TODO: Implement User defined rectangle for the tracker
import cv2
import GenericEvent as ev
from threading import Thread, Event
import threading
# import dlib
import TinyYoloDetection as TY
# from Tracker_new import *
import Tracker as DT
from enum import Enum
# import ipdb
import time


class ImgProMode(Enum):
    VIDEO = 1
    DETECTION = 2
    TRACKING = 3
    TRACKING_WITH_DETECTION = 4


# class that handles polling of frames and image processing
class ImageProcessor(object):

    # initialization function
    def __init__(self, name=None, mode=ImgProMode.VIDEO, tracking_with_detection=False, debug=False):
        self.name = name
        self._video = None
        self._debug = debug
        self.weight_path = "22-Aug-2018-yolo-tank-app-22.h5"
        # frame event
        self._frameEventSig = {'frame': None}
        self.frameEvent = ev.GenericEvent(**self._frameEventSig)
        # tracked frame event
        self._trackEventSig = {'rectangle': None, 'final_image': None, 'record_image': None}
        self.TrackEvent = ev.GenericEvent(**self._trackEventSig)
        # detection frame event
        self._detectEventSig = {'confidence_list': None, 'rectangles': None, 'final_image': None}
        self.DetectionEvent = ev.GenericEvent(**self._detectEventSig)
        self._stop_turretSig = {'stop': None}
        self.StopTurretEvent = ev.GenericEvent(**self._stop_turretSig)
        # thread that handles video capture
        self._videoThread = None
        # declaring a detection thread
        self._detectThread = None
        # declaring a tracking thread
        self._trackThread = None
        # threading event for video polling
        self._frameloop_stopevent = None
        # threading event for detection
        self._detectloop_stopevent = None
        # threading event for tracking
        self._trackloop_stopevent = None
        # mode of the imageprocessor
        self.mode = mode
        self.AZIMUTH_STEP_FACTOR = 10421
        self.ELEVATION_STEP_FACTOR = 22390
        self.dummy = False

        self._video = cv2.VideoCapture(self.name)
        self._video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self._video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.lag_start_time = None
        # Tracker
        # self.tracker = DT.Tracker()
        if not tracking_with_detection:
            self.tracker = RE3Tracker()
            self.tracker = RE3Tracker()
        else:
            self.tracker = DT.Tracker()

        # Detector
        self.detector = None
        self._config_path = r"E:\Deep_Learning\aeromodel_yolo_files\plane.cfg"
        self._weight_path = r"E:\Deep_Learning\aeromodel_yolo_files\plane_5000.weights"
        self._meta_path = r"E:\Deep_Learning\aeromodel_yolo_files\plane.data"
        # pdb.set_trace()
        self.detector = self._initialize_detector()

        # tracking with detection initializations
        self.td_initializer_data = None
        # few flags
        # flag to maintain the state of tracking: has it started or not
        self.has_tracking_started = False
        # flag to maintain the state of detection: has it starte or not
        self.has_detection_started = False
        # flag to maintain the state of tracking with detection
        self.has_tracking_with_detection_started = False
        # flag to denote whether or not to draw boxes
        self.track_draw_box = None

        # a few thread locks
        self._mode_lock = threading.Lock()

        # a few state variables of atgm
        self._atgm_azimuth = None
        self._atgm_elevation = None
        self._atgm_zoom = None
        self._atgm_azi_lock = threading.Lock()
        self._atgm_ele_lock = threading.Lock()
        self._atgm_zoom_lock = threading.Lock()

        # Variable to check if tracker is in ghost mode or not
        self.ghost_mode_flag = False

    # following properties maintain state variables for the atgm, mainly to be relayed to the IoU tracker
    @property
    def Azimuth_state(self):
        with self._atgm_azi_lock:
            result = self._atgm_azimuth
        return result

    @Azimuth_state.setter
    def Azimuth_state(self, value):
        with self._atgm_azi_lock:
            self._atgm_azimuth = value

    @property
    def Elevation_state(self):
        with self._atgm_ele_lock:
            result = self._atgm_elevation
        return result

    @Elevation_state.setter
    def Elevation_state(self, value):
        with self._atgm_ele_lock:
            self._atgm_elevation = value

    @property
    def Zoom(self):
        with self._atgm_zoom_lock:
            result = self._atgm_zoom
        return result

    @Zoom.setter
    def Zoom(self, value):
        with self._atgm_zoom_lock:
            self._atgm_zoom = value

    # stops/ pauses the video thread
    def stop(self, pause=False):

        # stops the frame thread permanently
        if not pause:
            # send an event to the polling loop to stop polling
            if self.mode == ImgProMode.VIDEO:
                if self._frameloop_stopevent is not None:
                    self._frameloop_stopevent.set()
                self._videoThread = None
            elif self.mode == ImgProMode.DETECTION:
                if self._detectloop_stopevent is not None:
                    self._detectloop_stopevent.set()
                self._detectThread = None
            elif self.mode == ImgProMode.TRACKING_WITH_DETECTION:
                if self._detectloop_stopevent is not None:
                    self._detectloop_stopevent.set()
                self._detectThread = None
            elif self.mode == ImgProMode.TRACKING:
                if self._trackloop_stopevent is not None:
                    self._trackloop_stopevent.set()
                self._trackThread = None
            if self._video is not None:
                self._video.release()
                print("video released")
            self._video = None

        elif pause:
            # send an event to the polling loop to stop polling
            if self.mode == ImgProMode.VIDEO:
                self._frameloop_stopevent.set()
            elif self.mode == ImgProMode.DETECTION:
                self._detectloop_stopevent.set()
            elif self.mode == ImgProMode.TRACKING:
                self._trackloop_stopevent.set()

        # time.sleep(5)
        return None

    # video thread handler
    def _frame_thread_handler(self):
        # main polling loop
        start = time.time()
        while not self._frameloop_stopevent._flag:
            # frame retrieval and checking
            start = time.time()
            if self._video is not None:
                # time.sleep(0.030)
                ok, frame = self._video.read()
                if not ok:
                    break

            if self._debug:
                # fps calculation
                end = time.time()
                if (end - start) != 0:
                    fps = 1 / (end - start)
                else:
                    fps = 0
                # pstring = 'FPS: ' + str(fps)
                # cv2.putText(frame, pstring, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            if self.frameEvent.isSubscribed:
                self.frameEvent(frame=frame)

        return None

    # Detection thread handler
    def _detect_thread_handler(self):
        """Thread handler for detection"""
        mode = None
        while not self._detectloop_stopevent._flag:
            start = time.time()
            if self._video is not None:
                self.lag_start_time = time.time()
                ok, frame = self._video.read()

                if not ok:
                    print("frame retrieval not successful")
                    break

            # frame retrieval and checking
            if self.has_detection_started:
                # detection step
                confidence_list, boxes, image = self.detector.detect(frame.copy(), draw=True)

                # dummy detections
                if not self.dummy:
                    boxes.append(time.time())
                else:
                    boxes = [[700, 450, 710, 460]]
                    boxes.append(time.time())

                # print(boxes)

                # accessing mode
                with self._mode_lock:
                    mode = self.mode

                # handling tracking with detection
                if mode == ImgProMode.TRACKING_WITH_DETECTION:
                    if not self.has_tracking_with_detection_started:
                        box = []
                        box.append(self.tracker.start_tracking(*self.td_initializer_data, frame))
                        # all drawing operations
                        if self._debug:
                            track_end = time.time()
                            if (track_end - start) != 0:
                                fps = 1 / (track_end - start)
                            else:
                                fps = 0
                            # print("fps: ", fps)
                            # pstring = 'FPS: ' + str(fps)
                            debug_frame = frame.copy()
                            # cv2.putText(debug_frame, pstring, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
                            #             cv2.LINE_AA)
                            debug_frame = self.tracker.draw_bboxes(debug_frame)
                        if self.TrackEvent.isSubscribed:
                            self.TrackEvent(rectangle=box[0], final_image=debug_frame, record_image=frame)
                        self.has_tracking_with_detection_started = True
                    else:
                        box = []

                        # following atgmState variable is a dictionary that contains current state of the atgm, to be sent to the tracker
                        atgmState = {}

                        if self.Azimuth_state is None:
                            self.Azimuth_state = 0
                        if self.Elevation_state is None:
                            self.Elevation_state = 0
                        if self.Zoom is None:
                            self.Zoom = 1

                        atgmState['zoom'] = self.Zoom
                        atgmState['azimuth'] = self.Azimuth_state / self.AZIMUTH_STEP_FACTOR
                        atgmState['elevation'] = -1 * self.Elevation_state / self.ELEVATION_STEP_FACTOR

                        box_temp = self.tracker.update_frame(boxes[:-1], boxes[-1], atgmState, frame)

                        # update ghost mode flag

                        if box_temp != None:
                            box.append(box_temp)
                            # all drawing operations
                            if self._debug:
                                track_end = time.time()
                                if (track_end - start) != 0:
                                    fps = 1 / (track_end - start)
                                else:
                                    fps = 0
                                # pstring = 'FPS: ' + str(fps)
                                debug_frame = frame.copy()
                                # cv2.putText(debug_frame, pstring, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                #             (0, 255, 255), 2, cv2.LINE_AA)
                                debug_frame = self.tracker.draw_bboxes(debug_frame)
                            if self.TrackEvent.isSubscribed:
                                self.TrackEvent(rectangle=box[0], final_image=debug_frame, record_image=frame)
                        else:
                            # switch back to detection if tracking fails
                            self.mode = ImgProMode.DETECTION
                            self.has_tracking_with_detection_started = False
                            self.StopTurretEvent(stop=True)
                            if self._debug:
                                pstring = 'Reinitialize tracker...'
                                cv2.putText(frame, pstring, (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
                                            cv2.LINE_AA)
                                frame = self.tracker.draw_bboxes(frame)
                            else:
                                pstring = 'Reinitialize tracker...'
                                cv2.putText(frame, pstring, (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
                                            cv2.LINE_AA)

                else:
                    if self._debug:
                        # fps calculation
                        end = time.time()
                        if (end - start) != 0:
                            fps = 1 / (end - start)
                        else:
                            fps = 0
                        pstring = 'FPS: ' + str(fps)
                        # print(fps)
                        cv2.putText(frame, pstring, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
                                    cv2.LINE_AA)
                    if self.DetectionEvent.isSubscribed:
                        self.DetectionEvent(confidence_list=confidence_list, rectangles=boxes, final_image=image)
            if not self.has_detection_started:
                confidence_list, boxes, image = self.detector.detect(frame.copy())
                # timestamping the detection
                boxes.append(time.time())
                if self.DetectionEvent.isSubscribed:
                    self.DetectionEvent(confidence_list=confidence_list, rectangles=boxes, final_image=image)
                self.has_detection_started = True

        # rekindle detection later
        self.has_detection_started = False
        self.has_tracking_with_detection_started = False

    def _track_thread_handler(self, roi, base_frame):
        """Thread handler for the tracker"""
        while not self._trackloop_stopevent._flag:
            # pdb.set_trace() # BREAKPOINT
            if self.has_tracking_started:
                # frame retrieval and checking
                ok, frame = self._video.read()
                if not ok:
                    break
                box = self.tracker.update_frame(frame)
                if self.track_draw_box:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]),
                                  (0, 0, 255), 2)
                if self.TrackEvent.isSubscribed:
                    self.TrackEvent(rectangle=box, final_image=frame)

            # tracking
            # if self.mode == ImgProMode.TRACKING:
            if not self.has_tracking_started:
                self._initialize_tracker(roi, base_frame)
                self.has_tracking_started = True

                # rekindle tracker later
        self.has_tracking_started = False

    # runs the main frame polling thread

    # runs the detection thread
    def detect_run(self, first_detect=False):
        """starts the detection thread
        params: first_detect - a boolean flag - denoting first frame to be
                detected"""
        # if first_detect:
        # checks whether the video thread is alive, and terminates it if its
        if self.mode == ImgProMode.VIDEO:
            # stops the video thread
            if self._videoThread is not None:
                self._frameloop_stopevent.set()
                self._videoThread = None

            # thread that handles detection
            self._detectThread = Thread(target=self._detect_thread_handler,
                                        name="Detect thread")

            # initialize a stop event that needs to be set to stop detection
            self._detectloop_stopevent = Event()
            # start the thread
            self.mode = ImgProMode.DETECTION
            self._detectThread.start()
        elif self.mode == ImgProMode.TRACKING:

            if self._trackThread is not None:
                self._trackloop_stopevent.set()
                self._trackThread = None
            # initialize thread again
            self._detectThread = Thread(target=self._detect_thread_handler,
                                        name="Detect thread")
            self._detectloop_stopevent = Event()
            # change mode id it is different
            self.mode = ImgProMode.DETECTION
            self._detectThread.start()
        elif self.mode == ImgProMode.TRACKING_WITH_DETECTION:
            # if self._detectThread is not None:
            #     self._detectloop_stopevent.set()
            #     self.detectThread = None
            # self._detectThread = Thread(target=self._detect_thread_handler, name='Detect thread')
            # self._detectloop_stopevent = Event()
            # self.mode = ImgProMode.DETECTION
            # self._detectThread.start()
            self.has_tracking_with_detection_started = False
            self.mode = ImgProMode.DETECTION
        return None

    def track_run(self, roi, base_frame, draw_box=True, first_track=False):
        """starts the tracking thread
        params: roi - a list of extreme pixel coordinates
                base_frame - np.array consisting of pixel values to be worked
                upon
                first_track - a boolean flag - denoting first frame to be
                Tracked
                draw_box - draws box
        return: None - this function just starts the tracking thread"""
        if self.mode == ImgProMode.DETECTION:
            # stops the detection thread
            self._detectloop_stopevent.set()
            self.track_draw_box = draw_box

            # thread that handles tracking
            self._trackThread = Thread(target=self._track_thread_handler,
                                       name="Track thread", args=(roi, base_frame))
            # initialize a stop event that needs to be set to stop tracking
            self._trackloop_stopevent = Event()
            # change the mode
            self.mode = ImgProMode.TRACKING
            # start the thread
            self._trackThread.start()
        elif self.mode == ImgProMode.VIDEO:
            # # thread that handles tracking
            # self._trackThread = Thread(target=self.track_thread_handler,
            #                            name="Track thread", args=(roi, base_frame))
            # # initialize a stop event that needs to be set to stop tracking
            # self._trackloop_stopevent = Event()
            # # change mode
            # if self.mode != ImgProMode.TRACKING:
            #     self.mode = ImgProMode.TRACKING
            # # start the thread
            # self._trackThread.start()
            return None
        elif self.mode == ImgProMode.TRACKING:
            return None
        return None

    # Mode change
    def change_mode(self, mode):
        """Changes the mode of ImageProcessor to one of the three available in
        the ImgProMode class"""
        with self._mode_lock:
            self.mode = mode
        return None

    # tracking and detection related functions
    # initialize the tracker
    def _initialize_tracker(self, rectangle, frame):

        # self.tracker = RE3Tracker()
        self.tracker.start_tracking(rectangle, frame)
        return None

        # initialize the detector

    def _initialize_detector(self):
        """intializes the yolo detector object"""
        detector = TY.YOLODetector(config_path=self._config_path, weight_path=self._weight_path,
                                   meta_path=self._meta_path, debug=True)
        return detector


# main function to test the class
def main():
    video_name = '/home/ujjawal/Desktop/demo/tank1.mp4'
    ip = ImageProcessor(video_name)

    return None


if __name__ == "__main__":
    # main()
    print(__name__)
