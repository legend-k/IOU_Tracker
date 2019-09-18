import glob
import operator

import numpy as np
import tqdm
import os
import sys

import Tracklet
import time
import math
import pdb
import pandas as pd
import cv2
from angle_compute import AngleMath


# from DetectionTracker.cv_tracker import Tracker_CV as CVT


class Tracker:
    def __init__(self, log=True):

        # internal vairiable that keeps a track of all the tracklets formed
        self.tracklet_list = []
        # iterator for id'ing the tracklets
        self.id_iterator = _IdIterator()
        self.frame_iterator = None
        # a log flag
        self._log = log

        self.score_threshold = 0.5
        self.frame_score = ""

        # tracklet of interest
        self.selected_tracklet: Tracklet.Tracklet = None
        self.gamma_threshold = 5
        self.angleMath = AngleMath()
        self.file_count = 0
        self.angles_file_path = ''

        # variables related to the parallel cv tracker
        # self._cv_tracker = CVT('DLIB')
        # self._cv_tracker_initialised = False # a flag representing the current nature of the tracker

    def start_tracking(self, current_detected_list, timestamp, selected_bbox, atgm_state, frame):
        '''
        Tracking initialization
        :param current_detected_list:
        :param timestamp:
        :param selected_bbox:
        :param atgm_state: (dict) {'zoom':<int>, 'azimuth':<float>, 'elevation':<float>}
        :return:
        '''
        # re-initializing the tracker in case it is being started again
        if self.tracklet_list != None:
            self.tracklet_list = None
            self.id_iterator = _IdIterator()
        if len(current_detected_list) > 0:
            self.tracklet_list = []
            for current_detection in current_detected_list:
                # Converting steps to angles
                # atgm_state = atgm_state.copy()
                # atgm_state['azimuth'] /= 10421
                # atgm_state['elevation']
                angular_position = self._bbox_to_angles(current_detection, atgm_state)
                self.tracklet_list += [Tracklet.Tracklet
                                       (current_detection, timestamp, next(self.id_iterator), angular_position)]

            self.selected_tracklet = self.tracklet_list[current_detected_list.index(selected_bbox)]

        else:
            raise ValueError("Detection list is empty")

        selected_angular_positions_list = list(self.selected_tracklet.angular_positions_queue.queue)
        selected_bbox_list = list(self.selected_tracklet.bbox_queue.queue)
        selected_timestamps_list = list(self.selected_tracklet.timestamp_queue.queue)

        # cv tracker related initialisations
        # self._cv_tracker.initialise(frame, selected_bbox)
        # self._cv_tracker_initialised = True

        # logging all tracker related variables

        return selected_bbox_list[-1]

    def update_frame(self, current_detected_list, timestamp, atgm_state, frame):
        # Converting steps to angles
        # atgm_state = atgm_state.copy()
        # atgm_state['azimuth'] /= 10421
        # atgm_state['elevation'] /= 22390
        # Selected tracklet is set to none if tracking is lost

        selected_angular_positions_list = list(self.selected_tracklet.angular_positions_queue.queue)
        selected_bbox_list = list(self.selected_tracklet.bbox_queue.queue)
        selected_timestamps_list = list(self.selected_tracklet.timestamp_queue.queue)

        # cv tracker update
        # if self._cv_tracker_initialised:
        #     bbox = self._cv_tracker.update(frame)
        #     print(bbox)
        #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,255,255), 2)

        self._update_tracklets_with_detections(timestamp, current_detected_list, None, atgm_state)
        if self.selected_tracklet is None:
            # raise Exception("Initialize tracker before updating frame")
            return None

        # if len(self.selected_tracklet.bbox_queue == None
        return selected_bbox_list[-1]

    def draw_bboxes(self, frame):
        # for bbox in bbox_list:
        #     cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
        #                   (255,0,0), 3)
        if self.selected_tracklet != None:
            # TODO: decide if atgm_state is necssary in draw_tracking_box
            self.selected_tracklet.draw_tracking_bbox(frame)
            return frame
        else:
            return frame

    def _create_frame_iterator(self, path):
        if os.path.isdir(path):  # is a folder (containing images)
            images_paths = glob.glob(path)

            if len(images_paths) == 0:  # the directory is empty

                raise Exception("Directory provided is empty")

            self.frame_iterator = iter(_TimedImageIterator(path))

        else:  # is a video file
            self.frame_iterator = iter(_TimedVideoIterator(path))

    def _update_single_tracklet(self, tracklet, current_detected_list, confidence_list, timestamp):

        if len(current_detected_list) > 0:
            scores_list = [tracklet.calc_score(bbox) for bbox in current_detected_list]
            best_idx, best_score = (max(enumerate(scores_list), key=operator.itemgetter(1)))

            if best_score > self.score_threshold:  # if a score is good enough

                current_bbox = current_detected_list[best_idx]
                current_detected_list[best_idx] = None

                if (confidence_list != None):
                    current_confidence = confidence_list[best_idx]
                    confidence_list[best_idx] = None

                tracklet.update_tracklet(current_bbox, timestamp)
            else:  # if no score is good enough
                tracklet.calc_ghost_box_and_update(timestamp)  # do ghost update

        else:  # if no detections are left unassigned in the current_detected_list
            tracklet.calc_ghost_box_and_update(timestamp)  # do ghost update

    def _update_tracklets_with_detections(self, timestamp, current_detected_list, confidence_list, atgm_state,
                                          debug=False):
        """
        Using the provided detected list, the existing tracklets are updated. Any new tracklets that need to be
        made are also made.
        :param timestamp: (float) epoch time
        :param current_detected_list: (list) list of bboxes
        :param confidence_list: (list) corresponding list of confidences
        :param atgm_state: (dict) a dictionary containing state of atgm (zoom, azi, elev)
        :param debug: (bool) make true while debugging (internal use only)
        :return: (None)
        """

        # Split the tracklet list into two groups based on high and low gamma value
        low_gamma_tracklet_list = []
        high_gamma_tracklet_list = []
        for tracklet in self.tracklet_list:
            if tracklet.gamma >= 60:
                if self.selected_tracklet == tracklet:
                    self.selected_tracklet = None
                self.tracklet_list.remove(tracklet)
            if tracklet.gamma <= self.gamma_threshold:
                low_gamma_tracklet_list.append(tracklet)
            else:
                high_gamma_tracklet_list.append(tracklet)

        self._make_associations(timestamp, low_gamma_tracklet_list, current_detected_list, confidence_list,
                                atgm_state.copy())
        self._make_associations(timestamp, high_gamma_tracklet_list, current_detected_list, confidence_list,
                                atgm_state.copy())

        # all unassigned detection boxes are used to make new tracklets
        # TODO add a confidence threshold to make a new tracklet
        for current_detection in current_detected_list:
            # TODO Look at the Nuns
            if current_detection != None:
                angular_position = self._bbox_to_angles(current_detection, atgm_state)
                self.tracklet_list.append(
                    Tracklet.Tracklet(current_detection, timestamp, next(self.id_iterator),
                                      angular_position))

    def _make_associations(self, timestamp, tracklet_list, current_detected_list, confidence_list, atgm_state):

        # a matrix containing scores of all tracklets against all detections
        scores_mat = self._create_score_matrix(tracklet_list, current_detected_list)

        while ((scores_mat.shape[0] != 0 and scores_mat.shape[1] != 0) and (np.max(scores_mat) > self.score_threshold)):
            # get index of tracker and index of detection box which have the highest score
            t_idx, d_idx = np.unravel_index(np.argmax(scores_mat), scores_mat.shape)

            tracklet_to_update = tracklet_list[t_idx]
            associated_detection_box = current_detected_list[d_idx]

            # update tracklet with associated detection box
            angular_position = self._bbox_to_angles(associated_detection_box, atgm_state)
            # Writing azimuth and elevation angles to file for  debugging

            tracklet_to_update.update_tracklet(associated_detection_box, timestamp, angular_position)

            # delete the row and column from scores_mat
            scores_mat = np.delete(scores_mat, t_idx, 0)
            scores_mat = np.delete(scores_mat, d_idx, 1)

            # remove the tracklet and detection box from the corresponding lists
            tracklet_list.remove(tracklet_list[t_idx])
            current_detected_list.remove(current_detected_list[d_idx])
        else:
            # do ghost update for left over tracklets

            for tracklet in tracklet_list:
                tracklet.calc_ghost_box_and_update(timestamp, atgm_state)

    def _create_score_matrix(self, tracklet_list, current_detected_list):

        score_matrix = np.array([[tracklet.calc_score(current_detection)
                                  for current_detection in current_detected_list] for tracklet in tracklet_list])

        return score_matrix

    def _draw_detection_bboxes(self, frame, bbox_list, scores=None, confidences_list=None, colour=(0, 0, 255),
                               box_thickness=1):
        """

        :param frame: one frame from video read using cv2
        :param bbox_list: a list of bboxes (each of which is a list of 4 elements)
        :param confidences_list: (list) list of confidences of the detections
        :param colour: (tuple) specify colour of bounding box in (B, G, R) format
        :param box_thickness: (integer) number of pixels for thickness of bbox
        :return: None (modifies frame)

        Receives a frame along with auxiliary info and draws a bbox around all the tanks detected.
        The frame is modified.

        """

        for i in range(len(bbox_list)):
            text = ''
            bbox = bbox_list[i]

            assert len(bbox) == 4

            # display id for detection box
            dbox_id = i + 1
            text += "D{} ".format(dbox_id)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          colour, box_thickness)

            if (scores != None):
                text += "score: " + str(round(scores[i], 2)) + " "

            # show confidence for detection if confidence list is present
            if confidences_list != None:
                confidence = confidences_list[i]
                text += "Confidence: {0}".format(str(round(confidence, 2)))

            cv2.putText(frame, text, (bbox[0], bbox[1] - 5)
                        , cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    def _draw_all_scores(self, scores_list_of_all_tracklets, frame):
        if len(scores_list_of_all_tracklets) > 0:
            data = [item[1] for item in scores_list_of_all_tracklets]
            index = ["T{}".format(item[0]) for item in scores_list_of_all_tracklets]
            columns = ["D{}".format(i) for i in range(1, len(scores_list_of_all_tracklets[0][1]) + 1)]

            df = pd.DataFrame(data=data, index=index, columns=columns)
            scores_txt = str(df.round(2))
        else:
            scores_txt = 'No detections made in this frame'

        y0, dy = 50, 20
        for i, line in enumerate(scores_txt.split('\n')):
            y = y0 + i * dy
            cv2.putText(frame, line, (100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        [255, 0, 0], 1, cv2.LINE_AA)

    def _bbox_to_angles(self, bbox, atgm_state):
        """
        Given a bbox (xmin, ymin, xmax, ymax) and the atgm_state, the angular position (theta, phi) are
        calculated and returned
        :param bbox: (tuple) (xmin, ymin, xmax, ymax)
        :param atgm_state: (dict) contains atgm state (azimuth, elevation, zoom)
        :return: (tuple) (azimuth angle, elevation angle)
        """
        atgm_zoom = atgm_state['zoom']
        atgm_azimuth = atgm_state['azimuth']
        atgm_elevation = atgm_state['elevation']

        x_center = ((bbox[0] + bbox[2]) / 2) - 360
        y_center = ((bbox[1] + bbox[3]) / 2) - 240

        azimuth_angle = atgm_azimuth + self.angleMath.get_azimuth_angle(atgm_zoom, x_center)
        elevation_angle = (atgm_elevation + self.angleMath.get_elevation_angle(atgm_zoom, y_center))

        return (azimuth_angle, elevation_angle)


class _TimedVideoIterator:
    """
    Class which creates an iterator for a video file
    The iterator returns video frames with timestamps
    """

    def __init__(self, vid_path):
        self.cap = cv2.VideoCapture(vid_path)

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if ret:
            return frame, time.time()
        else:
            raise StopIteration

    def __len__(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame_shape(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height


class _TimedImageIterator:
    """
    Class which creates an iterator for a folder containing images
    The iterator returns images with timestamps
    """

    def __init__(self, image_folder_path):
        # the iterator works only for images genereated by tank application
        sorted_image_paths = sorted(glob.glob(os.path.join(image_folder_path, "*.*")),
                                    key=lambda path: int(os.path.split(path)[1][7:-4]))
        self.image_paths = sorted_image_paths
        self.image_path_iterator = iter(self.image_paths)

    def __iter__(self):
        return self

    def __next__(self):
        image_path = next(self.image_path_iterator)
        image = cv2.imread(image_path)
        return image, time.time()

    def __len__(self):
        return len(self.image_paths)

    def get_frame_shape(self):
        frame_size = cv2.imread(self.image_paths[0]).shape[:-1][::-1]
        return frame_size


class _IdIterator:
    """
    Class which creates an iterator which runs forever and generates ids
    Ids are natural numbers
    """

    def __init__(self):
        self.n = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.n += 1
        return self.n
