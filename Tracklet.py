import math
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression as LR
from angle_compute import AngleMath
from queue import Queue


class Tracklet:

    # TODO: pass anglular position to tracklet
    def __init__(self, initial_bbox, timestamp, tracklet_id, angular_position):
        self.queue_length = 40
        self.bbox_queue: Queue = Queue(self.queue_length)
        self.bbox_queue.put(initial_bbox)
        self.ghost_mode_queue: Queue = Queue(self.queue_length)
        self.ghost_mode_queue.put(0)
        self.timestamp_queue: Queue = Queue(self.queue_length)
        self.timestamp_queue.put(timestamp)
        self.gamma = 0
        self.tracklet_id = tracklet_id
        self.angular_positions_queue: Queue = Queue(self.queue_length)
        self.angular_positions_queue.put(angular_position)
        self.angleMath = AngleMath()
        self.velocity = None

    def fit_to_line_and_predict_next(self, X, Y, x_next):
        """
        Fits a line with data columns X and Y
        Using that line, predicts y_next for x_next
        :param X: (list/1d array) x values
        :param Y: (list/1d array) y values
        :param x_next: (float) value for which y should be predicted
        :return: (float) predicted y value
        """

        # X = np.array(X).reshape((-1, 1))
        # X = np.array(X).flatten()
        # Y = np.array(Y).flatten()

        # line = LR()
        # line.fit(X, Y)
        #
        # y_next = line.predict(np.array((x_next,)).reshape((1, 1)))[0]
        #
        # return y_next

        # assert len(X) > 1 and len(Y) > 1

        velocity = self.calc_velocity(X, Y)
        result = Y[-1] + (x_next - X[-1]) * velocity

        return result

    def calc_velocity(self, X, Y):
        if len(X) == 1 or len(Y) == 1:
            return 0
        X = np.array(X).flatten()
        Y = np.array(Y).flatten()
        avg_velocity = np.mean((Y[1:] - Y[:-1]) / (X[1:] - X[:-1] + 10 ** (-2)))
        return avg_velocity

    def get_tracklet_velocity(self):
        angular_positions_list = list(self.angular_positions_queue.queue)
        timestamps_list = list(self.timestamp_queue.queue)

        theta_list = [pair[0] for pair in angular_positions_list]
        phi_list = [pair[1] for pair in angular_positions_list]

        velocity_theta = self.calc_velocity(timestamps_list, theta_list)
        velocity_phi = self.calc_velocity(timestamps_list, phi_list)

        return (velocity_theta, velocity_phi)

    def push(self, q, item):
        if q.full():
            q.get()
            q.put(item)
        else:
            q.put(item)

    def calc_ghost_box_and_update(self, current_timestamp, atgm_state):

        angular_positions_list = list(self.angular_positions_queue.queue)
        bbox_list = list(self.bbox_queue.queue)
        timestamps_list = list(self.timestamp_queue.queue)

        latest_bbox = bbox_list[-1]
        latest_angle = angular_positions_list[-1]
        old_x_center = (latest_bbox[0] + latest_bbox[2]) / 2
        old_y_center = (latest_bbox[1] + latest_bbox[3]) / 2

        # last n theta, phi and timestamp values
        theta_list = [pair[0] for pair in angular_positions_list]
        phi_list = [pair[1] for pair in angular_positions_list]

        if self.velocity is None:
            velocity_theta = self.calc_velocity(timestamps_list, theta_list)
            velocity_phi = self.calc_velocity(timestamps_list, phi_list)

            self.velocity = (1 * velocity_theta, 1.0 * velocity_phi)

            # get the next theta and phi values by fitting the past data to a line
            next_theta = self.fit_to_line_and_predict_next(timestamps_list,
                                                           theta_list, current_timestamp)
            next_phi = self.fit_to_line_and_predict_next(timestamps_list,
                                                         phi_list, current_timestamp)

        else:  # x2 = x1 + u * dt style update
            next_theta = theta_list[-1] + (self.velocity[0] * (current_timestamp - timestamps_list[-1]))
            next_phi = phi_list[-1] + (self.velocity[1] * (current_timestamp - timestamps_list[-1]))

        # add this angle pair to angle list
        self.push(self.angular_positions_queue, (next_theta, next_phi))

        # get the bbox coordinates for this angle pair
        zoom = atgm_state['zoom']
        relative_theta = next_theta - atgm_state['azimuth']
        relative_phi = next_phi - atgm_state['elevation']
        new_x_center = self.angleMath.get_horizontal_pixel(zoom, relative_theta) + 360
        new_y_center = self.angleMath.get_vertical_pixel(zoom, relative_phi) + 240

        x_diff = new_x_center - old_x_center
        y_diff = new_y_center - old_y_center

        ghost_bbox = [0, 0, 0, 0]
        ghost_bbox[0] = int(latest_bbox[0] + x_diff)
        ghost_bbox[1] = int(latest_bbox[1] + y_diff)
        ghost_bbox[2] = int(latest_bbox[2] + x_diff)
        ghost_bbox[3] = int(latest_bbox[3] + y_diff)

        self.push(self.bbox_queue, ghost_bbox)
        self.push(self.ghost_mode_queue, 1)
        self.push(self.timestamp_queue, current_timestamp)
        self.gamma += 1

    def update_tracklet(self, current_bbox, current_timestamp, angular_position):

        self.push(self.bbox_queue, current_bbox)
        self.push(self.ghost_mode_queue, 0)
        self.push(self.timestamp_queue, current_timestamp)
        self.push(self.angular_positions_queue, angular_position)

        self.gamma = 0
        self.velocity = None

    def calc_score(self, bbox):
        if bbox == None:
            return 0

        lambda_iou = 1
        lambda_dist = 0.5  ###########Changed from 0.2

        iou_score = lambda_iou * self.calc_iou_score(bbox)
        dist_score = lambda_dist * self.calc_dist_score(bbox)

        score = (1 + self.gamma / 10) * (iou_score + dist_score)

        return score

    def calc_iou_score(self, bbox):
        bbox_list = list(self.bbox_queue.queue)
        tracking_bbox = bbox_list[-1]
        detection_bbox = bbox
        xA = max(tracking_bbox[0], detection_bbox[0])
        yA = max(tracking_bbox[1], detection_bbox[1])
        xB = min(tracking_bbox[2], detection_bbox[2])
        yB = min(tracking_bbox[3], detection_bbox[3])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth rectangles
        tracking_bbox_area = abs((tracking_bbox[2] - tracking_bbox[0]) * (tracking_bbox[3] - tracking_bbox[1]))
        detection_bbox_area = abs((detection_bbox[2] - detection_bbox[0]) * (detection_bbox[3] - detection_bbox[1]))
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(tracking_bbox_area + detection_bbox_area - interArea)

        # return the intersection over union value

        return iou

    def calc_dist_score(self, bbox):
        bbox_list = list(self.bbox_queue.queue)
        tracking_bbox = bbox_list[-1]
        detection_bbox = bbox

        # compute the area of both the prediction and ground-truth rectangles
        tracking_bbox_area = abs((tracking_bbox[2] - tracking_bbox[0]) * (tracking_bbox[3] - tracking_bbox[1]))
        detection_bbox_area = abs((detection_bbox[2] - detection_bbox[0]) * (detection_bbox[3] - detection_bbox[1]))
        average_area = (tracking_bbox_area + detection_bbox_area) / 2

        detection_bbox_center_x = (detection_bbox[0] + detection_bbox[2]) / 2
        detection_bbox_center_y = (detection_bbox[1] + detection_bbox[3]) / 2
        tracking_bbox_center_x = (tracking_bbox[0] + tracking_bbox[2]) / 2
        tracking_bbox_center_y = (tracking_bbox[1] + tracking_bbox[3]) / 2

        x_dist = tracking_bbox_center_x - detection_bbox_center_x
        y_dist = tracking_bbox_center_y - detection_bbox_center_y

        dist_score = 1 / (math.sqrt((math.pow(x_dist, 2) + math.pow(y_dist, 2)) / average_area) + 1)
        return dist_score

    # TODO: take atgm_state
    def draw_tracking_bbox(self, frame, debug=False, box_thickness=1):
        """

        :param frame: one frame from video read using cv2
        :param box_thickness: (integer) number of pixels for thickness of box
        :return: None

        Drawing a tracking box on the frame provided using the lastest bbox_list value
        Marks the box with the tracklet id
        If it is a ghost box, its velocity is shown at the bottom left of the box
        """
        velocity = None
        ghost_mode_list = list(self.ghost_mode_queue.queue)
        bbox_list = list(self.bbox_queue.queue)

        if ghost_mode_list[-1]:
            colour = [12, 12, 12]  # yellow for ghost boxes

        else:
            colour = [0, 0, 255]  # green for normal boxes

        bbox = bbox_list[-1]
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      colour, box_thickness)

        text = ''

        text += "ID: T{}".format(self.tracklet_id)

        if debug and velocity != None:
            text += ", velocity: {} pixels/sec".format(
                round(math.sqrt(math.pow(velocity[0], 2) + math.pow(velocity[1], 2))))

        # cv2.putText(frame, text, (bbox[0] + 4, bbox[3] - 5)
        #             , cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1, cv2.LINE_AA)
