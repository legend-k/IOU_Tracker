import math


class AngleMath(object):

    def __init__(self, resolution=(720, 480), base_focal_length=3.5):
        self._width, self._height = resolution
        self._base_focal = base_focal_length

    def get_azimuth_angle(self, zoom_level, pixel_from_center):
        '''
        Converts pixel values to corresponding 2d horizontal angles with frame center as reference
        :param zoom_level: int - zoom level
        :param pixel_from_center:  - int - pixel distance from frame w/2
        :return:
        '''
        result = (180 / math.pi) * (
            math.atan((3.6 * pixel_from_center) / (zoom_level * self._base_focal * self._width)))
        return result

    def get_elevation_angle(self, zoom_level, pixel_from_center):
        '''
        Converts pixel values to corresponding 2d vertical angles with frame center as reference
        :param zoom_level: int - zoom level
        :param pixel_from_center:  - int - pixel distance from frame h/2
        :return:
        '''
        result = (180 / math.pi) * (
            math.atan((2.7 * pixel_from_center) / (zoom_level * self._base_focal * self._height)))
        return result

    def get_horizontal_pixel(self, zoom_level, relative_angle):
        '''
        Converts a given horizontal angle into pixel from center of frame
        :param zoom_level: int - zoom level
        :param relative_angle: angle in degrees relative to center of frame
        :return:
        '''
        result = (self._width * math.tan(relative_angle*(math.pi/180)) * zoom_level * 3.5) / 3.6
        return result


    def get_vertical_pixel(self, zoom_level, relative_angle):
        '''
        Converts a given horizontal angle into pixel from center of frame
        :param zoom_level: int - zoom level
        :param relative_angle: angle in degrees relative to center of frame
        :return:
        '''
        result = (self._height * math.tan(relative_angle*(math.pi/180)) * zoom_level * 3.5) / 2.7
        return result

    def bbox_to_steps(self, zoom_level, bbox):
        '''
        Given the bounding box coordinates, gives the x-steps and y-steps
        :param zoom_level: (int) zoom level of the atgm launcher
        :param bbox: (list) xmin, ymin, xmax, ymax
        :return: (tuple) (x_steps, y_steps)
        '''
        x_from_center = (bbox[0]+bbox[2])//2 - 360
        y_from_center = (bbox[1]+bbox[3])//2 - 240

        x_steps = 10421 * self.get_azimuth_angle(zoom_level, x_from_center)
        y_steps = 22390 * self.get_elevation_angle(zoom_level, y_from_center)

        return (x_steps, y_steps)