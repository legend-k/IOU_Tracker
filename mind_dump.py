def _light_left_click(self, x, y):
    point = (x, y)
    IoU_data = []
    _point_in_box_flag = False
    # self.count += 1

    if len(self.boxes) > 1:
        roi_list = self.boxes[:-1]
        timestamp = self.boxes[-1]

        atgmState = {}
        atgmState['zoom'] = self._atgmState.Zoom
        atgmState['azimuth'] = self._atgmState.Azimuth_steps / self.AZIMUTH_STEP_FACTOR
        atgmState['elevation'] = self._atgmState.Elevation_steps / self.ELEVATION_STEP_FACTOR

        for roi in roi_list:
            if self.point_in_box(point, roi):
                self.tracking_roi = roi
                _point_in_box_flag = True

        if _point_in_box_flag:
            IoU_data.append(roi_list)
            IoU_data.append(timestamp)
            IoU_data.append(self.tracking_roi)
            IoU_data.append(atgmState)

            self._imageProcessor.td_initializer_data = IoU_data
            self._imageProcessor.mode = ip.ImgProMode.TRACKING_WITH_DETECTION
        else:
            print("point not in bbox")
            return


def _lightmode_detection_event_handler(self, confidence_list=None, rectangles=None, final_image=None):
    #print("detection frame sent")
    self.boxes = rectangles
    self.atgm_server_frameSocket.write_frame(final_image)




