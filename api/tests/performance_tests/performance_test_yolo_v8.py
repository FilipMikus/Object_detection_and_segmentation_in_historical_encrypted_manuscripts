from locust import HttpUser, between, task


class YOLOv8TestUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def test_yolo_v8_digits(self):
        response = self.client.post(
            url="/yolo_v8/digits/",
            files={
                "image_file": open("../resources/digits_crop_test_img.jpg", "rb")
            },
            data={
                "confidence_parameter": 0.25,
                "iou_parameter": 0.7,
                "image_size_parameter": 640
            }
        )
        print("Status code: ", response.status_code)