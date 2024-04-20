from locust import HttpUser, between, task


class SAHIYOLOv8TestUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def test_sahi_yolo_v8_digits(self):
        response = self.client.post(
            url="/sahi/yolo_v8/digits/",
            files={
                "image_file": open("../resources/digits_test_img.jpg", "rb")
            },
            data={
                "slice_size_width_parameter": 160,
                "slice_size_height_parameter": 160,
                "slice_overlap_ratio_width_parameter": 0.3,
                "slice_overlap_ratio_height_parameter": 0.3
            }
        )
        print("Status code: ", response.status_code)
