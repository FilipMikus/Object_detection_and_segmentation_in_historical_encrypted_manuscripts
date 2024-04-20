from locust import HttpUser, between, task


class DataExplorerTestUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def test_data_explorer_similarity(self):
        response = self.client.post(
            url="/data/explorer/similarity/",
            files={
                "image_file": open("../resources/digits_crop_test_img.jpg", "rb"),
                "dataset_file": open("../resources/test_yolo_dataset.zip", "rb")
            }
        )
        print("Status code: ", response.status_code)
