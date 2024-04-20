from locust import HttpUser, between, task


class DataAnnotatorTestUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def test_data_annotator_digits(self):
        response = self.client.post(
            url="/data/annotator/digits/",
            files={
                "dataset_images_file": open("../resources/test_digits_images.zip", "rb")
            }
        )
        print("Status code: ", response.status_code)
