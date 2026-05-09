import io
import random
from locust import HttpUser, task, between

class DefectDetectionUser(HttpUser):
    # wait_time = between(1, 2) # Constant 10 RPS is specified, we can control via CLI

    @task
    def predict(self):
        category = random.choice(["bottle", "capsule", "carpet", "hazelnut", "leather", "pill"])
        
        # Create a dummy image
        from PIL import Image
        img = Image.new('RGB', (224, 224), color = (73, 109, 137))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)

        # POST /predict
        self.client.post(
            "/predict",
            data={"category": category},
            files={"file": ("dummy.jpg", img_byte_arr.read(), "image/jpeg")},
            name="/predict"
        )
