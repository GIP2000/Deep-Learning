import torch
import clip
from PIL import Image


class ClipHandler:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        self.model = model
        self.preprocess = preprocess
        self.labels = None

    def _processImage(self, path:str):
        return self.preprocess(Image.open(path)).unsqueeze(0).to(self.device)

    def setLabels(self, labels:list[str]):
        self.labels = self._tokenize(labels)

    def _tokenize(self, labels: list[str]):
        return clip.tokenize(labels).to(self.device)

    def predict(self, imagePath: str):
        if self.labels is None:
            raise ValueError("Please Set the label values first")

        image = self._processImage(imagePath)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(self.labels)

            logits_per_image, logits_per_text = self.model(image, self.labels)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        return probs


if __name__ == "__main__":
    print("Starting")
    ch = ClipHandler()
    ch.setLabels(["a diagram", "a dog", "a cat"])
    print(ch.predict("../data/clip.jpeg"))
