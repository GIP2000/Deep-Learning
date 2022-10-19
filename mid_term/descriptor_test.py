import openai
import json

openai.api_key = "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"

class LabelsWithDescriptors:

    def __init__(self, index, labels, descriptors=None):
        self.index = index
        self.labels = labels
        if descriptors is not None:
            self.descriptors = descriptors
        else:
            print(self.labels[0])
            print(self.get_parsed_response(self.labels[0]))

    @staticmethod
    def build_input(category_name: str) -> str:
        return f"""Q: What are useful visual features for distinguishing a lemur in a photo?
A: There are several useful visual features to tell there is a lemur in a
     photo:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet

Q: What are useful visual features for distinguishing a {category_name} in a photo?
A: There are several useful visual features to tell there is a {category_name} in a
     photo
-"""

    @staticmethod
    def get_response(input: str):
        return openai.Completion.create(
          model="text-davinci-002",
          prompt=LabelsWithDescriptors.build_input(input),
          temperature=0.7,
          max_tokens=100,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )

    @staticmethod
    def get_parsed_response(input:str):
        raw = LabelsWithDescriptors.get_response(input).get("choices")[0]["text"]
        raw.split("\n")


def main(f):
    cats = json.load(f)['cats']

    for (i, labels) in enumerate(cats):
        if i >= 1:
            break
        l = LabelsWithDescriptors(i,labels)



if __name__ == "__main__":
    with open("ImageNetClassifers.json") as f:
        main(f)

