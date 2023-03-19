import cv2
import os
import pandas as pd
from deepface import DeepFace

#### Viewing results for recognition of distinct human face 
# img = cv2.imread("faces/1 (1).jpeg")

# result = DeepFace.analyze(img, actions=("gender", "age", "race", "emotion"))

# print(result)

### Further lines of code are for multiple data within directorium

data = {
    "Name": [],
    "Age": [],
    "Gender": [],
    "Race": []
}

for file in os.listdir("testFaces"):
    try:
        result = DeepFace.analyze(cv2.imread(f"testFaces/{file}"), actions = ("age", "gender", "race"))
        data["Name"].append(file.split(".")[0])
        data["Age"].append(result[0]["age"])
        data["Gender"].append(result[0]["dominant_gender"])
        data["Race"].append(result[0]["dominant_race"])
    except ValueError as e:
        print(f"Error analyzing {file}: {str(e)}")
df = pd.DataFrame(data)

print(df)
df.to_csv("people.csv")

### For testing purposes, 
