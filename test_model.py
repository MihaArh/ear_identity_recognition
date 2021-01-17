import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory
HEIGHT = 224
WIDTH = 224

model_path = "models/model.h5"

model = models.load_model(model_path)
model.summary()

test_dir = "data/test"

test_dataset = image_dataset_from_directory(test_dir,
                                            labels="inferred",
                                            shuffle=False,
                                            batch_size=1,
                                            label_mode='int',
                                            image_size=(HEIGHT, WIDTH))

rank = [0] * 100
for x, y in test_dataset:
    subject_id = y.numpy()[0] + 1
    prediction = model.predict(x)
    prediction_list = list(enumerate(prediction[0], 1))
    prediction_list.sort(key=lambda x: x[1], reverse=True)
    found = False
    for id, predicted in enumerate(prediction_list):
        if predicted[0] == subject_id:
            print(f"Found subject {subject_id} with score {predicted[1]} on place {id}")
            found = True
            rank[id] = rank[id] + 1
        elif found:
            rank[id] = rank[id] + 1
rank = list(map(lambda x: x / 250, rank))
print(rank)
plt.plot(list(range(1, 101)), rank)
plt.ylabel('Rank - t Identification Rate (%)')
plt.xlabel('Rank (t)')
plt.title("CMC")
plt.show()

