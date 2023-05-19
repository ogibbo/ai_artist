import matplotlib.pyplot as plt

# Plotting 9 images from the dataset
def show_ds_examples(ds):
  plt.figure(figsize=(10, 10))
  for images in ds.take(1):
    for i in range(9):
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"))
      plt.axis("off")