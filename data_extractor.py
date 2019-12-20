import os
import csv
import urllib.request


def get_data(params):
    image_path = params["image_path"]

    if len(os.listdir(image_path)) > 100:
        return

    url_file = open(params["url_path"], "r")
    reader = csv.reader(url_file)

    print("Image retrieval: [", end="", flush=True)
    try:
        for i, row in enumerate(reader):
            imid, url = row

            if i % 100 == 0:
                print("=", end="", flush=True)

            try:
                urllib.request.urlretrieve(url, image_path + str(imid) + ".png")
            except:
                continue

    except KeyboardInterrupt:
        pass
    print("]")
