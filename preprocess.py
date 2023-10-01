import torch
import json
import os
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import PIL

def make_dir(new_dir):
    try:
        os.mkdir(new_dir)
        print("Directory ", new_dir, " Created ")
    except FileExistsError:
        print("Directory ", new_dir, " already exists...")
    return new_dir


def pre_process(LOCAL_DATA_DIR="didi_dataset/",
                JSON_FILES=["diagrams_wo_text_20200131.ndjson", "diagrams_20200131.ndjson"]):
    sketch_folder = make_dir(LOCAL_DATA_DIR + 'sketch')
    diagram_folder = make_dir(LOCAL_DATA_DIR + 'diagram')
    metadata = []
    for json_file in JSON_FILES:
        count = 0
        with open(os.path.join(LOCAL_DATA_DIR, json_file)) as f:
            for line in tqdm(f):
                ink = json.loads(line)
                filename = ink["label_id"] + ".png"
                # save scaled diagram
                im = cv2.imread(os.path.join(LOCAL_DATA_DIR, "png", filename))
                guide_width = ink["writing_guide"]["width"]
                guide_height = ink["writing_guide"]["height"]
                im_height, im_width, _ = im.shape
                scale = min(guide_width / im_width, guide_height / im_height)
                offset_x = (guide_width - scale * im_width) / 2
                offset_y = (guide_height - scale * im_height) / 2
                dim = (int(scale * im_width), int(scale * im_height))
                resized = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(diagram_folder, filename), resized)

                # save sketch
                image = np.ones(resized.shape) * 255
                for inkk in ink["drawing"]:
                    s = inkk
                    sy = [y for y in s[1]]
                    sx = s[0]
                    if len(sy) != len(sx):
                        print('error {} != {} '.format(len(sy), len(s[0])))
                    color = (0, 0, 255)
                    thickness = 4
                    if len(sy) > 1:
                        for i in range(len(sy)):
                            start = (int(sx[i] - offset_x), int(sy[i] - offset_y))
                            end = (int(sx[i + 1] - offset_x), int(sy[i + 1] - offset_y))
                            image = cv2.line(image, start, end, color, thickness)
                            if i + 2 >= len(sy):
                                break
                cv2.imwrite(os.path.join(sketch_folder, filename), image)
                del ink['drawing']
                metadata.append(ink)

    with open('diagrams_ids.json', 'w') as fout:
        json.dump(metadata, fout)


class DiagramDataset(Dataset):
    def __init__(self, root='./didi_dataset/', percent=1, transform=None):
        super().__init__()
        self.root = root
        self.json_file = './diagrams_ids.json'
        with open(self.json_file) as f:
            self.data_idx = json.load(f)

        if percent < 1:
            self.data_idx = self.data_idx[0: int(len(self.data_idx) * percent)]

        self.transforms = transform

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        diagram_path = os.path.join(self.root, "diagram", self.data_idx[idx]["label_id"] + ".png")
        draw_path = os.path.join(self.root, "sketch", self.data_idx[idx]["label_id"] + ".png")
        diagram = cv2.imread(diagram_path)
        draw = cv2.imread(draw_path)
        dim = (1024, 512)
        diagram = cv2.resize(diagram, dim, interpolation = cv2.INTER_AREA)
        draw = cv2.resize(draw, dim, interpolation = cv2.INTER_AREA)
        diagram = cv2.cvtColor(diagram, cv2.COLOR_BGR2GRAY)  # binary
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)
        _, diagram = cv2.threshold(diagram, 127, 255, cv2.THRESH_BINARY)
        _, draw = cv2.threshold(draw, 127, 255, cv2.THRESH_BINARY)
        diagram = diagram / 255  # normalize (0-1)
        draw = draw

        if self.transforms:
            diagram = PIL.Image.fromarray(diagram)
            draw = PIL.Image.fromarray(draw)
            diagram = self.transforms(diagram)
            draw = self.transforms(draw)
        else:
            diagram = TF.to_tensor(diagram).float()
            draw = TF.to_tensor(draw).float()

        return draw, diagram


if __name__ == '__main__':
    pre_process()