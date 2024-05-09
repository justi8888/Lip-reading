import os

import torch
import torchvision


def cut_or_pad(data, size, dim=0):
    """
    Pads or trims the data along a dimension.
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
        size = data.size(dim)
    elif data.size(dim) > size:
        data = data[:size]
    assert data.size(dim) == size
    return data


def load_video(path):
    """
    rtype: torch, T x C x H x W
    """
    vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    vid = vid.permute((0, 3, 1, 2))
    return vid


class AVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        label_path,
        video_transform,
        output_type,
        rate_ratio=640,
    ):

        self.root_dir = root_dir
        self.output_type = output_type

        self.rate_ratio = rate_ratio

        self.list = self.load_list(label_path)

        self.video_transform = video_transform

    def load_list(self, label_path):
        paths_counts_labels = []
        for path_count_label in open(label_path).read().splitlines():
            dataset_name, rel_path, input_length, char_id, char_nod_id, token_id = path_count_label.split(",")
            
            if self.output_type == 'char':
                paths_counts_labels.append(
                    (
                        dataset_name,
                        rel_path,
                        int(input_length),
                        torch.tensor([int(_) for _ in char_id.split()]),
                    )
                )
            elif self.output_type == 'char_nod':
                paths_counts_labels.append(
                    (
                        dataset_name,
                        rel_path,
                        int(input_length),
                        torch.tensor([int(_) for _ in char_nod_id.split()]),
                    )
                )
            elif self.output_type == 'token':
                paths_counts_labels.append(
                    (
                        dataset_name,
                        rel_path,
                        int(input_length),
                        torch.tensor([int(_) for _ in token_id.split()]),
                    )
                )
            else:
                raise NotImplementedError(f"{self.output_type} is not supported. Supported options are char, char_nod or token.")
        return paths_counts_labels


    def __getitem__(self, idx):
        dataset_name, rel_path, input_length, token_id = self.list[idx]
        path = os.path.join(self.root_dir, dataset_name, rel_path)
        video = load_video(path)
        #print(path)
        video = self.video_transform(video)
        return {"input": video, "target": token_id}
        
        # if self.output_type == 'char':
        #     return {"input": video, "target": char_id}
        # elif self.output_type == 'char_nod':
        #     return {"input": video, "target": char_nod_id}
        # elif self.output_type == 'token':
            
        # else:
        #     raise NotImplementedError(f"{self.output_type} is not supported. Supported options are char, char_nod or token.")

        

    def __len__(self):
        return len(self.list)
