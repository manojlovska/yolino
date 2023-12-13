import os
import math

import torch
import torchvision
import torchvision.transforms as transforms
from yolino.dataset.dataset_base import DatasetInfo
from yolino.utils.enums import Dataset
import xml.etree.ElementTree as ET
import numpy as np
from yolino.utils.duplicates import LineDuplicates
from yolino.utils.logger import Log

class DAISDataSet(DatasetInfo):

    @classmethod
    def height(self) -> int:
        return 640

    @classmethod
    def width(self) -> int:
        return 1280
    
    def __init__(self, split, args, augment=False, sky_crop=80, side_crop=0, load_only_labels=False, # proveri sto znacat argumentite
                 show=False, load_full_dataset=False, lazy=False, ignore_duplicates=False, store_lines=False):
        super().__init__(Dataset.DAIS, split, args, sky_crop=sky_crop, side_crop=side_crop, augment=augment,
                         num_classes=0, train=3751, test=-1, val=1044,
                         override_dataset_path="/home/manojlovska/Documents/YOLinO/dais_po_8p_dn19/DAIS-COCO", # preveri kaj je to
                         load_only_labels=load_only_labels, show=show, load_sequences=load_full_dataset, lazy=lazy,
                         ignore_duplicates=ignore_duplicates, store_lines=store_lines)
        
        self.xml_file = "annotations.xml"

        self.tree = ET.parse(os.path.join(self.get_dataset_path("dais", self.override_dataset_path), "annotations_xml", self.xml_file))
        self.root = self.tree.getroot()

        self.file_names = []
        self.img_list = []
        self.tus_img_size = (640, 1280)
        self.annotations = self.load_annotations()
        # print("self.annotations: ", self.annotations[:])


    def print_instance_attributes(self):
        for attribute, value in self.__dict__.items():
            print(attribute, '=', value)


    def load_annotations(self):

        all_images = self.root.findall("image")
        images = [all_images[i] for i in range(0,len(all_images)) if all_images[i].attrib["subset"]==self.split]

        annotations = []
        for image_elem in images:
            file_name = image_elem.get("name")
            self.file_names.append(file_name)
            self.img_list.append(os.path.join(self.get_dataset_path("dais", self.override_dataset_path), self.split, file_name))

            img_info = (int(image_elem.get("height")), int(image_elem.get("width")))

            height = img_info[0]
            width = img_info[1]

            objs = []
            image_res = []
            for obj in image_elem.findall("polyline"):
                points_string = obj.get("points")

                all_points = []
                for point_string in points_string.split(";"):
                    coord_string = point_string.split(",")
                    x = float(coord_string[0])
                    y = float(coord_string[1])
                    all_points.append((x,y)) # for single magnetic tape

                obj.set("all_points", all_points)
                objs.append(obj)
                num_points = len(all_points)

                # res = np.zeros((num_points,),dtype='i,i').tolist()
                image_res.append(all_points)

            num_tapes = len(objs)
            # print(num_tapes)
            
            # logger.info("num_objs: {}".format(num_objs))

            r_height =  self.tus_img_size[0] / height
            r_width = self.tus_img_size[1] / width
            # min(self.tus_img_size[0] / height, self.tus_img_size[1] / width) # popravi
            #  
            image_res_adapted = []
            for res in image_res:
                res = [(tup[0] * r_width, tup[1] * r_height) for tup in res] # (y, x) je vsak point
                # res = [tuple(x * r for x in tpl) for tpl in res]
                # print(res)
                # print(" ")
                # print(list(reversed(res)))
                # print(" ")
                image_res_adapted.append(res) # list(reversed(res))
            # print(image_res_adapted)
            

            img_info = (height, width)
            resized_info = (int(height * r_height), int(width * r_width))
            # print(img_info)
            # print(resized_info)

            annotation = (image_res_adapted, img_info, resized_info, file_name)
            annotations.append(annotation)
            # print(annotation)
            # print("")
            # print(annotation[0])
            # print("")
        # logger.info("len(annotations): {}".format(len(annotations)))


        # Make all the annotations to have the same number of instances and the same number of points for every instance
        # 1. Check the max number of points an instance has in all images
        self.max_num_points = np.max([len(annotations[i][0][j]) for i in range(len(annotations)) for j in range(len(annotations[i][0]))])
        print("max_num_points: ", self.max_num_points)

        # 2. Check if number of instances is 5, if not add empty lists
        self.lanes = annotations

        for i in range(len(self.lanes)):
            if len(self.lanes[i][0]) < 5:
                listOfLists = [[] for l in range(5 - len(self.lanes[i][0]))]
                self.lanes[i][0].extend(listOfLists)

        # 3. Add negative numbers if number of points is smaller than max_num_points
        for i in range(len(self.lanes)):
            for j in range(len(self.lanes[i][0])):
                if len(self.lanes[i][0][j]) < self.max_num_points:
                    self.lanes[i][0][j].extend(((-2,-2), ) * (self.max_num_points-len(self.lanes[i][0][j])))

        return annotations
    
    def load_anno(self, index):
        return self.annotations[index][0]
    
    def __get_labels__(self, idx):
        gridable_lines = torch.ones((5, self.max_num_points, 2), dtype=torch.float32) * torch.nan

        # Tusimple GT runs from horizon to bottom of the image.
        # We want to encode the driving direction in the arrows and thus reverse the GT.
        # print("self.lanes[idx]: ", self.lanes[idx])
        
        for i in range(len(self.lanes[idx][0])):
            gridable_lines[i, :, 1] = torch.tensor([coords[0] for coords in self.lanes[idx][0][i][:]])
            gridable_lines[i, :, 0] = torch.tensor([coords[1] for coords in self.lanes[idx][0][i][:]])

        gridable_lines[gridable_lines[:, :, 1] < 0] = torch.tensor([torch.nan, torch.nan])

        # print("Gridable lines:", gridable_lines)
        # print(" ")

        return gridable_lines
    
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if self.load_only_labels:
            image = self.dummy_image([1080, 1920])
        else:
            cv_image = self.__load_image__(idx)
            image = self.__make_torch__(cv_image)
            del cv_image

        lines = self.__get_labels__(idx)

        # print(self.annotations[idx][-1])
        # print(" ")
        # print(self.annotations[idx][0])
        # print(" ")
        # print("tensors: ")

        # tensors = [torch.tensor(self.annotations[idx][0][i]) for i in range(len(self.annotations[idx][0]))]

        # print(tensors)

        # lines =  torch.cat(tensors) if self.annotations[idx][0] else torch.tensor([[torch.nan, torch.nan]]) # torch.tensor(self.annotations[idx][0])

        # print(" ")
        # print(lines)
        # print("unsqueeze")
        # print(torch.unsqueeze(lines, dim=0))
        # print("lines unsqueezed")
        # lines = torch.unsqueeze(lines, dim=0)
        # print(lines)
        # print("******************************")
        image, lines, params = self.__augment__(idx, image, lines)

        try:
            duplicates = LineDuplicates(filename=self.file_names[idx], grid_shape=self.args.grid_shape,
                                        num_predictors=self.args.num_predictors)
            grid_tensor, grid = self.__get_grid_labels__(torch.unsqueeze(lines, dim=0), [],
                                                         idx, image=image,
                                                         duplicates=duplicates)
        except ValueError as e:
            Log.error("Error in %s" % (self.img_list[idx]))
            raise e

        if idx == 0:
            Log.debug("Shapes from TuSimple:")
            Log.debug("\tImage: %s" % str(image.shape))  # (3, h, w)
            Log.debug("\tGrid Lines: %s" % str(grid_tensor.shape))  # (grid_shape, preds, coords)

        return image, grid_tensor, self.file_names[idx], duplicates.dict(), params
    
    @classmethod
    def get_max_image_size(self):
        # Log.error("You should not use the dataset base!")
        return 640, 1280

    def check_img_size(self):
        if not np.all(math.isclose(np.divide(*self.args.img_size), self.height() / self.width())) or \
            not np.all(np.mod(self.args.img_size, 32) == 0):
            # 1280 x 720 px
            Log.warning("Tusimple together with Grid-YOLO on 32x32 px cells only accepts "
                        "an aspect ratio of %f. In addition the height and width must be dividable by 32. "
                        "You provided %s. "
                        "We suggest 320x640 or 640x1280." %
                        (self.height() / self.width(), str(self.args.img_size)))
            return False

        return True
    
    @classmethod
    def get_img_width(self, height) -> int:
        return math.ceil(height * self.width() / self.height())
    
