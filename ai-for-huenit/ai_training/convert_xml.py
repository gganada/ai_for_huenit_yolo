import os
import argparse
from tqdm import tqdm
import cv2
from lxml import etree

def convert_to_xml(image_path, label_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    
    if not os.path.exists(label_path):
        print(f"Label path does not exist: {label_path}")
        return
    
    label_files = [f for f in os.listdir(label_path) if f.endswith(".txt")]
    
    for filename in tqdm(label_files):
        filename_str = os.path.splitext(filename)[0]
        img_file = os.path.join(image_path, filename_str + ".jpg")
        
        if not os.path.exists(img_file):
            print(f"Image file not found: {img_file}")
            continue
        
        annotation = etree.Element("annotation")

        folder = etree.Element("folder")
        folder.text = os.path.basename(image_path)
        annotation.append(folder)

        filename_xml = etree.Element("filename")
        filename_xml.text = filename_str + ".jpg"
        annotation.append(filename_xml)

        path = etree.Element("path")
        path.text = img_file
        annotation.append(path)

        source = etree.Element("source")
        annotation.append(source)
        database = etree.Element("database")
        database.text = "Unknown"
        source.append(database)

        size = etree.Element("size")
        annotation.append(size)

        img = cv2.imread(img_file)
        try:
            width = etree.Element("width")
            width.text = str(img.shape[1])
            height = etree.Element("height")
            height.text = str(img.shape[0])
            depth = etree.Element("depth")
            depth.text = str(img.shape[2])
            
            size.append(width)
            size.append(height)
            size.append(depth)
        except AttributeError:
            print(f"Error reading image: {img_file}")
            continue

        segmented = etree.Element("segmented")
        segmented.text = "0"
        annotation.append(segmented)

        label_file_path = os.path.join(label_path, filename)
        with open(label_file_path, 'r') as label_file:
            for line in label_file:
                line = line.strip()
                l = line.split(' ')

                if len(l) < 5:
                    continue

                class_name = "_".join(l[:-4])
                xmin_l = str(int(round(float(l[-4]))))
                ymin_l = str(int(round(float(l[-3]))))
                xmax_l = str(int(round(float(l[-2]))))
                ymax_l = str(int(round(float(l[-1]))))

                obj = etree.Element("object")
                annotation.append(obj)

                name = etree.Element("name")
                name.text = class_name
                obj.append(name)

                pose = etree.Element("pose")
                pose.text = "Unspecified"
                obj.append(pose)

                truncated = etree.Element("truncated")
                truncated.text = "0"
                obj.append(truncated)

                difficult = etree.Element("difficult")
                difficult.text = "0"
                obj.append(difficult)

                bndbox = etree.Element("bndbox")
                obj.append(bndbox)

                xmin = etree.Element("xmin")
                xmin.text = xmin_l
                bndbox.append(xmin)

                ymin = etree.Element("ymin")
                ymin.text = ymin_l
                bndbox.append(ymin)

                xmax = etree.Element("xmax")
                xmax.text = xmax_l
                bndbox.append(xmax)

                ymax = etree.Element("ymax")
                ymax.text = ymax_l
                bndbox.append(ymax)
        
        xml_output_path = os.path.join(output_path, filename_str + ".xml")
        with open(xml_output_path, 'wb') as f:
            s = etree.tostring(annotation, pretty_print=True, xml_declaration=True, encoding='UTF-8')
            f.write(s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert label files to XML format")
    parser.add_argument("image_path", type=str, help="Path to the images directory")
    parser.add_argument("label_path", type=str, help="Path to the labels directory")
    parser.add_argument("output_path", type=str, help="Path to save XML files")

    args = parser.parse_args()
    convert_to_xml(args.image_path, args.label_path, args.output_path)
