
from conter import *
from showClassInModel import *
from ultralytics import YOLO

def main():
    video=r"F:\152\Count-entering-and-exiting-people-using-YOLOv8-main\p.mp4"
    model=r"F:\152\Count-entering-and-exiting-people-using-YOLOv8-main\yolov8s.pt"
    
    model=YOLO(model)
    # showClass(model.names)
    # showDatainFile()
    
    conter=Conter(video,model)
    conter()

if __name__ == "__main__":
    main()