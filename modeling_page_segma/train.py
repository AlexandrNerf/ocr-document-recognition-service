import ultralytics

def train_obb():
    model = ultralytics.YOLO('yolov8m-obb.pt')

    model.train(
        data="data/data.yaml", 
        epochs=100,
        imgsz=640
        )

if __name__ == '__main__':
    train_obb()