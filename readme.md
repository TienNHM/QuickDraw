![](https://v1.screenshot.11ty.dev/https%3A%2F%2Fquickdraw.withgoogle.com%2F/opengraph)

# QUICK, DRAW!

Tập dữ liệu: Các dữ liệu được lấy bộ dữ liệu Quick Draw của Google, gồm 4 loại hình vẽ tay đơn giản:
- [Hình tròn](https://storage.cloud.google.com/quickdraw_dataset/full/numpy_bitmap/circle.npy)
- [Hình trái táo](https://storage.cloud.google.com/quickdraw_dataset/full/numpy_bitmap/apple.npy)
- [Hình tam giác](https://storage.cloud.google.com/quickdraw_dataset/full/numpy_bitmap/triangle.npy)
- [Hình gấp khúc](https://storage.cloud.google.com/quickdraw_dataset/full/numpy_bitmap/zigzag.npy)

Để chạy được, bạn phải tải đủ 4 file trên, đặt vào thư mục data/ 

## Cấu trúc thư mục

```shell
QuickDraw
│   cvt2img.py
│   Figure_1.png
│   Figure_2.png
│   LoadData.py
│   QD_trainer.py
│   QuickDrawApp.py
│   readme.md
│   requirements.txt
│   Screenshot 2021-01-04 210639.png
│
├───data
├───emojis
│       0.png
│       1.png
│       2.png
│       3.png
│       4.png
│
└───images
    ├───apple
    ├───circle
    ├───triangle
    └───zigzac
```

### Training

![](/Layers.png)

![](/Figure_1.png)

![](/Figure_2.png)

### References:
 
 - [Dataset](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap;tab=objects?pli=1&prefix=&forceOnObjectsSortingFiltering=false&pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22)))
 - [Google's Quick, Draw](https://quickdraw.withgoogle.com/) 
 - [The Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)
 - [Quick Draw: the world’s largest doodle dataset](https://towardsdatascience.com/quick-draw-the-worlds-largest-doodle-dataset-823c22ffce6b)
 - [Github akshaybahadur21/QuickDraw](https://github.com/akshaybahadur21/QuickDraw)
