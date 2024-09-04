# HSI-feature-extraction
HSI-feature-extraction
# How to build CPU Environment
docker info | grep -i "storage driver"
```
docker build -f Dockerfile-cpu -t hsi-cpu .
docker run -it --rm -v $(pwd):/workspace/HSI hsi-cpu
```
# How to build GPU Environment
```
docker build -t hsi .
docker run --gpus all -it --rm -v $(pwd):/workspace/HSI hsi
```
docker run時にエラーが起きたらnvidia-dockerをインストールしましょう。[Installing the NVIDIA Container Toolkit
](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
