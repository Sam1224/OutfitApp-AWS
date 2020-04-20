# OutfitApp-AWS
Deploy the AI models to AWS for the final year project Custom Outfit App.

- Table Of Contents
  - [OutfitApp-AWS](#outfitapp-aws)
    * [Basic Information](#basic-information)
    * [Environment](#environment)
    * [Usage](#usage)
    * [References](#references)

## Basic Information
- ID: 20086454
- Name: Qianxiong Xu
- Major: BSc (Hons) in Software Systems Practice Year 1

## Environment
- OS: Ubuntu 16.04
- AMI: Deep Learning AMI (Ubuntu 16.04) Version 27.0 64-bit(x86)
- Instance: AWS g3s.xlarge
- vCPUs: 4
- Memory: 30.5GB
- GPU: NVIDIA Tesla M60 (8GB)
- Storage: 300GB

## Usage
- Clone:
```
$ cd <your_dir>
$ git clone https://github.com/Sam1224/OutfitApp-AWS.git
$ cd OutfitApp-AWS/api_server
$ git clone https://github.com/Sam1224/mmfashion.git
$ cd ..
```
- Download models and datasets:
```
$ sh download_models.sh
```
- Build docker containers and images:
```
$ docker-compose up -d
(1st time setup takes a long period)
(the flask server needs some time to initialize)
```

## References
- [Docker-compose](http://www.runoob.com/docker/docker-compose.html)
- [Yagami360](https://github.com/Yagami360/virtual-try-on_webapi_flask)