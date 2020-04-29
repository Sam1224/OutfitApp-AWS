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
    - When creating the instance, you can let all traffic in for convenience; if not, open 80, 443 ports.
- vCPUs: 4
- Memory: 30.5GB
- GPU: NVIDIA Tesla M60 (8GB)
- Storage: 300GB
- Docker: v19.03.6
- Docker-compose: v1.26.0-rc3

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
- (optional) Setup a nginx server and config for `https` (if you need to use https for `production usage`):
    - Purchase a domain, e.g. from [Freenom](https://www.freenom.com/en/index.html?lang=en).
    - Request a elastic ip from AWS and bind to the current instance.
    - Bind the elastic ip to your domain in Freenom (the config will take about 1-2 days to take effects).
    - Install `certbot`:
```
$ sudo apt-get install software-properties-common
$ sudo add-apt-repository ppa:certbot/certbot
$ sudo apt-get update
$ sudo apt-get install certbot
```
    - Generate `certificate`, here, I use my own domain as an example, please replace them by your own domain:
```
$ sudo certbot certonly --standalone -d xusam2412.ml -d www.xusam2412.ml
$ sudo ls /etc/letsencrypt/live/
(you can find the generated keys here)
```
    - Install and `Nginx`:
```
$ sudo apt-get install nginx
$ sudo vim /etc/nginx/sites-available/default
1. find the row containing 'listen [::]:80 default_server;'.
2. modify the row as 'listen [::]:80 ipv6only=on default_server;'.
(if not, the nginx service may not start up normally due to the confliction of port used)

$ sudo vim /etc/nginx/nginx.conf
1. add the following script inside the 'http' block, you should replace the ssl_certificate and ssl_certificate_key values with your own files' paths.
server {
  listen 443;
  server_name xusam2412.ml www.xusam2412.ml;
  ssl on;
  ssl_certificate /etc/letsencrypt/live/xusam2412.ml/fullchain.pem;
  ssl_certificate_key /etc/letsencrypt/live/xusam2412.ml/privkey.pem;
  location /api_server {
    proxy_pass http://0.0.0.0:5000/api_server;
    proxy_http_version 1.1;
    proxy_set_header X_FORWARDED_PROTO https;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header Host $host;
  }
  location /mmfashion {
    proxy_pass http://0.0.0.0:5012/mmfashion;
    proxy_http_version 1.1;
    proxy_set_header X_FORWARDED_PROTO https;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header Host $host;
  }
}

$ sudo service nginx restart
```
- Build docker containers and images:
```
$ docker-compose up -d
1. 1st time setup takes a long period to create the containers and images.

$ watch -n 1 nvidia-smi
1. once the containers and images are built, it still needs some time to start up because the flask server needs some time to initialize. A tip is using this command to monitor the gpu memory usage to judge if the initialize is done.
2. when the process of initializing finishing, the gpu memory usage will be about 2700MiB.
```

## References
- [Docker-compose](http://www.runoob.com/docker/docker-compose.html)
- [Yagami360](https://github.com/Yagami360/virtual-try-on_webapi_flask)