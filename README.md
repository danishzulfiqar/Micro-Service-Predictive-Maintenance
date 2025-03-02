# Predictive Maintenance 

### Technical Details

Predictive maintenance backend server for model inferance, upload and update.


### Techstack

- Tenserflow
- Fast-API
- Docker
- Shap
- Keras
- Numpy
- Pandas

## Running the code

### Python version: Python 3.10.12 to 3.10.14

### Method 1: Docker Container

1. Clone the repository
```bash
git clone https://github.com/danishzulfiqar/Micro-Service-Predictive-Maintenance.git
```

2. Change directory
```bash
cd Micro-Service-Predictive-Maintenance
```

3. Make image
```bash
docker build -t api .
```

4. Run image in a container
```bash
docker run --name pred-men -p 8000:8000 api
```

5. Open App at localhost:
http://localhost:8000


### Method 2: venv

1. Clone the repository
```bash
git clone https://github.com/danishzulfiqar/Micro-Service-Predictive-Maintenance.git
```

2. Change directory
```bash
cd Micro-Service-Predictive-Maintenance
```

3. Make env
```bash
python -m venv .venv
```

4. Select env
```bash
source .venv/bin/activate
```

5. Install dependencies
```bash
pip install -r requirements.txt --no-cache-dir
```

6. Run Fast-API
```bash
fastapi run app/main.py --port 8000 --reload
```

6. Open App at localhost:
http://localhost:8000


### Deployment

## AWS EC2

### Recommended: ubunto t2.small or greater machines

Note: 
1. The application is optimised and runs with ubunto (t2.micro) as well but for higher traffic and optimised response use recommended machines.
2. Change tensorflow to tensorflow-cpu in requiremnts.text for deployment.


## Running Server

### 1. Without Docker

1. Make .venv
```bash
python -m venv .venv
```

2. Move to venv
```bash
source .venv/bin/activate
```

3. Install requirements
```bash
pip install -r requirements.txt --no-cache-dir 
```

4. Run app
```bash
nohup fastapi run app/main.py --port 8000
```

### 2. With Docker

1. Build image
```bash
docker build -t api .
```

2. Run image
```bash
docker run --name pred-men -p 8000:8000 api
```

## Microservice Usage

### Documentation and Routes

1. Open the browser and paste the link below
```bash
http://localhost:8000/docs
```

### Note:

For development of models compatible with this microservice, please refer: [Predictive-Maintenance-Model-Training](https://github.com/danishzulfiqar/Predictive-Maintenance-Model-Training.git)