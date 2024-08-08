# Predictive Maintenance 

### Technical Details

Predictive maintenance backend server for model inferance upload and update.

### Techstack

- Tenserflow
- Fast-API
- Docker
- Shap
- Keras
- Numpy
- Pandas

## Usage

### Python version: Python 3.10.12 to 3.10.14

### Method 1: Docker Container

1. Clone the repository
```bash
git clone https://github.com/danishzulfiqar/wiser-chenab-model-container.git
```

2. Change directory
```bash
cd wiser-chenab-model-container
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
git clone https://github.com/danishzulfiqar/wiser-chenab-model-container.git
```

2. Change directory
```bash
cd wiser-chenab-model-container
```

3. Make env
```bash
python -m venv .venv
```

4. Install dependencies
```bash
pip install -r requirements.txt --no-cache-dir
```
4. Run Fast-API
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