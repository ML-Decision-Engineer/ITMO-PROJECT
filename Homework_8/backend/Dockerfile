FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/project

WORKDIR /project

COPY requirements_ml.txt /tmp/requirements_ml.txt
COPY backend/requirements.txt /tmp/requirements_backend.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements_ml.txt \
    && pip install --no-cache-dir -r /tmp/requirements_backend.txt

COPY backend/app /project/app
COPY backend/tests /project/tests
COPY backend/pytest.ini /project/pytest.ini

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
