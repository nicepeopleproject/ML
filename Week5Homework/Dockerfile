FROM agrigorev/zoomcamp-model:3.8.12-slim

RUN pip install pipenv

COPY ["model1.bin", "dv.bin", "Pipfile", "Pipfile.lock", "predict.py", "predict_test.py", "./"]

RUN pipenv install --system --deploy

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]