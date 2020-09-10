FROM tensorflow/tensorflow

ADD model.py model.py

RUN pip3 install --no-cache-dir -U polyaxon["s3","polyboard","polytune"]

ENTRYPOINT [ "python", "model.py" ]
