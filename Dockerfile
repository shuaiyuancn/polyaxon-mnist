FROM tensorflow/tensorflow

RUN pip3 install --no-cache-dir -U polyaxon["s3","polyboard","polytune"]
