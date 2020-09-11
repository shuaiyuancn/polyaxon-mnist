import argparse
import tensorflow as tf

# Polyaxon
from polyaxon.tracking import Run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    mnist = tf.keras.datasets.mnist

    # Polyaxon
    experiment = Run()

    # Polyaxon
    experiment.log_data_ref(content=x_train, name="x_train")
    experiment.log_data_ref(content=y_train, name="y_train")
    experiment.log_data_ref(content=x_test, name="x_test")
    experiment.log_data_ref(content=y_test, name="y_test")

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(args.dropout),
            tf.keras.layers.Dense(10),
        ]
    )

    predictions = model(x_train[:1]).numpy()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=args.optimizer, loss=loss_fn, metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=args.epochs)

    loss, acc = model.evaluate(x_test, y_test, verbose=2)

    # Polyaxon
    experiment.log_metrics(loss=loss)
    experiment.log_metrics(accuracy=acc)
