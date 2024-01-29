import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(10,)),
        Dense(1, activation='sigmoid')
    ])
    return model

def train_model(config):
    # Check if GPU is available and being used
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        print('GPU device not found')
    else:
        print('Found GPU at: {}'.format(device_name))

    model = create_model()
    model.compile(optimizer=Adam(learning_rate=config["lr"]), loss='binary_crossentropy', metrics=['accuracy'])

    # Dummy data
    x_train = tf.random.normal([100, 10])
    y_train = tf.random.uniform([100], minval=0, maxval=2, dtype=tf.int32)

    # Training
    history = model.fit(x_train, y_train, epochs=10, batch_size=10)
    accuracy = history.history['accuracy'][-1]
    tune.report(accuracy=accuracy)

def main():
    ray.init()

    config = {
        "lr": tune.grid_search([0.001, 0.01, 0.1])
    }

    analysis = tune.run(
        train_model,
        resources_per_trial={"gpu": 1},
        config=config,
        metric="accuracy",
        mode="max",
        num_samples=1,
        scheduler=ASHAScheduler()
    )

    print("Best config: ", analysis.best_config)
    ray.shutdown()

if __name__ == "__main__":
    main()
