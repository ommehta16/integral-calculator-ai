import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input,TextVectorization,Embedding, LSTM,Dense,Concatenate,)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

#Input csv name
df = pd.read_csv( " ", header=None, names=["function", "lower", "upper", "true_value"])

df["lower"] = df["lower"].astype(float)
df["upper"] = df["upper"].astype(float)
df["true_value"] = df["true_value"].astype(float)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

#Adjust sequence length and token count if I want
max_tokens = 1000
sequence_length = 32

vectorizer = TextVectorization(max_tokens=max_tokens,output_mode="int",output_sequence_length=sequence_length)
vectorizer.adapt(train_df["function"].astype(str).values)

# Function for model
def df_to_dataset(dataframe, batch_size=32, shuffle=True):
    df = dataframe.copy()
    functions = df["function"].astype(str).values
    lowers = df["lower"].values
    uppers = df["upper"].values
    targets = df["true_value"].values

    ds = tf.data.Dataset.from_tensor_slices(((functions, lowers, uppers), targets))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Making datasets
batch_size = 32
train_ds = df_to_dataset(train_df, batch_size=batch_size)
val_ds = df_to_dataset(val_df, batch_size=batch_size, shuffle=False)

func_input = Input(shape=(), dtype=tf.string, name="func_input")
x = vectorizer(func_input)
x = Embedding(input_dim=max_tokens, output_dim=32)(x) 
x = LSTM(32)(x)

# Creating model
lower_input = Input(shape=(1,), dtype=tf.float32, name="lower_input")
upper_input = Input(shape=(1,), dtype=tf.float32, name="upper_input")

# Concatenating lower and upper bounds
y = Concatenate()([lower_input, upper_input])
y = Dense(16, activation="relu")(y)
y = Dense(8, activation="relu")(y)

z = Concatenate()([x, y])
z = Dense(32, activation="relu")(z)
z = Dense(16, activation="relu")(z)
output = Dense(1, activation="linear", name="predicted_integral")(z)

model = Model(inputs=[func_input, lower_input, upper_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])

model.summary()

# Training
epochs = 20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
)

# Evaluating on validation set
val_loss, val_mae = model.evaluate(val_ds)
print(f"Validation MSE: {val_loss:.4f}, MAE: {val_mae:.4f}")

# Inferring (not really needed but maybe)
example_func = np.array(["x**2"])
example_lower = np.array([0.0])
example_upper = np.array([1.0])
pred = model.predict([example_func, example_lower, example_upper])
print(f"Predicted integral of x^2 from 0 to 1: {pred[0][0]:.4f}")
