import numpy as np

vector = np.array(
    [1.5, 2.5, 3.5],
    dtype=np.float32
)

print(vector)
print(type(vector))
blob = vector.tobytes()

print(blob)
print(type(blob))
restored = np.frombuffer(
    blob,
    dtype=np.float32
)

print(restored)
print(type(restored))
blob = vector.tobytes()

print(blob)
print(type(blob))
restored = np.frombuffer(
    blob,
    dtype=np.float32
)

print(restored)
print(type(restored))