from brainnet_model import build_brainnet_model

model = build_brainnet_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
