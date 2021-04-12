import tensorflow as tf

x = [[0,1,2,3,4,3,2,1,0,-1,-2,-3,-4,-3,-2,-1,0,0,0,0,0,0,0,0,0,0,0,0,0],  #training dataset's list of list, use more to improve accuracy. all list sizes must be same
           [0,0,0,0,0,0,0,0,1,2,3,4,3,2,1,0,-1,-2,-3,-4,-3,-2,-1,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-2,-3,-4,-3,-2,-1,0,0,0,0,0,0],
           [0,1,0,0,2,0,0,0,1,2,3,4,3,2,1,0,-1,-2,-3,-4,-3,-2,-1,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0,-0,-0,-0,-0,-0,-0,0,0,0,0,0,0],
           [0,1,0,0,2,0,0,0,1,2,3,4,3,2,1,0,-1,-2,-3,-4,-3,-2,-1,0,3,0,4,0,1],
           [0,0,0,0,1,2,3,-3,4,0,0,0,4,0,0,0,-0,-2,-0,-0,-1,-0,-0,0,0,0,0,0,0],
           [1,2,3,4,5,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-4,-3,-2,-1,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,1,2,3,4,5,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-5,-4,-3,-2,-1],
           [0,0,0,1,2,3,4,5,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-4,-3,-2,-1,0,0,0,0,0],
           [0,0,0,0,0,2,4,8,4,2,0,1,0,0,0,0,0,0,0,-1,-2,-3,-4,-3,-2,-1,0,0,0],
           [0,0,0,0,0,-1,-2,-3,-4,-3,-2,-1,0,0,0,0,0,0,0,-1,-2,-3,-4,-3,-2,-1,0,0,0],
           [2,3,4,3,2,1,0,-1,-2,-3,-2,-1,-0,-0,-0,2,4,8,4,2,0,-2,-4,-8,-4,-2,0,0,0],
           [2,3,4,3,2,1,0,-1,-2,-3,-2,-1,-0,-0,-0,-2,-4,-8,4,2,0,-2,-4,-8,-4,-2,0,0,0],
           [2,3,4,3,2,1,0,1,2,-3,-2,-1,-0,-0,-0,2,4,8,4,2,0,-2,-4,-8,-4,-2,0,0,0]]


y=[1,1,0,1,0,1,0,0,1,1,0,0,1,0,1] # 1 for sinewave type shape, 0 otherwise, put serially accodrding to the trainisng set's list of list (x)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x, y, epochs=5)



model.save("NN.model")

while(1):
    pass
