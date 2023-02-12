import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Asetetaan polut 'train/' ja 'test/' hakemistoihin
train_dir = 'images/train/'
test_dir = 'images/test/'

# Luodaan CNN malli opettamista varten
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(3, activation='sigmoid'))

# Käännetään/rakennetaan malli
opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

## Kuvien lukeminen hakemistoista ja mallin opettaminen 

# Luodaan datageneraattori kuvien lukemiseksi hakemistoista
datagen = ImageDataGenerator(rescale=1.0/255.0)

# Valmistellaa kuvien lukuvirrat datageneraattorin avulla (useammalle kuvaluokalle käytä class_mode='categorical')
train_iterator = datagen.flow_from_directory(train_dir, class_mode='categorical', batch_size=64, target_size=(200, 200))
test_iterator = datagen.flow_from_directory(test_dir, class_mode='categorical', batch_size=64, target_size=(200, 200))

# Opetetaan malli kuvien avulla (tämä kestää riippuen koneen laskentakyvystä...)
# Tallennetaan myös opetusprosessin historiadata kuvaajien näytämistä varten
epochs=20 
history = model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator), validation_data=test_iterator, validation_steps=len(test_iterator), epochs=epochs)

# Tallennetaan opetettu malli samalla tiedostoon, jotta sitä voidaan käyttää kuvien tunnistamiseen
filename = 'basemodel'
model.save('models/' + filename + '.h5')

## Arvioidaan mallin toimivuus tarkkuuden avulla (= miten hyvin malli tunnistaa test kuvia)
_, accuracy = model.evaluate_generator(test_iterator, steps=len(test_iterator))
print('> Accuracy: %.2f' % (accuracy * 100.0))

## Piirretään opetusprosessin kuvaajat, jotta nähdään miten hyvin malli on oppinut

# Piirretään kuvaaja loss arvolle 
pyplot.subplot(211)
pyplot.title('Cross Entropy Loss')
pyplot.plot(history.history['loss'], color='blue', label='train')
pyplot.plot(history.history['val_loss'], color='orange', label='test')

# Piirretään kuvaaja tarkkuudelle
pyplot.subplot(212)
pyplot.title('Classification Accuracy')
pyplot.plot(history.history['accuracy'], color='blue', label='train')
pyplot.plot(history.history['val_accuracy'], color='orange', label='test')

# Tallenetaan kuvaajat kuvatiedostoon
pyplot.savefig('plots/' + filename + '_plot.png')
pyplot.close()