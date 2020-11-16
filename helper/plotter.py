#plotters helpers
def plotAccuracyLoss(name, training_file):
    """Imprime la comparativa entre Accuracy y Loss para el entrenamiento de un modelo dado
      Parámetros:
        name -- Nombre del modelo 
        training_file -- ubicacion del archivo que quiere ser pintado
      Excepciones:
      FileNotFoundError -- Si el archivo historico no ha sido encontrado
      """
    with open(training_file, 'rb') as handle: 
      tr = pickle.load(handle)
    plt.figure(figsize=(20,5))
    plt.suptitle(name)
    plt.subplot(1,2,2)
    plt.xlabel('# Epocas')
    plt.plot(tr['accuracy'], label ='train_accuracy') 
    plt.plot(tr['val_accuracy'], label ='val_accuracy')
    plt.legend(loc="lower right", title="Accuracy", frameon=False)
    plt.subplot(1,2,1)
    plt.xlabel('# Epocas')
    plt.plot(tr['loss'], label ='train_loss')
    plt.plot(tr['val_loss'], label ='val_loss')
    plt.legend(loc="upper right", title="Loss", frameon=False)
    plt.show()

def plotMultipleAccuracyLoss(name, training_history):
    plt.figure(figsize=(20,5))
    plt.suptitle(name)
    for training in training_history:
      model = training[0]
      filename = training[1]
      with open(filename, 'rb') as handle: 
        tr = pickle.load(handle)
      plt.subplot(1,2,1)
      plt.plot(tr['val_loss'], label = model + ' loss')
      plt.subplot(1,2,2)
      plt.plot(tr['val_accuracy'], label = model + ' accuracy')
    plt.subplot(1,2,1)
    plt.xlabel('Época')
    plt.legend(loc="upper right", title="Loss", frameon=False)
    plt.subplot(1,2,2)
    plt.xlabel('Época')
    plt.legend(loc="lower right", title="Accuracy", frameon=False)
    plt.show()

def plotPredictions(model, classes, elements, mean_image, target_size):
        plt.figure(figsize=(13,13))
        true_positive = 0
        size = len(elements)
        i = 0
        for index,item in enumerate(elements):
            label = item[0]
            real = item[1]
            filename = item[2]
            image1 = imgproc.process_image(data.read_image(filename, 3), target_size)
            image2 = image1 - mean_image
            image = tf.expand_dims(image2, 0)        
            pred = model.predict(image)
            pred = pred[0]
            pred = np.exp(pred - max(pred))
            pred = pred / np.sum(pred)            
            prediction = classes[np.argmax(pred)]
            i = 1 if i >= 15 else i + 1
            plt.subplot(size/5,5,i)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(image1)
            color='red'
            if real == prediction :
              color='blue'
              true_positive += 1
            title= "R: {0} vs P: {1} ".format(real, prediction)
            #print(title)
            plt.xlabel(title, color=color, fontsize=15)
        
        resume = "Se han acertado {0} de {1}".format(true_positive,size)
        plt.suptitle(resume)
        plt.tight_layout()
        plt.show()