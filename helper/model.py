#Model Helpers
def get_alexnet(input_image, num_classes,  model_path = '', embedding = False):
  model = AlexNetEmbedding(num_classes) if embedding else alexnet.AlexNetModel(num_classes)
  model(input_image)    
  if model_path != '':
    model.load_weights(model_path)
  return model

def get_restnet(block_sizes, filters, input_image, num_classes, model_path = '', embedding = False):
  if embedding:
    model = ResNetEmbedding(block_sizes, filters, num_classes, use_bottleneck = True, se_factor = 0)
    model.build((1,input_shape[0], input_shape[1], input_shape[2]))
  else:
    model = resnet.ResNet(block_sizes, filters, num_classes, use_bottleneck = True, se_factor = 0)
    model(input_image)
  if model_path != '':
    model.load_weights(model_path)
  model.summary()
  return model

def get_optimizer_sgd(learning_rate, decay_steps, momentum, alpha = 0.0001, nesterov = True):
  lr_schedule = tf.keras.experimental.CosineDecay(initial_learning_rate = learning_rate,
                                                decay_steps = decay_steps,
                                                alpha = alpha)
  optimizer = tf.keras.optimizers.SGD(learning_rate = lr_schedule, momentum = momentum , nesterov = nesterov)
  return optimizer

def get_optimizer_adam():
  optimizer = tf.keras.optimizers.Adam()
  return optimizer

def compile(model, optimizer):
  print("compile {}".format(model.name))
  model.compile(optimizer=optimizer,loss= losses.crossentropy_loss, metrics=['accuracy'])
  model.summary()
  return model

def run_predict(model, name, tr_dataset, val_dataset, save=False):
  catalog = model.predict(tr_dataset)
  query = model.predict(val_dataset)
  if save == True:
    with open(name +"_catalog.txt", 'wb') as pyfile:  
      pickle.dump(catalog, pyfile)
    with open(name +"_query.txt", 'wb') as pyfile:  
      pickle.dump(query, pyfile)
  return catalog, query

def load_predict_from_disk(name):
  catalog = pickle.load(open(name +"_catalog.txt", "rb" ))
  query = pickle.load(open(name +"_query.txt", "rb" ))
  return catalog, query